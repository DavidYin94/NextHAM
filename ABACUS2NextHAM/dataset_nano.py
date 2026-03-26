from torch_geometric.data import InMemoryDataset
import time
import torch
from tg_src.utils import orbital_analysis, process_targets
from tg_src.e3modules import e3TensorDecomp
from e3nn.o3 import Irreps
import json
import os

class DatasetInfo:
    
    def __init__(self, spinful, index_to_Z, orbital_types):
        
        self.spinful = spinful
        if isinstance(index_to_Z, list):
            self.index_to_Z = torch.tensor(index_to_Z)
        elif isinstance(index_to_Z, torch.Tensor):
            self.index_to_Z = index_to_Z
        else:
            raise NotImplementedError
        self.orbital_types = orbital_types
        
        self.Z_to_index = torch.full((118,), -1, dtype=torch.int64)
        self.Z_to_index[self.index_to_Z] = torch.arange(len(index_to_Z))

    @classmethod
    def from_dataset(cls, info):
        return cls(info['spinful'], info['index_to_Z'], info['orbital_types'])
    
    
    def __eq__(self, __o) -> bool:
        if __o.__class__ != __class__:
            raise ValueError
        a = __o.spinful == self.spinful
        b = torch.all(__o.index_to_Z == self.index_to_Z)
        c = __o.orbital_types == self.orbital_types
        return a * b * c

class NetOutInfo:
    
    def __init__(self, target_blocks, dataset_info: DatasetInfo):
        self.target_blocks = target_blocks
        self.dataset_info = dataset_info
        self.blocks, self.js, self.slices = process_targets(dataset_info.orbital_types, 
                                                            dataset_info.index_to_Z, 
                                                            target_blocks)

    def save_json(self, src_dir):
        if os.path.isfile(os.path.join(src_dir, 'dataset_info.json')):
            dataset_info_o = DatasetInfo.from_json(src_dir)
            assert self.dataset_info == dataset_info_o
        else:
            self.dataset_info.save_json(src_dir)
        with open(os.path.join(src_dir, 'target_blocks.json'), 'w') as f:
            json.dump(self.target_blocks, f)
    
    @classmethod
    def from_json(cls, src_dir):
        dataset_info = DatasetInfo.from_json(src_dir)
        with open(os.path.join(src_dir, 'target_blocks.json'), 'r') as f:
            target_blocks = json.load(f)
        return cls(target_blocks, dataset_info)
    
    def merge(self, other):
        assert other.__class__ == __class__
        assert self.dataset_info == other.dataset_info
        self.target_blocks.extend(other.target_blocks)
        self.blocks.extend(other.blocks)
        self.js.extend(other.js)
        length = self.slices.pop()
        for i in other.slices:
            self.slices.append(i + length)

    def __eq__(self, __o) -> bool:
        if __o.__class__ != __class__:
            raise ValueError
        flag = True
        for k in self.__dict__.keys():
            flag *= getattr(self, k) == getattr(__o, k)
        return flag
    
def set_target(orbital_types, index_to_Z, spinful, conf, output_file):
    atom_orbitals = {}
    for Z, orbital_type in zip(index_to_Z, orbital_types):
        atom_orbitals[str(Z.item())] = orbital_type
    

    # get target section from config file
    target = conf.target
    tbt = conf.target_blocks_type
    assert len(tbt) > 0, 'Invalid target_blocks_type'
    tbt0 = tbt[0].lower()
    default_target_blocks = None
    sep = None
    element_pairs = eval(sep) if sep else None
    no_parity = conf.no_parity
    
    target_blocks, net_out_irreps, irreps_post_edge = orbital_analysis(atom_orbitals, tbt0, spinful, targets=default_target_blocks, element_pairs=element_pairs, no_parity=no_parity, verbose=output_file)
                                                                       
    if conf.convert_net_out:
            net_out_irreps = Irreps(net_out_irreps).sort().irreps.simplify()
    return target_blocks, irreps_post_edge, net_out_irreps   

def config_set_target(dataset_info, conf, verbose=None):
                                                       
    o, i, s = dataset_info.orbital_types, dataset_info.index_to_Z, dataset_info.spinful
    target_blocks, irreps_post_edge, net_out_irreps = set_target(o, i, s, conf, verbose)
    net_out_info = NetOutInfo(target_blocks, dataset_info)

    return target_blocks, irreps_post_edge, net_out_irreps, net_out_info


class nanotube_weak(InMemoryDataset):
    def __init__(self, data_file: str):

        self.data_file = data_file
        self.transform = None
        self.__indices__ = None
        self.__data_list__ = None
        self._indices = None
        self._data_list = None

        
        begin = time.time()
        loaded_data = torch.load(self.data_file)
        self.data, self.slices, self.info = loaded_data
        # print(self.data)
        # print(self.slices)
        # print(self.info)
        # exit(0)
        
        print(f'Finish loading the processed {len(self)} structures (spinful: {self.info["spinful"]}, '
            f'the number of atomic types: {len(self.info["index_to_Z"])}), cost {time.time() - begin:.2f} seconds')
        
    
    def set_mask(self, targets, del_Aij=True, convert_to_net=False):
        begin = time.time()
        print("\nSetting mask for dataset...")
        
        spinful = self.info['spinful']
        
        dtype = torch.get_default_dtype()
        if spinful:
            if dtype == torch.float32:
                dtype = torch.complex64
            elif dtype == torch.float64:
                dtype = torch.complex128
            else:
                raise ValueError(f'Unsupported dtype: {dtype}')
        
        equivariant_blocks, out_js_list, out_slices = process_targets(self.info['orbital_types'], self.info["index_to_Z"], targets)
        if convert_to_net:
            construct_kernel = e3TensorDecomp(None, out_js_list, torch.get_default_dtype(), spinful=spinful, if_sort=True) # todo: dtype
        
        atom_num_orbital = [sum(map(lambda x: 2 * x + 1,atom_orbital_types)) for atom_orbital_types in self.info['orbital_types']]

        data_list_mask = []
        for data in self:
            assert data.spinful == spinful
            if data.Aij is not None:
                if not torch.all(data.Aij_mask):
                    raise NotImplementedError("Not yet have support for graph radius including Aij without calculation")

            # label of each edge is a vector which is each target H block flattened and concatenated
            if spinful:
                label = torch.zeros(data.num_edges, 4, out_slices[-1], dtype=dtype)
                mask = torch.zeros(data.num_edges, 4, out_slices[-1], dtype=torch.int8)
            else:
                label = torch.zeros(data.num_edges, out_slices[-1], dtype=dtype)
                mask = torch.zeros(data.num_edges, out_slices[-1], dtype=torch.int8)

            atomic_number_edge_i = data.x[data.edge_index[0]]
            atomic_number_edge_j = data.x[data.edge_index[1]]


            for index_out, equivariant_block in enumerate(equivariant_blocks):
                for N_M_str, block_slice in equivariant_block.items():
                    condition_atomic_number_i, condition_atomic_number_j = map(lambda x: self.info["Z_to_index"][int(x)], N_M_str.split())
                    condition_slice_i = slice(block_slice[0], block_slice[1])
                    condition_slice_j = slice(block_slice[2], block_slice[3])
                    if spinful:
                        condition_slice_i_ds = slice(atom_num_orbital[condition_atomic_number_i] + block_slice[0],
                                                      atom_num_orbital[condition_atomic_number_i] + block_slice[1]) # ds = down spin
                        condition_slice_j_ds = slice(atom_num_orbital[condition_atomic_number_j] + block_slice[2],
                                                     atom_num_orbital[condition_atomic_number_j] + block_slice[3])
                    if data.Aij is not None:
                        out_slice = slice(out_slices[index_out], out_slices[index_out + 1])
                        condition_index = torch.where(
                            (atomic_number_edge_i == condition_atomic_number_i)
                            & (atomic_number_edge_j == condition_atomic_number_j)
                        )
                        if spinful:
                            # noncollinear spin block order:
                            # 0(uu) 1(ud)
                            # 2(du) 3(dd)
                            label[condition_index[0], 0, out_slice] += data.Aij[:, condition_slice_i, condition_slice_j].reshape(data.num_edges, -1)[condition_index]
                            label[condition_index[0], 1, out_slice] += data.Aij[:, condition_slice_i, condition_slice_j_ds].reshape(data.num_edges, -1)[condition_index]
                            label[condition_index[0], 2, out_slice] += data.Aij[:, condition_slice_i_ds, condition_slice_j].reshape(data.num_edges, -1)[condition_index]
                            label[condition_index[0], 3, out_slice] += data.Aij[:, condition_slice_i_ds, condition_slice_j_ds].reshape(data.num_edges, -1)[condition_index]
                            mask[condition_index[0], :, out_slice] += 1
                        else:
                            label[condition_index[0], out_slice] += data.Aij[:, condition_slice_i, condition_slice_j].reshape(data.num_edges, -1)[condition_index]
                            mask[condition_index[0], out_slice] += 1
            if del_Aij:
                del data.Aij_mask
            if data.Aij is not None:
                if convert_to_net:
                    label = construct_kernel.get_net_out(label)
                data.label = label
                assert torch.all((mask == 1) | (mask == 0)), 'Some blocks are required to predict multiple times'
                mask = mask.bool()
                if spinful and convert_to_net:
                    mask = construct_kernel.convert_mask(mask)
                data.mask = mask
                if del_Aij:
                    del data.Aij
            data_list_mask.append(data)

        self.__indices__ = None
        self.__data_list__ = None
        self._indices = None
        self._data_list = None
        data, slices = self.collate(data_list_mask)
        self.data, self.slices = data, slices
        print(f"Finished setting mask for dataset, cost {time.time() - begin:.2f} seconds")

        return out_js_list, out_slices