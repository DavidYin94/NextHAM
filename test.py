import argparse
import datetime
import itertools
import pickle
import subprocess
import time
import torch
import numpy as np
import random
#torch.autograd.set_detect_anomaly(True)
import sys
#from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset, DataLoader
from tg_src.e3modules import e3TensorDecomp, get_random_R
from output_data_convert import get_hamiltion_data
import gc
import os
from logger import FileLogger
from pathlib import Path
from typing import Iterable, Optional
import copy
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)


import nets
from nets import model_entrypoint

from timm.utils import ModelEmaV2, get_state_dict
from timm.scheduler import create_scheduler

from engine import AverageMeter, compute_stats
from dataset_nano import nanotube_weak, config_set_target, DatasetInfo
from operator import itemgetter
from scipy.linalg import block_diag
from tg_src.graph import Collater


ModelEma = ModelEmaV2

elements_index_info = [
    (1, "H", 1, 1), (2, "He", 18, 1),
    (3, "Li", 1, 2), (4, "Be", 2, 2), (5, "B", 13, 2), (6, "C", 14, 2), 
    (7, "N", 15, 2), (8, "O", 16, 2), (9, "F", 17, 2), (10, "Ne", 18, 2),
    (11, "Na", 1, 3), (12, "Mg", 2, 3), (13, "Al", 13, 3), (14, "Si", 14, 3), 
    (15, "P", 15, 3), (16, "S", 16, 3), (17, "Cl", 17, 3), (18, "Ar", 18, 3),
    (19, "K", 1, 4), (20, "Ca", 2, 4), (21, "Sc", 3, 4), (22, "Ti", 4, 4), 
    (23, "V", 5, 4), (24, "Cr", 6, 4), (25, "Mn", 7, 4), (26, "Fe", 8, 4), 
    (27, "Co", 9, 4), (28, "Ni", 10, 4), (29, "Cu", 11, 4), (30, "Zn", 12, 4), 
    (31, "Ga", 13, 4), (32, "Ge", 14, 4), (33, "As", 15, 4), (34, "Se", 16, 4), 
    (35, "Br", 17, 4), (36, "Kr", 18, 4),
    (37, "Rb", 1, 5), (38, "Sr", 2, 5), (39, "Y", 3, 5), (40, "Zr", 4, 5), 
    (41, "Nb", 5, 5), (42, "Mo", 6, 5), (43, "Tc", 7, 5), (44, "Ru", 8, 5), 
    (45, "Rh", 9, 5), (46, "Pd", 10, 5), (47, "Ag", 11, 5), (48, "Cd", 12, 5), 
    (49, "In", 13, 5), (50, "Sn", 14, 5), (51, "Sb", 15, 5), (52, "Te", 16, 5), 
    (53, "I", 17, 5), (54, "Xe", 18, 5),
    (55, "Cs", 1, 6), (56, "Ba", 2, 6), 
    (72, "Hf", 4, 6), (73, "Ta", 5, 6), (74, "W", 6, 6), (75, "Re", 7, 6), 
    (76, "Os", 8, 6), (77, "Ir", 9, 6), (78, "Pt", 10, 6), (79, "Au", 11, 6), 
    (80, "Hg", 12, 6), (81, "Tl", 13, 6), (82, "Pb", 14, 6), (83, "Bi", 15, 6), 
    (84, "Po", 16, 6), (85, "At", 17, 6), (86, "Rn", 18, 6)
]

ele_dict = {}

for tuple_ele in elements_index_info:
    if not tuple_ele[1] in ele_dict:
        ele_dict[tuple_ele[1]] = int(tuple_ele[0])-1

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)    
    torch.manual_seed(seed)    
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False    
    os.environ["PYTHONHASHSEED"] = str(seed)

def get_args_parser():
    parser = argparse.ArgumentParser('Testing general equivariant networks for electronic-structure prediction', add_help=False)
    parser.add_argument('--output-dir', type=str, default=None)
    # network architecture
    parser.add_argument('--model-name', type=str, default='graph_attention_transformer_nonlinear_l2_md17')
    parser.add_argument('--input-irreps', type=str, default=None)
    parser.add_argument('--radius', type=float, default=8.0)
    parser.add_argument('--num-basis', type=int, default=128)
    # training hyper-parameters
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=24)
    # regularization
    parser.add_argument('--drop-path', type=float, default=0.0)
    # optimizer (timm)
    parser.add_argument('--opt', default='adam', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adam"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-3,
                        help='weight decay (default: 5e-3)')
    # learning rate schedule parameters (timm)
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-6)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=0, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')
    # logging
    parser.add_argument("--print-freq", type=int, default=20)
    # task and dataset
    parser.add_argument("--target", type=str, default='hamiltonian')
    parser.add_argument("--target-blocks-type", type=str, default='all')
    parser.add_argument("--no-parity", action='store_true')
    parser.add_argument("--convert-net-out", action='store_true')
    parser.add_argument("--data-path", type=str, default='datasets/md17')
    parser.add_argument("--weakdata-path", type=str, default='datasets/md17')
    parser.add_argument("--data-ratio", type=float, default=0.1)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--is-accurate-label", action='store_true')
    parser.add_argument("--with-trace", action='store_true')
    parser.add_argument("--trace-out-len", type=int, default=25)
    parser.add_argument("--select-stru-id", type=int, default=-1)
    parser.add_argument("--start-layer", type=int, default=0)

    parser.add_argument('--compute-stats', action='store_true', dest='compute_stats')
    parser.set_defaults(compute_stats=False)
    parser.add_argument('--test-interval', type=int, default=10, 
                        help='epoch interval to evaluate on the testing set')
    parser.add_argument('--test-max-iter', type=int, default=1000, 
                        help='max iteration to evaluate on the testing set')
    parser.add_argument('--energy-weight', type=float, default=0.2)
    parser.add_argument('--force-weight', type=float, default=0.8)
    # random
    parser.add_argument("--seed", type=int, default=1)
    # data loader config
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)
    # evaluation
    parser.add_argument('--checkpoint-path1', type=str, default=None)
    parser.add_argument('--checkpoint-path2', type=str, default=None)
    parser.add_argument('--checkpoint-path3', type=str, default=None)
    parser.add_argument('--checkpoint-path4', type=str, default=None)

    parser.add_argument('--evaluate', action='store_true', dest='evaluate')
    parser.set_defaults(evaluate=False)
    return parser 
    
def reverse_transform_matrix(tensor, ls):
    C = tensor.shape[0]
    total_HW = sum(ls)
    original = torch.zeros((C, total_HW, total_HW), dtype=tensor.dtype, device=tensor.device)
    total_idx = 0 
    a = 0
    for i in ls:
        b = 0
        for j in ls:
            original[:, a:a+i, b:b+j] = tensor[:, total_idx:total_idx+i*j].reshape((C, i, j))
            b += j
            total_idx += i*j
        a += i
    return original


def convert_label_with_overlap(pred_h, label, overlap):
    Denominator = torch.sum(overlap * torch.conj(overlap))
    Numerator =  torch.real(torch.sum((pred_h-label) * torch.conj(overlap)))
    delta_mu = Numerator/(Denominator+1e-6)
    new_label = label + delta_mu*overlap
    return new_label


class MaskedMAELosswithGuage(torch.nn.Module):
    def __init__(self, threshold_max=100000000, threshold_min=-100000000, factor=1.0):
        super(MaskedMAELosswithGuage, self).__init__()
        self.mae_loss = torch.nn.L1Loss(reduction='none')
        self.threshold_max = threshold_max
        self.threshold_min = threshold_min
        self.factor = factor

    def forward(self, input, target, overlap, mask, cal_new_target = False):
        if cal_new_target:
            target = convert_label_with_overlap(input, target, overlap)
        loss = self.mae_loss(input, target)
        threshold_mask = ((self.threshold_min < target.abs()) & (target.abs() < self.threshold_max)).float()
        combined_mask = mask * threshold_mask
        loss = loss * combined_mask * self.factor
        combined_mask_sum = combined_mask.sum()
        masked_loss = loss.sum() / (combined_mask_sum+1e-7)
        return target, masked_loss.abs()
    

class AttributeDict(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"No such attribute: {name}")

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError:
            raise AttributeError(f"No such attribute: {name}")

def safe(t): 
    return t.detach().cpu().contiguous()


def get_WA_data(WA_data_root):
    root_path = Path(WA_data_root)
    results = {}
    for file_path in root_path.rglob("*.pth"):
        if file_path.is_file():
            absolute_path = str(file_path.resolve())
            parent_name = file_path.parent.name  # 直接通过Path对象获取父目录名[3,5](@ref)
            results[parent_name] = absolute_path    
            # print(parent_name.strip(), absolute_path.strip())
    return results


class Material_Project_Dataset(torch.utils.data.Dataset):
    def __init__(self, mode, construct_kernel, device, dataset_root='/work/home/nextham/NextHAM_V1/datasets/'):
        super().__init__()
        self.mode = mode
        self.construct_kernel = construct_kernel
        self.samples = []
        self.label_norm_tensor = None
        self.descriptor_norm_tensor = None
        self.norm_mask_tensor = None
        time1 = time.time()
        dataset_file = open(dataset_root+mode+'.txt', "r")
        self.file_list = []
        for line in dataset_file.readlines():                          
            self.file_list.append(line.strip())
        print('total load time: ', time.time()-time1)
        print('len of self.samples: ', len(self.file_list))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        return torch.load(file_path, weights_only=True)
    

def get_material_project_dataset(construct_kernel, device):
    """Process and save datasets individually for train, val, test."""
    datasets = {}

    datasets["train"], datasets["val"], datasets["test"] = Material_Project_Dataset('train', construct_kernel, device), Material_Project_Dataset('val', construct_kernel, device), Material_Project_Dataset('test', construct_kernel, device)

    return datasets["train"], datasets["val"], datasets["test"]

def get_hamiltonian_size(args, spinful):
    dataset_info = AttributeDict(spinful= spinful, index_to_Z= torch.Tensor([idx for idx in range(118)]).long(), Z_to_index= torch.Tensor([idx for idx in range(118)]).long(), orbital_types= [[0, 0, 0, 0, 1, 1, 2, 2, 3]])
    _, _, net_out_irreps, net_out_info = config_set_target(dataset_info, args, verbose='target.txt')
    irreps_edge = net_out_irreps
    js = net_out_info.js
    spinful = dataset_info.spinful
    no_parity = args.no_parity
    if_sort = args.convert_net_out
    construct_kernel = e3TensorDecomp(irreps_edge, 
                                    js, 
                                    default_dtype_torch=torch.get_default_dtype(), 
                                    spinful=spinful,
                                    no_parity=no_parity, 
                                    if_sort=if_sort, 
                                    device_torch=torch.device('cpu'))
    return irreps_edge, construct_kernel

def process_worker(q, model_idx, test_dataset, device, model, range_dis):
    try:
        with torch.no_grad():
            model.eval()
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory = False)
            print(f"Test loader length: {len(test_loader)}")        
            for step, data in enumerate(test_loader):
                file_path, _, _, _, _, _, _, _, H0_ds, H0, overlap_tensor, mask_tensor, edge_vec, edge_src, edge_dst, ele_list, mp_stru_name, delta_H_dp, H0_raw, overlap_tensor_raw, mask_tensor_raw, delta_H_raw = data
                file_path, H0_ds, H0, overlap_tensor, mask_tensor, edge_vec, edge_src, edge_dst, delta_H_dp, H0_raw, overlap_tensor_raw, mask_tensor_raw, delta_H_raw = file_path[0], H0_ds[0].to(device, non_blocking=True), H0[0].to(device, non_blocking=True), overlap_tensor[0].to(device, non_blocking=True), mask_tensor[0].to(device, non_blocking=True), edge_vec[0].to(device, non_blocking=True), edge_src.to(torch.int64)[0].to(device, non_blocking=True), edge_dst.to(torch.int64)[0].to(device, non_blocking=True), delta_H_dp[0].to(device, non_blocking=True), H0_raw[0].to(device, non_blocking=True), overlap_tensor_raw[0].to(device, non_blocking=True), mask_tensor_raw[0].to(device, non_blocking=True), delta_H_raw[0].to(device, non_blocking=True)        
                node_num = max(int(max(edge_src)+1), int(max(edge_dst)+1))
                batch = torch.ones((node_num,), dtype=torch.int32).to(device, non_blocking=True)
                node_atom = [-1 for _ in range(node_num)]
                for ele_idx in range(len(ele_list)):
                    node_atom[edge_src[ele_idx]] = ele_dict[ele_list[ele_idx][0][0]]
                node_atom = torch.tensor(node_atom, dtype=torch.long, device=device)
                pred_h_direct_sum, _, _ = model(weak_ham_in = H0_ds,
                                        node_num = node_num,
                                        edge_src = edge_src,
                                        edge_dst = edge_dst, 
                                        edge_vec = edge_vec, 
                                        batch = batch,
                                        node_atom = node_atom,
                                        use_sep = True,
                                        range_dis = range_dis)        
                q.put((mp_stru_name[0], model_idx, safe(pred_h_direct_sum), safe(H0_raw), safe(overlap_tensor_raw), safe(mask_tensor_raw), safe(delta_H_raw)))
    except Exception as e:
        print(f"Process {model_idx} encountered an error: {e}")


def main(args):

    mp.set_start_method("spawn", force=True)
    mp.set_sharing_strategy("file_system")

    _log = FileLogger(is_master=True, is_rank0=True, output_dir=args.output_dir)
    _log.info(args)
    

    ''' Config '''
    irreps_edge, construct_kernel = get_hamiltonian_size(args, spinful=True)


    mean = 0.
    std = 1. 
    _log.info('Training set mean for [energy] training: {}, std: {}\n'.format(mean, std))

    # since dataset needs random 
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)


    ''' Network '''
    create_model = model_entrypoint(args.model_name)

    devices = ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3']

    models = []

    for model_idx in range(4):
        models.append(create_model(irreps_in=args.input_irreps, irreps_edge=irreps_edge,
            radius=args.radius, 
            num_basis=args.num_basis, 
            task_mean=mean, 
            task_std=std, 
            atomref=None,
            start_layer=args.start_layer,
            drop_path_rate=args.drop_path,
            with_trace=args.with_trace,
            trace_out_len=args.trace_out_len,
            use_w2v=False,
            ).to(devices[model_idx]))

    checkpoint_paths = [args.checkpoint_path1, args.checkpoint_path2, args.checkpoint_path3, args.checkpoint_path4]

    for model_idx in range(4):
        checkpoint_path = checkpoint_paths[model_idx]
        if checkpoint_path is not None:
            state_dict = torch.load(checkpoint_path, map_location='cpu')['state_dict']
            models[model_idx].load_state_dict(state_dict)
            print('load pre-trained model')
        else:
            print('no pre-trained model')


    n_parameters = sum(p.numel() for p in models[0].parameters())*5
    _log.info('Number of params: {}'.format(n_parameters))

    ''' Dataset '''
    _, _, test_dataset = get_material_project_dataset(construct_kernel = construct_kernel, device=devices[0])

    _log.info('')
    _log.info('Testing set size:    {}\n'.format(len(test_dataset)))

    ''' Processors '''
    mgr = mp.Manager()
    q = mgr.Queue()
    testing_num = len(test_dataset)
    range_dis = [[0.0, 1.0], [1.0, 2.0], [2.0, 4.0], [4.0, 6.0]]
    for model_idx in range(4):
        p = mp.Process(
            target=process_worker,
            args=(q, model_idx, test_dataset, devices[model_idx], models[model_idx], range_dis[model_idx])
            )
        p.start()
    MAE_metric = MaskedMAELosswithGuage()
    MAE_list = []
    ls = [1, 1, 1, 1, 3, 3, 5, 5, 7]
    buffers = {} 
    total_process_sample = 0 
    file_res_w_root = '/work/home/nextham/NextHAM_V1/test_res.txt'
    file_res_w_obj = open(file_res_w_root, 'w')
    time1 = time.time()
    with torch.no_grad():
        while total_process_sample < testing_num:
            mp_key, model_idx, pred_h_direct_sum, H0_raw, overlap_tensor_raw, mask_tensor_raw, delta_H_raw = q.get()
            print('mp_key, model_idx: ', mp_key, model_idx)
            if mp_key not in buffers:
                buffers[mp_key] = [None, None, None, None]
            buffers[mp_key][model_idx] = pred_h_direct_sum
            if all(x is not None for x in buffers[mp_key]):
                pred_h = torch.sum(torch.stack(buffers[mp_key]), dim=0)
                pred_h = construct_kernel.get_H(pred_h)
                delta_H_pred_real = reverse_transform_matrix(pred_h[:,0,:].real, ls)
                H_gt = delta_H_raw + H0_raw
                H_pred = H0_raw.clone()
                H_pred, H_gt, H0_raw, overlap_tensor_raw, mask_tensor_raw = H_pred.reshape(-1, 2, 27, 2, 27), H_gt.reshape(-1, 2, 27, 2, 27), H0_raw.reshape(-1, 2, 27, 2, 27), overlap_tensor_raw.reshape(-1, 2, 27, 2, 27), mask_tensor_raw.reshape(-1, 2, 27, 2, 27)
                H_pred[:, 0, :, 0, :].real = H_pred[:, 0, :, 0, :].real + delta_H_pred_real
                H_pred[:, 1, :, 1, :].real = H_pred[:, 1, :, 1, :].real + delta_H_pred_real 
                H_pred, H_gt, H0_raw, overlap_tensor_raw, mask_tensor_raw = H_pred.reshape(-1, 54, 54), H_gt.reshape(-1, 54, 54), H0_raw.reshape(-1, 54, 54), overlap_tensor_raw.reshape(-1, 54, 54), mask_tensor_raw.reshape(-1, 54, 54)
                _, mae_H0 = MAE_metric(H0_raw, H_gt, overlap_tensor_raw, mask_tensor_raw, cal_new_target = True)
                _, mae_H_pred = MAE_metric(H_pred, H_gt, overlap_tensor_raw,mask_tensor_raw, cal_new_target = True)
                file_res_w_obj.write(mp_key+' '+str(mae_H0.item())+' '+str(mae_H_pred.item())+'\n')
                file_res_w_obj.flush()
                buffers.pop(mp_key)
                MAE_list.append(mae_H_pred.item())
                total_process_sample += 1
        print('np.mean(MAE_list): ', np.mean(MAE_list))
        print('mean time: ', (time.time()-time1)/total_process_sample)
    file_res_w_obj.close()

    
if __name__ == "__main__":
    set_seed()
    parser = argparse.ArgumentParser('Testing NextHAM on Materials-HAM-SOC', parents=[get_args_parser()])
    args = parser.parse_args()  
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)