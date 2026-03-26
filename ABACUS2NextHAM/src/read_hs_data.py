from typing import Optional, Union, List, Tuple, Dict, Any
from pathlib import Path
import sys
import os

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.linalg import block_diag
from concurrent.futures import ThreadPoolExecutor

try:
    from ase.io import read
    from ase.atoms import Atoms
    from ase.neighborlist import NeighborList
except ImportError:
    print("ASE library not found. Please install it using `pip install ase`.")
    pass

try:
    from timing_utils import time_execution
except ImportError:
    # If timing_utils is not found (e.g. running script directly), try adding parent dir
    sys.path.append(str(Path(__file__).parent.parent))
    try:
        from timing_utils import time_execution
    except ImportError:
        # Fallback decorator if import fails
        def time_execution(func):
            return func


class HamiltonianDataReader:
    """
    Reads and processes Hamiltonian and Overlap matrix data from ABACUS calculations.
    """

    ORB_ORIGIN: Dict[str, int] = {
        'H': 5, 'He': 5, 'Li': 7, 'Be': 7, 'B': 13, 'C': 13, 'N': 13, 'O': 13,
        'F': 13, 'Ne': 13, 'Na': 15, 'Mg': 15, 'Al': 13, 'Si': 13, 'P': 13,
        'S': 13, 'Cl': 13, 'Ar': 13, 'K': 15, 'Sc': 27, 'V': 27, 'Fe': 27,
        'Co': 27, 'Ni': 27, 'Cu': 27, 'Zn': 27, 'Ga': 25, 'Ge': 25, 'Br': 13,
        'Y': 27, 'Nb': 27, 'Mo': 27, 'Pd': 25, 'Ag': 27, 'Cd': 27, 'In': 25,
        'Sn': 25, 'Sb': 25, 'Te': 25, 'I': 13, 'Xe': 13, 'Hf': 27, 'Ta': 27,
        'Re': 27, 'Pt': 27, 'Au': 27, 'Hg': 27, 'Tl': 25, 'Pb': 25, 'Bi': 25,
        'Ca': 15, 'Ti': 27, 'Cr': 27, 'Mn': 27, 'Kr': 13, 'Rb': 15, 'Sr': 15,
        'Zr': 27, 'Tc': 27, 'Ru': 27, 'Rh': 27, 'Cs': 15, 'Ba': 15, 'W': 27,
        'Os': 27, 'Ir': 27, 'As': 13, 'Se': 13
    }

    def __init__(self,
                 usage: str,
                 nspin: int,
                 out_path: Union[str, Path],
                 stru_file: Union[str, Path],
                 weak_file: Union[str, Path],
                 overlap_file: Union[str, Path],
                 label_file: Optional[Union[str, Path]] = None,
                 ):
        """
        Initialize the HamiltonianDataReader.

        Args:
            usage (str): 'inference' or 'train'.
            nspin (int): Number of spins (1 or 4).
            out_path (Union[str, Path]): Output directory path.
            stru_file (Union[str, Path]): Path to the structure file.
            weak_file (Union[str, Path]): Path to the weak label file.
            overlap_file (Union[str, Path]): Path to the overlap matrix file.
            label_file (Optional[Union[str, Path]]): Path to the strong label file (required for training).
        """
        self.usage = usage
        self.nspin = nspin
        self.out_path = Path(out_path)
        self.stru_file = Path(stru_file)
        self.weak_file = Path(weak_file)
        self.overlap_file = Path(overlap_file)
        if usage != 'inference':
            self.label_file = Path(label_file) if label_file is not None else None
        else:
            self.label_file = None

        self.atoms: Optional[Atoms] = None
        self.stru_dim: int = 0
        self.index_relation: Dict[int, List[Any]] = {}
        
        self.unify_orb_num = 27
        if self.nspin == 4:
            self.unify_orb_num = 2 * self.unify_orb_num
            
        self.index_list: List[np.ndarray] = []
        self.ele_list: List[Tuple[str, str]] = []
        self.matrix_list: List[np.ndarray] = []
        self.data: List[Any] = []
        self.label: Optional[List[torch.Tensor]] = None
        
        self.abacus2deeph: Dict[int, np.ndarray] = {}
        self._init_transforms()

    def _init_transforms(self):
        """Initializes transformation matrices."""
        # Determine the dtype based on nspin
        self.abacus2deeph[0] = np.eye(1)
        self.abacus2deeph[1] = np.eye(3)[[1, 2, 0]]
        self.abacus2deeph[2] = np.eye(5)[[0, 3, 4, 1, 2]]
        self.abacus2deeph[3] = np.eye(7)[[0, 1, 2, 3, 4, 5, 6]]
        minus_dict = {
            1: [0, 1],
            2: [3, 4],
            3: [1, 2, 5, 6],
        }
        for k, v in minus_dict.items():
            self.abacus2deeph[k][v] *= -1

    def _get_orbital_patch_pattern(self, orb_type: int) -> List[int]:
        """Returns the orbital patch pattern based on the orbital type identifier."""
        if orb_type == 5:    # 2s1p
            return [1, 1, 0, 0, 1, 1, 1] + [0]*20
        elif orb_type == 7:  # 4s1p
            return [1, 1, 1, 1, 1, 1, 1] + [0]*20
        elif orb_type == 13: # 2s2p1d
            return [1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] + [0]*12
        elif orb_type == 15: # 4s2p1d
            return [1]*15 + [0]*12
        elif orb_type == 25: # 2s2p2d1f
            return [1, 1, 0, 0] + [1]*23
        elif orb_type == 27: # 4s2p2d1f
            return [1]*27
        else:
            return [0]*27 

    @time_execution
    def _get_transform_matrix(self) -> np.ndarray:
        """Generates the transformation matrix based on nspin."""
        if self.nspin == 1:
            orb_list = [0, 0, 0, 0, 1, 1, 2, 2, 3]
        elif self.nspin == 4:
            orb_list = [0, 0, 0, 0, 1, 1, 2, 2, 3, 0, 0, 0, 0, 1, 1, 2, 2, 3]
        else:
            raise ValueError(f"Unsupported nspin value: {self.nspin}")

        return block_diag(*[self.abacus2deeph[l_number] for l_number in orb_list])

    def build_neighbor_pairs_by_R(self, atoms, r_cut: float,) -> Dict[Tuple[int, int, int], List[Tuple[int, int, np.ndarray]]]:
        """
        Build neighbor pairs grouped by R vector using ASE NeighborList.
        Returns: pairs_by_R[(Rx,Ry,Rz)] = [(ii, jj, distance_vec(3,)), ...]
        """
        n = len(atoms)
        cutoffs = [r_cut * 0.5] * n
        nl = NeighborList(cutoffs, skin=0.0, bothways=True, self_interaction=True)
        nl.update(atoms)

        cell = atoms.cell
        pos = atoms.positions
        pairs_by_R: Dict[Tuple[int, int, int], List[Tuple[int, int, np.ndarray]]] = {}
        for ii in range(n):
            neigh_idx, neigh_offsets = nl.get_neighbors(ii)  # offsets: integer shift vectors
            for jj, off in zip(neigh_idx, neigh_offsets):
                R = (int(off[0]), int(off[1]), int(off[2]))
                # distance definition: R*cell + rj - ri
                distance_vec = off[0] * cell[0] + off[1] * cell[1] + off[2] * cell[2] + pos[jj] - pos[ii]
                if np.linalg.norm(distance_vec) < r_cut:
                    pairs_by_R.setdefault(R, []).append((int(ii), int(jj), np.array(distance_vec, dtype=float)))
        
        # Sort pairs for consistency
        for R in pairs_by_R:
            pairs_by_R[R].sort(key=lambda x: (x[0], x[1]))
        return pairs_by_R

    @time_execution
    def read_stru(self) -> None: 
        """Reads the structure file and sets up orbital mappings."""
        import time
        t_start = time.time()
        if not self.stru_file.exists():
            raise FileNotFoundError(f"Structure file not found: {self.stru_file}")

        self.atoms = read(str(self.stru_file), format='abacus')
        t_read = time.time()
        print(f"  [Time Detail] Read ASE structure: {t_read - t_start:.6f} s")

        elements = self.atoms.get_chemical_symbols()
        
        self.stru_dim = sum(self.ORB_ORIGIN.get(e, 0) for e in elements)
        if self.nspin == 4:
            self.stru_dim *= 2
        
        t_dim = time.time()
        print(f"  [Time Detail] Calc dimensions: {t_dim - t_read:.6f} s")

        count_matrix_dim = -1
        self.index_relation = {}
        
        for i, element in enumerate(elements):
            orb_type = self.ORB_ORIGIN.get(element, 0)
            base_patch = self._get_orbital_patch_pattern(orb_type)
            
            if self.nspin != 4:
                orbital_patch = base_patch
            else:
                orbital_patch = []
                for val in base_patch:
                    orbital_patch.extend([val, val])

            orbital_index = []
            for val in orbital_patch:
                count_matrix_dim += val
                orbital_index.append(count_matrix_dim)
            
            self.index_relation[i] = [element, orbital_patch, orbital_index]

        t_end = time.time()
        print(f"  [Time Detail] Process orbital index loop: {t_end - t_dim:.6f} s")

    @time_execution
    def _read_all_matrices(self, file_path: Path) -> Dict[Tuple[int, int, int], csr_matrix]:
        """Reads the entire sparse matrix file into a dictionary of sparse CSR matrices."""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Auto-detect binary or text mode
        is_binary = False
        with open(file_path, 'rb') as f:
            header = f.read(4)
            if header != b'STEP':
                is_binary = True

        matrices = {}
        if not is_binary:
            with open(file_path, 'r') as f:
                # Skip header
                f.readline()
                # Line 1: Basis num
                line = f.readline()
                if not line: return {}
                basis_num = int(line.split()[-1])
                if basis_num != self.stru_dim:
                    print(f"Dimension mismatch in {file_path}: Expected {self.stru_dim}, got {basis_num}")
                    sys.exit(2)
                
                # Line 2: R num
                line = f.readline()
                if not line: return {}
                # r_num = int(line.split()[-1])
                
                while True:
                    line = f.readline()
                    if not line: break
                    line = line.strip()
                    if not line: continue
                        
                    parts = line.split()
                    if len(parts) < 4: continue
                        
                    rx, ry, rz = int(parts[0]), int(parts[1]), int(parts[2])
                    data_size = int(parts[3])
                    
                    if data_size == 0:
                        matrices[(rx, ry, rz)] = csr_matrix((self.stru_dim, self.stru_dim), dtype=np.complex64 if self.nspin == 4 else float)
                        continue

                    # Read Data
                    data_line = f.readline()
                    col_idx_line = f.readline()
                    indptr_line = f.readline()
                    
                    if self.nspin != 4:
                        data = np.fromstring(data_line, sep=' ', dtype=float)
                    else:
                        cleaned = data_line.replace('(', ' ').replace(')', ' ').replace(',', ' ')
                        vals = np.fromstring(cleaned, sep=' ', dtype=float)
                        data = vals[0::2] + 1j * vals[1::2]
                        
                    indices = np.fromstring(col_idx_line, sep=' ', dtype=int)
                    indptr = np.fromstring(indptr_line, sep=' ', dtype=int)
                    
                    mat = csr_matrix((data, indices, indptr), shape=(self.stru_dim, self.stru_dim))
                    matrices[(rx, ry, rz)] = mat
        else:
            import struct
            with open(file_path, 'rb') as f:
                # Binary format: step(int), nlocal(int), nR(int)
                raw_header = f.read(12)
                if not raw_header: return {}
                step, nlocal, nR = struct.unpack('iii', raw_header)
                if nlocal != self.stru_dim:
                    print(f"Dimension mismatch in binary {file_path}: Expected {self.stru_dim}, got {nlocal}")
                    sys.exit(2)
                
                for _ in range(nR):
                    raw_R = f.read(16) # rx, ry, rz, nnz
                    if not raw_R: break
                    rx, ry, rz, nnz = struct.unpack('iiii', raw_R)
                    
                    if nnz == 0:
                        f.read((self.stru_dim + 1) * 4)
                        matrices[(rx, ry, rz)] = csr_matrix((self.stru_dim, self.stru_dim), dtype=np.complex64 if self.nspin == 4 else float)
                        continue
                    
                    if self.nspin == 4:
                        # complex double is 16 bytes
                        data = np.frombuffer(f.read(nnz * 16), dtype=complex)
                    else:
                        data = np.frombuffer(f.read(nnz * 8), dtype=float)
                    
                    indices = np.frombuffer(f.read(nnz * 4), dtype=np.int32)
                    indptr = np.frombuffer(f.read((self.stru_dim + 1) * 4), dtype=np.int32)
                    
                    mat = csr_matrix((data, indices, indptr), shape=(self.stru_dim, self.stru_dim))
                    matrices[(rx, ry, rz)] = mat
            
        return matrices

    @time_execution
    def read_data(self) -> Tuple[List[Any], Optional[List[torch.Tensor]]]:
        """Reads and processes all data files."""
        import time
        t_start = time.time()
        
        # Determine paths based on usage
        if self.usage == 'inference':
            strong_label_path = self.weak_file 
        else:
            strong_label_path = self.label_file
        
        if strong_label_path is None:
             raise ValueError("Strong label path is None, but required for processing logic.")

        # 1. Read all matrices into memory
        # This replaces the line-by-line seeking with a single pass per file
        dict_s = self._read_all_matrices(strong_label_path)
        dict_w = self._read_all_matrices(self.weak_file)
        dict_o = self._read_all_matrices(self.overlap_file)
        
        t_read = time.time()
        print(f"  [Time Detail] Read all matrices: {t_read - t_start:.6f} s")
        
        # Intersection of keys
        keys_s = set(dict_s.keys())
        keys_w = set(dict_w.keys())
        # keys_o = set(dict_o.keys()) # Assume overlap covers these
        
        # r_coor_both = [r for r in keys_s if r in keys_w] 
        # To match original order (sorted by R probably?), let's sort them.
        # Original code: read header of strong, filtered by weak.
        # Here we just take intersection.
        r_coor_both = sorted(list(keys_s.intersection(keys_w)))

        t_coords = time.time()
        print(f"  [Time Detail] Find common coordinates: {t_coords - t_read:.6f} s")

        # 2. Iterate and process (Multi-threaded)
        self.index_list = []
        self.ele_list = []
        self.matrix_list = []
        
        # Pre-calc transformation matrix
        trans_orb_mat_np = self._get_transform_matrix()
        trans_orb_mat_np_T = trans_orb_mat_np.T
        
        symbols = self.atoms.get_chemical_symbols()
        tot_num = len(symbols)
        r_cut = 8.0
        
        # Build neighbor pairs
        pairs_by_R = self.build_neighbor_pairs_by_R(self.atoms, r_cut)
        
        # Pre-fetch orbital indices to avoid dict lookup in loop
        atom_indices_map = {}
        patch_map = {}
        for i in range(tot_num):
            atom_indices_map[i] = np.array(self.index_relation[i][2])
            patch_map[i] = np.array(self.index_relation[i][1])

        # Collect all tasks for parallel processing
        all_tasks = []
        for target_r in r_coor_both:
            if target_r not in dict_w or target_r not in dict_o:
                continue
            mat_s = dict_s[target_r]
            mat_w = dict_w[target_r]
            mat_o = dict_o[target_r]
            valid_pairs = pairs_by_R.get(target_r, [])
            for ii, jj, distance in valid_pairs:
                all_tasks.append((target_r, mat_s, mat_w, mat_o, ii, jj, distance))
        t_tasks = time.time()
        print(f"  [Time Detail] Prepare tasks: {t_tasks - t_coords:.6f} s")
        
        def process_unit(task):
            target_r, mat_s, mat_w, mat_o, ii, jj, distance = task
            
            p_ii = patch_map[ii]
            p_jj = patch_map[jj]
            mask = np.outer(p_ii, p_jj)
            
            if not np.any(mask):
                return None

            dtype = np.complex64 if self.nspin == 4 else float
            temp_data = np.zeros((4, self.unify_orb_num, self.unify_orb_num), dtype=dtype)
            temp_data[3] = mask
            
            idx_ii = atom_indices_map[ii]
            idx_jj = atom_indices_map[jj]
            
            # Efficiently extract submatrix from CSR and convert to dense
            sub_s = mat_s[idx_ii, :][:, idx_jj].toarray()
            sub_w = mat_w[idx_ii, :][:, idx_jj].toarray()
            sub_o = mat_o[idx_ii, :][:, idx_jj].toarray()
            
            temp_data[0] = sub_s * mask
            temp_data[1] = sub_w * mask
            temp_data[2] = sub_o * mask
            
            # Apply Transformation (Batched)
            if self.nspin == 1:
                temp_data[:3] = trans_orb_mat_np @ temp_data[:3] @ trans_orb_mat_np_T
            elif self.nspin == 4:
                # Spin-orbital reshuffling + transformation
                reshaped = temp_data[:3].reshape((3, 27, 2, 27, 2))
                reshaped = reshaped.transpose((0, 2, 1, 4, 3)).reshape((3, 54, 54))
                temp_data[:3] = trans_orb_mat_np @ reshaped @ trans_orb_mat_np_T
            
            temp_label = np.zeros(8, dtype=float)
            temp_label[0:3] = target_r
            temp_label[3] = ii
            temp_label[4] = jj
            temp_label[5:8] = distance
            
            return temp_label, (symbols[ii], symbols[jj]), temp_data
        num_workers = 24 #min(24, (os.cpu_count() or 1))
        print(f"all tasks: {len(all_tasks)}")
        if all_tasks:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                results = list(executor.map(process_unit, all_tasks))
            
            for res in results:
                if res is not None:
                    self.index_list.append(res[0])
                    self.ele_list.append(res[1])
                    self.matrix_list.append(res[2])
        del all_tasks
        del dict_o
        del dict_w
        del dict_s
        t_loop = time.time()
        print(f"  [Time Detail] Main loop processing: {t_loop - t_coords:.6f} s")

        return self._finalize_tensors()

    @time_execution
    def _finalize_tensors(self) -> Tuple[List[Any], Optional[List[torch.Tensor]]]:
        """Convert collected lists to final PyTorch tensors with reduced memory overhead."""
        if not self.matrix_list:
             print("Warning: No data found.")
             return [], None

        # Use torch.stack to avoid doubling memory with np.array(...)
        matrix_tensor = torch.stack([torch.from_numpy(m) for m in self.matrix_list]).to(torch.complex64)
        del self.matrix_list
        print(f"        # Extract matrix_tensor")
        
        label_tensor = matrix_tensor[:, 0].clone() - matrix_tensor[:, 1].clone()
        print(f"        # Extract label")

        descriptor_tensor = matrix_tensor[:, 1].clone() # Weak
        print(f"        # Extract descriptor_tensor")
        
        if self.usage == 'inference':
            overlap_tensor = None
        else:
            overlap_tensor = matrix_tensor[:, 2].clone() # Overlap
        print(f"        # Extract overlap_tensor")
    
        mask_tensor = matrix_tensor[:, 3].clone().real.to(torch.float32)
        print(f"        # Extract mask_tensor")
        
        del matrix_tensor
        
        arr_idx = np.array(self.index_list)
        index_tensor = torch.from_numpy(arr_idx)
        del self.index_list
        
        edge_vec = index_tensor[:, 5:8].to(torch.float32)
        edge_src = index_tensor[:, 3].to(torch.long)
        edge_dst = index_tensor[:, 4].to(torch.long)
        
        del index_tensor
        del self.atoms
        del self.stru_dim
        del self.index_relation
        del self.unify_orb_num
        print(f"        # Extract edge information")
        output_path = self.out_path / "output_inference.pth"
        input_path = os.path.join(self.out_path, "input_inference.pth")
        self.data = [
            descriptor_tensor, 
            overlap_tensor, 
            mask_tensor, 
            edge_vec, 
            edge_src, 
            edge_dst, 
            self.ele_list, 
            str(output_path)
        ]
        self.label = None
        if self.usage == 'inference':
            self.label = None
        else:
            self.label = [label_tensor]
        
        sample = (self.data, self.label)
        torch.save(sample, input_path)
        return sample

def main():
    pass

if __name__ == "__main__":
    main()