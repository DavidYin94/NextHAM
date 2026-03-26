import os
import re
import argparse
import torch
import numpy as np
import glob
from scipy.linalg import block_diag

# ===== Custom Modules =====
# Note: Keep 'scripts.' if outwf.py and read_hs_data.py are inside a 'scripts' folder.
from src.read_hs_data import HamiltonianDataReader
from src.outwf import output_wfc 
from tg_src.e3modules import e3TensorDecomp
from dataset_nano import config_set_target

class AttributeDict(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"No such attribute: {name}")
    def __setattr__(self, name, value):
        self[name] = value

def get_fermi(log_path):
    """Extract Fermi energy from ABACUS SCF log."""
    with open(log_path, 'r') as f:
        for line in f:
            if 'EFERMI' in line:
                match = re.search(r'EFERMI\s*=\s*([+-]?[0-9\.Ee-]+)', line)
                if match:
                    return float(match.group(1))
    print("Warning: Fermi energy not found in log, using 0.0 as default.")
    return 0.0

def get_args_parser():
    parser = argparse.ArgumentParser('Data Pipeline for NextHAM', add_help=False)
    # Target and network params required for e3TensorDecomp
    parser.add_argument("--target", type=str, default='hamiltonian')
    parser.add_argument("--target-blocks-type", type=str, default='all')
    parser.add_argument("--no-parity", action='store_true')
    parser.add_argument("--convert-net-out", action='store_true')
    
    # Custom args for this pipeline
    parser.add_argument('--simulate-dir', type=str, default='./simulate', help='Directory containing case-x folders')
    parser.add_argument('--output-dir', type=str, default='./processed_data', help='Directory to save final .pth files')
    return parser 

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

def process_single_case(case_dir, case_id, out_dir, construct_kernel):
    """Process a single case: R-space + K-space -> Tensor formatting -> E(3) Decomp -> Save."""
    print(f"\n========== Processing: {case_id} ==========")
    
    # --- 1. Path configurations ---
    stru_file = os.path.join(case_dir, "abacus_scf/STRU")
    weak_file = os.path.join(case_dir, "abacus_geths/OUT.ABACUS/hrs1_nao.csr")
    overlap_file = os.path.join(case_dir, "abacus_geths/OUT.ABACUS/srs1_nao.csr")
    label_file = os.path.join(case_dir, "abacus_scf/OUT.ABACUS/data-HR-sparse_SPIN0.csr")
    scf_log = os.path.join(case_dir, "abacus_scf/OUT.ABACUS/running_scf.log")
    
    temp_dir = os.path.join(out_dir, f"temp_{case_id}")
    os.makedirs(temp_dir, exist_ok=True)

    # --- 2. Extract R-space features ---
    print(">>> Step A: Extracting ABACUS R-space matrices...")
    reader = HamiltonianDataReader(
        usage="train", nspin=4, out_path=temp_dir, stru_file=stru_file, 
        weak_file=weak_file, overlap_file=overlap_file, label_file=label_file
    )
    reader.read_stru()
    sample_data = reader.read_data() 
    
    input_data, delta_H_list = sample_data
    H0, overlap_tensor, mask_tensor, edge_vec, edge_src, edge_dst, ele_list, _ = input_data
    delta_H = delta_H_list[0]

    # --- 3. Extract K-space wavefunctions ---
    print(">>> Step B: Calculating K-space wavefunctions...")
    fermi_energy = get_fermi(scf_log)
    wf_processor = output_wfc(
        nspin=4, stru_file=stru_file, hr=label_file, sr=overlap_file, 
        ecut=10.0, fermi_energy=fermi_energy, mp_grid=np.array([4, 4, 4]), 
        save_path=temp_dir
    )
    wf_processor.read_stru()
    wf_processor.output_wfc() 
    
    wfc_data = torch.load(os.path.join(temp_dir, "wfc.pth"), weights_only=True)
    lattice_vector_torch, position_torch, tot_basis_num_torch, band_cut_index_torch, tot_kunm_torch, kpt_torch, eigenvectors_enlager_torch = wfc_data

    # --- 4. Tensor Reconstruction & E(3) Decomposition ---
    print(">>> Step C: Tensor reconstruction and E(3) decomposition (H0_ds)...")
    
    # Convert Ry to eV (1 Ry = 13.6056980659 eV)
    H0_raw = H0 * 13.6056980659
    delta_H_raw = delta_H * 13.6056980659
    overlap_tensor_raw = overlap_tensor
    mask_tensor_raw = mask_tensor

    # Spin-orbital decoupling: reshape 54x54 into 2(spin) x 27(orb) x 2(spin) x 27(orb)
    H0_5d = H0_raw.reshape((H0_raw.shape[0], 2, 27, 2, 27))
    overlap_tensor_5d = overlap_tensor_raw.reshape((overlap_tensor_raw.shape[0], 2, 27, 2, 27))
    mask_tensor_5d = mask_tensor_raw.reshape((mask_tensor_raw.shape[0], 2, 27, 2, 27))
    delta_H_5d = delta_H_raw.reshape((delta_H_raw.shape[0], 2, 27, 2, 27))

    H0_convert_list, overlap_tensor_convert_list, mask_tensor_convert_list, delta_H_convert_list = [], [], [], []
    
    # Orbital partition for 4s2p2d1f
    ls = [1, 1, 1, 1, 3, 3, 5, 5, 7] 

    for d1 in [0, 1]: # Spin indices
        for d2 in [0, 1]:
            delta_H_dp_list, overlap_list, mask_list, descriptor_list = [], [], [], []
            a = 0
            for i in ls:
                b = 0
                for j in ls:
                    delta_H_dp_list.append(delta_H_5d[:, d1, a:a+i, d2, b:b+j].reshape(delta_H_5d.shape[0], -1))
                    overlap_list.append(overlap_tensor_5d[:, d1, a:a+i, d2, b:b+j].reshape(overlap_tensor_5d.shape[0], -1))
                    mask_list.append(mask_tensor_5d[:, d1, a:a+i, d2, b:b+j].reshape(mask_tensor_5d.shape[0], -1))
                    descriptor_list.append(H0_5d[:, d1, a:a+i, d2, b:b+j].reshape(H0_5d.shape[0], -1))
                    b += j
                a += i
            H0_convert_list.append(torch.cat(descriptor_list, dim=-1).reshape((-1, 1, 27*27)))
            overlap_tensor_convert_list.append(torch.cat(overlap_list, dim=-1).reshape((-1, 1, 27*27)))
            mask_tensor_convert_list.append(torch.cat(mask_list, dim=-1).reshape((-1, 1, 27*27)))
            delta_H_convert_list.append(torch.cat(delta_H_dp_list, dim=-1).reshape((-1, 1, 27*27)))

    # Concatenate along Spin channel
    H0_final = torch.cat(H0_convert_list, dim=1)
    overlap_tensor_final = torch.cat(overlap_tensor_convert_list, dim=1)
    mask_tensor_final = torch.cat(mask_tensor_convert_list, dim=1)
    delta_H_dp_final = torch.cat(delta_H_convert_list, dim=1)

    edge_vec_final = edge_vec.reshape(edge_vec.shape[0], -1)
    edge_src_final = edge_src.reshape(-1)
    edge_dst_final = edge_dst.reshape(-1)

    # Extract E(3) equivariant features
    H0_ds = construct_kernel.get_net_out(H0_final)

    # --- 5. Save Final Packaged Tensor ---
    final_save_path = os.path.join(out_dir, f"{case_id}.pth")
    torch.save([
        final_save_path, lattice_vector_torch, position_torch, tot_basis_num_torch, 
        band_cut_index_torch, tot_kunm_torch, kpt_torch, eigenvectors_enlager_torch, 
        H0_ds, H0_final, overlap_tensor_final, mask_tensor_final, edge_vec_final, 
        edge_src_final, edge_dst_final, ele_list, case_id, delta_H_dp_final, 
        H0_raw, overlap_tensor_raw, mask_tensor_raw, delta_H_raw
    ], final_save_path)

    efermi_path = os.path.join(out_dir, f"efermi.pth")
    torch.save(fermi_energy, efermi_path)

    os.system("rm -rf "+temp_dir)
    
    print(f">>> Successfully generated network input: {final_save_path}")
    return final_save_path

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Initializing E(3) tensor decomposer...")
    _, construct_kernel = get_hamiltonian_size(args, spinful=True)
    
    case_dirs = sorted(glob.glob(os.path.join(args.simulate_dir, "case-*")))
    if not case_dirs:
        print(f"No case folders found in {args.simulate_dir}.")
        return

    processed_files = []
    for case_dir in case_dirs:
        case_id = os.path.basename(case_dir)
        pth_path = process_single_case(case_dir, case_id, args.output_dir, construct_kernel)
        processed_files.append(pth_path)

    # Generate data.txt containing all absolute paths
    dataset_txt_path = os.path.join(args.output_dir, "data.txt")
    with open(dataset_txt_path, "w") as f:
        for pth_path in processed_files:
            f.write(os.path.abspath(pth_path) + "\n")
            
    print(f"\nAll tasks completed! {len(processed_files)} paths written to {dataset_txt_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Data Pipeline for NextHAM', parents=[get_args_parser()])
    args = parser.parse_args()  
    main(args)