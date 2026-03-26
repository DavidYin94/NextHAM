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

from timm.utils import ModelEmaV2, get_state_dict
from timm.scheduler import create_scheduler

from engine import AverageMeter, compute_stats
from dataset_nano import nanotube_weak, config_set_target, DatasetInfo
from operator import itemgetter
from scipy.linalg import block_diag

# from check_equivariance_soc import *

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

def get_args_parser():
    parser = argparse.ArgumentParser('Training general equivariant networks for electronic-structure prediction', add_help=False)
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
    parser.add_argument('--weight-decay', type=float, default=0.0,
                        help='weight decay (default: 5e-3)')

    # learning rate schedule parameters (timm)
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine")')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.0, metavar='PERCENT',
                        help='learning rate noise limit percent (set to 0.0 for off)')
    parser.add_argument('--lr-noise-std', type=float, default=0.0, metavar='STDDEV',
                        help='learning rate noise std-dev (set to 0.0 for off)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--decay-epochs', type=float, default=0, metavar='N',
                        help='not used for cosine scheduler')
    parser.add_argument('--decay-rate', '--dr', type=float, default=1.0, metavar='RATE',
                        help='not used for cosine scheduler')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-6)')
    parser.add_argument('--cooldown-epochs', type=int, default=0, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=0, metavar='N',
                        help='not used for cosine scheduler')

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

def reverse_transform_matrix(tensor, ls):
    # 获取原始通道数
    C = tensor.shape[0]
    # 计算原始张量的高度和宽度（sum(ls)）
    total_HW = sum(ls)
    # 初始化原始形状的张量
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


def Process_Material_Dataset(mode, construct_kernel, src_root='/work/home/nextham/NextHAM_Ca2O3/datasets/', dst_root='/work/home/nextham/NextHAM_Ga_O_More_More_Orbitals/datasets/', save_pth_root = '/work/home/nextham/NextHAM_Ga_O_More_More_Orbitals/raw_data/'):
    dataset_file_r = open(src_root+mode+'.txt', "r")
    dataset_file_w = open(dst_root+mode+'.txt', "w")
    sample_num = 0
    for line in dataset_file_r.readlines():  
        file_path = line.strip()
        sample = torch.load(file_path, weights_only=True)
        input_data, delta_H = sample[0], sample[1]
        H0, overlap_tensor, mask_tensor, edge_vec, edge_src, edge_dst, ele_list, output_path = input_data
        node_num = max(int(max(edge_src)+1), int(max(edge_dst)+1))
        delta_H = delta_H[0]
        H0_raw = H0*13.6
        delta_H_raw = delta_H*13.6
        H0 = H0_raw.reshape((H0_raw.shape[0], 2, 27, 2, 27)) 
        overlap_tensor_raw = overlap_tensor
        overlap_tensor = overlap_tensor_raw.reshape((overlap_tensor_raw.shape[0], 2, 27, 2, 27))
        mask_tensor_raw = mask_tensor
        mask_tensor = mask_tensor_raw.reshape((mask_tensor_raw.shape[0], 2, 27, 2, 27))
        delta_H = delta_H_raw.reshape((delta_H_raw.shape[0], 2, 27, 2, 27))
        H0_convert_list = []
        overlap_tensor_convert_list = []
        mask_tensor_convert_list = []
        delta_H_convert_list = []
        ls = [1, 1, 1, 1, 3, 3, 5, 5, 7]
        for d1 in [0,1]:
            for d2 in [0,1]:
                delta_H_dp_list = []
                overlap_list = []
                mask_list = []
                descriptor_list = []
                a = 0
                for i in ls:
                    b = 0
                    for j in ls:
                        delta_H_dp_list.append(delta_H[:, d1, a:a+i, d2, b:b+j].reshape(delta_H.shape[0], -1))
                        overlap_list.append(overlap_tensor[:, d1, a:a+i, d2, b:b+j].reshape(overlap_tensor.shape[0], -1))
                        mask_list.append(mask_tensor[:, d1, a:a+i, d2, b:b+j].reshape(mask_tensor.shape[0], -1))
                        descriptor_list.append(H0[:, d1, a:a+i, d2, b:b+j].reshape(H0.shape[0], -1))
                        b += j
                    a += i
                H0_convert_list.append(torch.cat(descriptor_list, dim = -1).reshape((-1, 1, 27*27)))
                overlap_tensor_convert_list.append(torch.cat(overlap_list, dim = -1).reshape((-1, 1, 27*27)))
                mask_tensor_convert_list.append(torch.cat(mask_list, dim = -1).reshape((-1, 1, 27*27)))
                delta_H_convert_list.append(torch.cat(delta_H_dp_list, dim = -1).reshape((-1, 1, 27*27)))
        H0 = torch.cat(H0_convert_list, dim = 1)
        overlap_tensor = torch.cat(overlap_tensor_convert_list, dim = 1)
        mask_tensor = torch.cat(mask_tensor_convert_list, dim = 1)
        delta_H_dp = torch.cat(delta_H_convert_list, dim = 1)
        edge_vec, edge_src, edge_dst = edge_vec.reshape(edge_vec.shape[0], -1), edge_src.reshape(-1), edge_dst.reshape(-1)
        H0_ds = construct_kernel.get_net_out(H0)
        file_path_wa = file_path.replace('input_inference_0', 'wfc')
        mp_stru_name = str(sample_num)
        lattice_vector_torch, position_torch, tot_basis_num_torch, band_cut_index_torch, tot_kunm_torch, kpt_torch, eigenvectors_enlager_torch = torch.load(file_path_wa)
        torch.save([file_path, lattice_vector_torch, position_torch, tot_basis_num_torch, band_cut_index_torch, tot_kunm_torch, kpt_torch, eigenvectors_enlager_torch, H0_ds, H0, overlap_tensor, mask_tensor, edge_vec, edge_src, edge_dst, ele_list, mp_stru_name, delta_H_dp, H0_raw, overlap_tensor_raw, mask_tensor_raw, delta_H_raw], save_pth_root+mp_stru_name+'.pth')
        sample_num += 1
        dataset_file_w.write(save_pth_root+mp_stru_name+'.pth\n')
        print('save ', save_pth_root+mp_stru_name+'.pth')
    dataset_file_w.close()
    

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


def main(args):  
    _log = FileLogger(is_master=True, is_rank0=True, output_dir=args.output_dir)
    _log.info(args)
    irreps_edge, construct_kernel = get_hamiltonian_size(args, spinful=True)
    Process_Material_Dataset('train', construct_kernel)
    Process_Material_Dataset('val', construct_kernel)
    Process_Material_Dataset('test', construct_kernel) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Processing Data', parents=[get_args_parser()])
    args = parser.parse_args()  
    main(args)