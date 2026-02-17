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

import nets
from nets import model_entrypoint

from timm.utils import ModelEmaV2, get_state_dict
from timm.scheduler import create_scheduler

from engine import AverageMeter, compute_stats
from dataset_nano import nanotube_weak, config_set_target, DatasetInfo
from operator import itemgetter
from scipy.linalg import block_diag
from torch.nn.parallel import parallel_apply


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

def convert_label_with_overlap(pred_h, label, overlap):
    Denominator = torch.sum(overlap * torch.conj(overlap))
    Numerator =  torch.real(torch.sum((pred_h-label) * torch.conj(overlap)))
    delta_mu = Numerator/(Denominator+1e-6)
    new_label = label + delta_mu*overlap
    return new_label


class MaskedMAELoss(torch.nn.Module):
    def __init__(self, threshold_max=10000, threshold_min=-10000, factor=1.0):
        super(MaskedMAELoss, self).__init__()
        self.mae_loss = torch.nn.L1Loss(reduction='none')
        self.threshold_max = threshold_max
        self.threshold_min = threshold_min
        self.factor = factor

    def forward(self, input, target, mask):
        loss = self.mae_loss(input, target)
        threshold_mask = ((self.threshold_min < target.abs()) & (target.abs() < self.threshold_max)).float()
        combined_mask = mask * threshold_mask
        loss = loss * combined_mask * self.factor
        masked_loss = loss.sum() / (combined_mask.sum() + 1e-6) 
        return masked_loss
    

class MaskedMAELosswithGuage(torch.nn.Module):
    def __init__(self, threshold_max=100000000, threshold_min=-100000000, factor=1.0):
        super(MaskedMAELosswithGuage, self).__init__()
        self.mae_loss = torch.nn.L1Loss(reduction='none')
        self.threshold_max = threshold_max
        self.threshold_min = threshold_min
        self.factor = factor

    def forward(self, input, target, overlap, mask):
        target = convert_label_with_overlap(input, target, overlap)
        loss = self.mae_loss(input, target)
        threshold_mask = ((self.threshold_min < target.abs()) & (target.abs() < self.threshold_max)).float()
        combined_mask = mask * threshold_mask
        loss = loss * combined_mask * self.factor
        combined_mask_sum = combined_mask.sum()
        masked_loss = loss.sum() / (combined_mask_sum+1e-7)
        return combined_mask_sum, masked_loss.real

class MaskedWALoss_Guage(torch.nn.Module):
    def __init__(self, threshold_max=10000, threshold_min=-10000, factor_pspace = 0.0002, factor_qspace = 0.0001, factor_overlap = 0.00015):
        super(MaskedWALoss_Guage, self).__init__()
        self.wa_loss =  torch.nn.L1Loss(reduction='none')
        self.threshold_max = threshold_max
        self.threshold_min = threshold_min
        self.factor_pspace = factor_pspace
        self.factor_qspace = factor_qspace
        self.factor_overlap = factor_overlap
        self.unify_orb_num = 27 * 2 
    
    def _switch_wa_loss_mse(self):
        self.wa_loss = torch.nn.MSELoss(reduction='none')

    def _switch_wa_loss_mae(self):
        self.wa_loss = torch.nn.L1Loss(reduction='none')

    def get_R_list(self, edge_vec, edge_src, edge_dst, lattice_vector, atoms_positions):
        self.pair_num = edge_dst.shape[0]
        cell_inv = torch.inverse(torch.transpose(lattice_vector,0,1))
        threshold = 0.001
        self.R_tot_list = []
        self.tot_num = atoms_positions.shape[0]
        for ii in range(self.pair_num):
            # 注意：edge_src[ii] 与 edge_dst[ii]若为单个整数，直接作为索引使用即可
            posit_ii = atoms_positions[edge_src[ii]]
            posit_jj = atoms_positions[edge_dst[ii]]
            # 计算两原子之间的位移差，再与 edge_vec 修正后获得差值向量
            R_dis = edge_vec[ii] - (posit_jj - posit_ii)
            # 用 cell_inv 与 R_dis 得到晶胞内的坐标差
            R_temp = cell_inv @ R_dis
            # 四舍五入得到最近整数晶胞平移向量
            R_tot = torch.round(R_temp)
            diff = torch.abs(R_tot - R_temp)
            # 当任一分量的差值超过阈值，则报错退出
            if torch.any(diff > threshold):
                print("转换数据出错，请检查结构或者输入数据")
                import sys
                sys.exit(2)
            self.R_tot_list.append(R_tot)
        return self.R_tot_list
    
    def divide_space(self, tot_kunm_torch, tot_basis_num_torch, kpt_data, band_cut_index, eigenvectors_enlager_torch):
        # 随机选取一个 k 点索引
        rand_kpt_index = torch.randint(low=0, high=tot_kunm_torch, size=(1,)).item()
        # 获取该 k 点的坐标
        kpt_coord = kpt_data[rand_kpt_index]  # shape: [3]
        # 获取波函数数据
        nbasis = eigenvectors_enlager_torch.shape[1]
        eigenvectors_recovered = eigenvectors_enlager_torch.view(tot_kunm_torch, tot_basis_num_torch, nbasis)
        # 随机获取波函数信息并分为 P 和 Q 两部分空间
        eigenvectors_P_space = eigenvectors_recovered[rand_kpt_index, :band_cut_index, :]
        eigenvectors_Q_space = eigenvectors_recovered[rand_kpt_index, band_cut_index:, :]

        return kpt_coord, eigenvectors_P_space, eigenvectors_Q_space 

    def cal_wfc_hk_vectorized(self, edge_src, edge_dst, hr_matrix, kpt_coord, eigenvectors_P_space, eigenvectors_Q_space):
        device = hr_matrix.device

        # 1) 堆成 (pair_count, 3) 的 R_tot 张量
        R_tot_tensor = torch.stack(self.R_tot_list, dim=0).to(device)      # (pair_count, 3)
        pair_count   = R_tot_tensor.shape[0]

        # 2) 基本尺寸
        orb_per_atom     = self.unify_orb_num                             # 每原子轨道数
        total_atom_count = self.tot_num                                   # 原子总数
        total_kpoints    = 1
        matrix_dim       = orb_per_atom * total_atom_count               # hk 矩阵维度

        # 3) 准备 kpt_data 和 hr_matrix
        kpt_tensor = kpt_coord.to(device).float()                         # (total_kpoints, 3)
        hr_tensor  = hr_matrix.to(device).view(pair_count, orb_per_atom, orb_per_atom)  # (pair_count, orb, orb)

        # 4) 计算 phase_factors：exp(2πi k·R)
        dot_products  = kpt_tensor @ R_tot_tensor.t()                     # (total_kpoints, pair_count)
        phase_factors = torch.exp(2j * torch.pi * dot_products)          # (total_kpoints, pair_count)

        # 5) 生成每个 k, pair 的贡献 contrib
        hr_expand    = hr_tensor.unsqueeze(0)                            # (1, pair_count, orb, orb)
        phase_expand = phase_factors.view(total_kpoints, pair_count, 1, 1) 
        contrib      = phase_expand * hr_expand                           # (total_kpoints, pair_count, orb, orb)
        contrib_flat = contrib.reshape(total_kpoints, -1)                 # (total_kpoints, pair_count*orb*orb)

        # 6) 计算扁平化索引，用于 scatter_add
        idx_local = torch.arange(orb_per_atom, device=device)
        row_local = idx_local.view(orb_per_atom,1).expand(orb_per_atom,orb_per_atom)
        col_local = idx_local.view(1,orb_per_atom).expand(orb_per_atom,orb_per_atom)

        start_row = (edge_src.to(device) * orb_per_atom).view(pair_count,1,1)
        start_col = (edge_dst.to(device) * orb_per_atom).view(pair_count,1,1)

        global_row_idx = (start_row + row_local).reshape(-1)  # (pair_count*orb*orb,)
        global_col_idx = (start_col + col_local).reshape(-1)

        flat_index   = global_row_idx * matrix_dim + global_col_idx
        index_tensor = flat_index.unsqueeze(0).expand(total_kpoints, -1)  # (total_kpoints, pair_count*orb*orb)

        # 7) scatter_add 到 flat hk 矩阵
        hk_flat = torch.zeros((total_kpoints, matrix_dim * matrix_dim),
                            dtype=torch.complex64,
                            device=device)
        hk_flat.scatter_add_(1, index_tensor, contrib_flat)

        # 8) reshape 回 (total_kpoints, matrix_dim, matrix_dim)
        hk_matrix = hk_flat.view(total_kpoints, matrix_dim, matrix_dim)

        # 9) 计算 reduce_space
        eigenvectors_P_space = eigenvectors_P_space.unsqueeze(0)
        eigenvectors_Q_space = eigenvectors_Q_space.unsqueeze(0)
        reduce_P_space  = eigenvectors_P_space.conj() @ hk_matrix @ eigenvectors_P_space.transpose(1, 2)
        reduce_Q_space  = eigenvectors_Q_space.conj() @ hk_matrix @ eigenvectors_Q_space.transpose(1, 2)
        reduce_PQ_space = eigenvectors_P_space.conj() @ hk_matrix @ eigenvectors_Q_space.transpose(1, 2)
        return reduce_P_space, reduce_Q_space, reduce_PQ_space 

    def grep_min_mu(self,  reduce_P_space1, reduce_Q_space1,  
                           reduce_P_space2, reduce_Q_space2, 
                           H_gt, pred_H, overlap, mask_tensor):
        
        # 由于实空间 H(R) 矩阵与k空间 H(k) 矩阵mae定义的角度不一致，两者进行混合时需调节有效的factor参数
        self.factor_R = 1 - self.factor_pspace - self.factor_qspace - self.factor_overlap

        # 计算总的矩阵数目
        self.N_number1 = torch.sum(mask_tensor)
        self.N_number2 = reduce_P_space1.shape[0] * reduce_P_space1.shape[1] * reduce_P_space1.shape[2]
        self.N_number3 = reduce_Q_space1.shape[0] * reduce_Q_space1.shape[1] * reduce_Q_space1.shape[2]

        # H(R)的贡献
        n1 = self.factor_R * torch.real(torch.sum((pred_H - H_gt) * torch.conj(overlap)))/self.N_number1
        d1 = self.factor_R * torch.real(torch.sum(overlap * torch.conj(overlap)))/self.N_number1

        # P空间的贡献
        reduce_P_space =  reduce_P_space2 - reduce_P_space1
        n2 = self.factor_pspace * torch.real(reduce_P_space.diagonal(dim1=1, dim2=2).sum())/self.N_number2 
        d2 = self.factor_pspace / reduce_P_space1.shape[1]

        # Q空间的贡献
        reduce_Q_space =  reduce_Q_space2 - reduce_Q_space1
        n3 = self.factor_qspace * torch.real(reduce_Q_space.diagonal(dim1=1, dim2=2).sum())/self.N_number3
        d3 = self.factor_qspace / reduce_Q_space1.shape[1]

        # PQ空间耦合与mu值无关

        # 计算mu值
        Numerator = n1 + n2 + n3
        Denominator = d1 + d2 + d3
        self.mu = Numerator/Denominator  
        # print(d1,d2,d3)
        return self.mu
    
    def cal_loss(self, 
                reduce_P_space1, reduce_Q_space1, reduce_PQ_space1,
                reduce_P_space2, reduce_Q_space2, reduce_PQ_space2,
                H_gt, pred_H, overlap):

        try:
            self.mu = float(self.mu)
        except:
            print('mu except!')
            self.mu = 0.0
          
        # 获取单位矩阵
        eye_matrix_p = torch.eye(reduce_P_space1.shape[1]).to(H_gt.device)          
        eye_batch_p = eye_matrix_p.unsqueeze(0).repeat(reduce_P_space1.shape[0], 1, 1).float().to(H_gt.device)
        eye_matrix_q = torch.eye(reduce_Q_space1.shape[1]).to(H_gt.device)
        eye_batch_q = eye_matrix_q.unsqueeze(0).repeat(reduce_Q_space1.shape[0], 1, 1).float().to(H_gt.device)

        # 计算 mae
        loss_hr = self.wa_loss(torch.real(H_gt + self.mu * overlap).float(), torch.real(pred_H).float())

        loss_p_space_real = self.wa_loss(torch.real(reduce_P_space1).float() + self.mu*eye_batch_p, torch.real(reduce_P_space2).float())
        loss_p_space_imag = self.wa_loss(torch.imag(reduce_P_space1).float(), torch.imag(reduce_P_space2).float())

        loss_q_space_real = self.wa_loss(torch.real(reduce_Q_space1).float() + self.mu*eye_batch_q, torch.real(reduce_Q_space2).float())
        loss_q_space_imag = self.wa_loss(torch.imag(reduce_Q_space1).float(), torch.imag(reduce_Q_space2).float())

        loss_pq_space_real = self.wa_loss(torch.real(reduce_PQ_space1).float(), torch.real(reduce_PQ_space2).float())
        loss_pq_space_imag = self.wa_loss(torch.imag(reduce_PQ_space1).float(), torch.imag(reduce_PQ_space2).float())

        self.N_number4 = reduce_PQ_space1.shape[0] * reduce_PQ_space1.shape[1] * reduce_PQ_space1.shape[2]
        tot_loss = self.factor_R * loss_hr.sum() /self.N_number1 + \
                   self.factor_pspace * (loss_p_space_real.sum() + loss_p_space_imag.sum()) / self.N_number2 + \
                   self.factor_qspace * (loss_q_space_real.sum() + loss_q_space_imag.sum()) / self.N_number3 + \
                   self.factor_overlap * (loss_pq_space_real.sum() + loss_pq_space_imag.sum()) / self.N_number4


        return tot_loss
    

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


# from https://github.com/Open-Catalyst-Project/ocp/blob/main/ocpmodels/modules/loss.py#L7
class L2MAELoss(torch.nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction
        assert reduction in ["mean", "sum"]

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        dists = torch.norm(input - target, p=2, dim=-1)
        if self.reduction == "mean":
            return torch.mean(dists)
        elif self.reduction == "sum":
            return torch.sum(dists)

class Material_Project_Dataset(torch.utils.data.Dataset):
    def __init__(self, mode, construct_kernel, device, dataset_root='/your_path/NextHAM/datasets/'):
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
    
def save_sample(sample, save_dir, sample_name):
    """Save individual sample as a .pth file."""
    os.makedirs(save_dir, exist_ok=True)
    sample_path = os.path.join(save_dir, f"{sample_name}.pth")
    torch.save(sample, sample_path)  # Save using torch.save


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
                                    device_torch=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    return irreps_edge, construct_kernel

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)    
    torch.manual_seed(seed)    
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False    
    os.environ["PYTHONHASHSEED"] = str(seed)

def main(args):
    
    _log = FileLogger(is_master=True, is_rank0=True, output_dir=args.output_dir)
    _log.info(args)
    

    ''' Config '''
    irreps_edge, construct_kernel = get_hamiltonian_size(args, spinful=True)

    mean = 0.
    std = 1. 

    # since dataset needs random 
    set_seed(args.seed)

    ''' Network '''
    create_model = model_entrypoint(args.model_name)

    devices = [[0], [1], [2], [3]]

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
            ).to(f'cuda:{devices[model_idx][0]}'))

    checkpoint_paths = [args.checkpoint_path1, args.checkpoint_path2, args.checkpoint_path3, args.checkpoint_path4]

    for model_idx in range(4):
        checkpoint_path = checkpoint_paths[model_idx]

        if checkpoint_path is not None:
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            model_range_state_dict = models[model_idx].state_dict()

            compatible_state_dict = {k: v for k, v in state_dict['state_dict'].items() 
                                    if k in model_range_state_dict and v.size() == model_range_state_dict[k].size()}
        
            model_range_state_dict.update(compatible_state_dict)
            models[model_idx].load_state_dict(model_range_state_dict)

            print('model_idx, len(compatible_state_dict), len(model_range_state_dict): ', model_idx, len(compatible_state_dict), len(model_range_state_dict))
        else:
            print('no pre-trained model')


    n_parameters = sum(p.numel() for p in models[0].parameters())*4
    _log.info('Number of params: {}'.format(n_parameters))
  
    ''' Dataset '''
    train_dataset, val_dataset, test_dataset = get_material_project_dataset(construct_kernel = construct_kernel, device=devices[0][0])

    _log.info('')
    _log.info('Training set size:   {}'.format(len(train_dataset)))
    _log.info('Validation set size: {}'.format(len(val_dataset)))
    _log.info('Testing set size:    {}\n'.format(len(test_dataset)))

    ''' Data Loader '''
    from tg_src.graph import Collater
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=args.workers, pin_memory = True)
    val_loader = DataLoader(val_dataset, batch_size=1, num_workers=args.workers, pin_memory = True)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=args.workers, pin_memory = True)

    ''' Optimizer and LR Scheduler '''
    optimizers = []
    lr_schedulers = []
    for model_idx in range(4):
        params = list(filter(lambda p: p.requires_grad, models[model_idx].parameters()))
        optimizer_h = torch.optim.Adam(params, lr=args.lr, betas=(0.9, 0.999))
        optimizers.append(optimizer_h)
        lr_scheduler_h, _ = create_scheduler(args, optimizer_h)
        lr_schedulers.append(lr_scheduler_h)

    criterion = MaskedWALoss_Guage() 
    
    ''' Compute stats '''
    if args.compute_stats:
        compute_stats(train_loader, max_radius=args.radius, logger=_log, print_freq=args.print_freq)
        return
    
    # record the best validation and testing errors and corresponding epochs
    best_metrics = {'val_epoch': 0, 'test_epoch': 0, 
        'val_ham_err': float('inf'),  'val_trace_err': float('inf'), 
        'test_ham_err': float('inf'), 'test_trace_err': float('inf')}
 
    epoch = 0
    
    best_val_err = 1000.0

    while epoch < args.epochs:
        
        for model_idx in range(4):
            lr_schedulers[model_idx].step(epoch)

        train_err = train_eval_one_epoch(args=args,  models=models, devices=devices, criterion=criterion, data_loader=train_loader, optimizers=optimizers, epoch=epoch, print_freq=args.print_freq, logger=_log, construct_kernel=construct_kernel)

        val_err = train_eval_one_epoch(args=args, models=models, devices=devices, criterion=criterion, data_loader=val_loader, optimizers=optimizers, epoch=epoch, print_freq=args.print_freq, logger=_log, construct_kernel=construct_kernel, train = False, print_progress=True)

        if val_err < best_val_err:
            best_val_err = val_err
            for model_idx in range(4):
                torch.save(
                    {'state_dict': models[model_idx].state_dict()}, 
                    os.path.join(args.output_dir, 'model_range'+str(model_idx)+'_best.pth.tar')
                )

        epoch += 1


def train_eval_one_epoch(args, 
                    models: list, 
                    devices: list, 
                    criterion: torch.nn.Module,
                    data_loader: Iterable,
                    optimizers: list,
                    epoch: int, 
                    print_freq: int = 100, 
                    logger=None, construct_kernel = None, train = True,
                    print_progress=False):

    global ele_dict

    if train:
        for model_idx in range(len(models)):
            models[model_idx].train()
        if epoch > 130:
            criterion._switch_wa_loss_mse()
        criterion.train()
    else:
        for model_idx in range(len(models)):
            models[model_idx].eval()  
        criterion._switch_wa_loss_mae()
        criterion.eval()

    loss_metrics = {'ham': AverageMeter(), 'trace': AverageMeter(), 'baseline_ham': AverageMeter()}
    mae_metrics  = {'ham': AverageMeter(), 'ham_lt_10': AverageMeter(), 'ham_10_100': AverageMeter(), 'ham_100_1000': AverageMeter(), 'ham_gt_1000': AverageMeter(), 'trace': AverageMeter(), 'baseline_ham': AverageMeter(), 'baseline_ham_lt_10': AverageMeter(), 'baseline_ham_10_100': AverageMeter(), 'baseline_ham_l00_1000': AverageMeter(), 'baseline_ham_gt_1000': AverageMeter(), 'ham_ratio': AverageMeter(), 'ham_lt_10_ratio': AverageMeter(), 'ham_10_100_ratio': AverageMeter(), 'ham_100_1000_ratio': AverageMeter(), 'ham_gt_1000_ratio': AverageMeter(), 'ham_on_site': AverageMeter(),  'ham_1_2': AverageMeter(), 'ham_2_4': AverageMeter(), 'ham_4_6': AverageMeter(),}
    loss_h_all = []
    mae_h_all = []
    
    sample_num = 0 
    start_time = time.perf_counter()

    MAE_metric = MaskedMAELosswithGuage()
    MAE_metric_lt_10 = MaskedMAELosswithGuage(threshold_max=0.01, threshold_min=-100000000)
    MAE_metric_10_100 = MaskedMAELosswithGuage(threshold_max=0.1, threshold_min=0.01)
    MAE_metric_100_1000 = MaskedMAELosswithGuage(threshold_max=1, threshold_min=0.1)
    MAE_metric_gt_1000 = MaskedMAELosswithGuage(threshold_max=100000000, threshold_min=1)

    criterion_trace = MaskedMAELoss()

    ls = [1, 1, 1, 1, 3, 3, 5, 5, 7]
    range_dis = [[0.0, 1.0], [1.0, 2.0], [2.0, 4.0], [4.0, 6.0]]
    for step, data in enumerate(data_loader):
        file_path, lattice_vector_torch, position_torch, tot_basis_num_torch, band_cut_index_torch, tot_kunm_torch, kpt_torch, eigenvectors_enlager_torch, H0_ds, H0, overlap_tensor, mask_tensor, edge_vec, edge_src, edge_dst, ele_list, mp_stru_name, delta_H_dp, H0_raw, overlap_tensor_raw, mask_tensor_raw, delta_H_raw = data
        file_path, lattice_vector_torch, position_torch, tot_basis_num_torch, band_cut_index_torch, tot_kunm_torch, kpt_torch, eigenvectors_enlager_torch, H0_ds, H0, overlap_tensor, mask_tensor, edge_vec, edge_src, edge_dst, delta_H_dp, H0_raw, overlap_tensor_raw, mask_tensor_raw, delta_H_raw = file_path[0], lattice_vector_torch[0].to(devices[0][0], non_blocking=True), position_torch[0].to(devices[0][0], non_blocking=True), tot_basis_num_torch[0].to(devices[0][0], non_blocking=True), band_cut_index_torch[0].to(devices[0][0], non_blocking=True), tot_kunm_torch[0].to(devices[0][0], non_blocking=True), kpt_torch[0].to(devices[0][0], non_blocking=True), eigenvectors_enlager_torch[0].to(devices[0][0], non_blocking=True), H0_ds[0].to(devices[0][0], non_blocking=True), H0[0].to(devices[0][0], non_blocking=True), overlap_tensor[0].to(devices[0][0], non_blocking=True), mask_tensor[0].to(devices[0][0], non_blocking=True), edge_vec[0].to(devices[0][0], non_blocking=True), edge_src.to(torch.int64)[0].to(devices[0][0], non_blocking=True), edge_dst.to(torch.int64)[0].to(devices[0][0], non_blocking=True), delta_H_dp[0].to(devices[0][0], non_blocking=True), H0_raw[0].to(devices[0][0], non_blocking=True), overlap_tensor_raw[0].to(devices[0][0], non_blocking=True), mask_tensor_raw[0].to(devices[0][0], non_blocking=True), delta_H_raw[0].to(devices[0][0], non_blocking=True)        
        node_num = max(int(max(edge_src)+1), int(max(edge_dst)+1))
        batch = torch.ones((node_num,), dtype=torch.int32).to(devices[0][0], non_blocking=True)
        node_atom = [-1 for _ in range(node_num)]
        for ele_idx in range(len(ele_list)):
            node_atom[edge_src[ele_idx]] = ele_dict[ele_list[ele_idx][0][0]]

        node_atom = torch.tensor(node_atom, dtype=torch.long, device=devices[0][0])

        data_list_device = [[] for _ in range(4)]

        for model_idx in range(4):
            if model_idx > 0:
                data_list_device[model_idx] = [H0_ds.to(devices[model_idx][0], non_blocking=True), edge_src.to(devices[model_idx][0], non_blocking=True), edge_dst.to(devices[model_idx][0], non_blocking=True),  edge_vec.to(devices[model_idx][0], non_blocking=True),  batch.to(devices[model_idx][0], non_blocking=True), node_atom.to(devices[model_idx][0], non_blocking=True)]
            else:
                data_list_device[model_idx] = [H0_ds, edge_src, edge_dst, edge_vec, batch, node_atom]    
        pred_h_all = []
        pred_h_trace_all = []  
        mask_dis_list = []
        inputs_list = []      
        kwargs_list = []      
        for model_idx in range(4):
            kw_params = {
                'weak_ham_in':    data_list_device[model_idx][0],
                'node_num':       node_num,
                'edge_src':       data_list_device[model_idx][1],
                'edge_dst':       data_list_device[model_idx][2],
                'edge_vec':       data_list_device[model_idx][3],
                'batch':          data_list_device[model_idx][4],
                'node_atom':      data_list_device[model_idx][5],
                'use_sep':        True,
                'range_dis':      range_dis[model_idx]
            }
            inputs_list.append(())
            kwargs_list.append(kw_params)

        outputs = parallel_apply(models, inputs_list, kwargs_tup=kwargs_list)

        pred_h_all = []
        pred_h_trace_all = []
        mask_dis_list = []

        for i, output_tuple in enumerate(outputs):
            pred_h_direct_sum, pred_h_trace, mask_dis = output_tuple
            pred_h_all.append(pred_h_direct_sum.to(devices[0][0]))
            pred_h_trace_all.append(pred_h_trace.to(devices[0][0]))
            mask_dis_list.append(mask_dis.to(devices[0][0]))

        pred_h = torch.sum(torch.stack(pred_h_all), dim=0)
        pred_h_trace = torch.sum(torch.stack(pred_h_trace_all), dim=0)
        pred_h = construct_kernel.get_H(pred_h)

        delta_H_pred_real = reverse_transform_matrix(pred_h[:,0,:].real, ls)

        H_gt = delta_H_raw + H0_raw
        H_pred = H0_raw.clone()

        H_pred = H_pred.reshape(-1, 2, 27, 2, 27)
        H_pred[:, 0, :, 0, :].real = H_pred[:, 0, :, 0, :].real + delta_H_pred_real
        H_pred[:, 1, :, 1, :].real = H_pred[:, 1, :, 1, :].real + delta_H_pred_real
        H_pred = H_pred.reshape(-1, 54, 54)

        edge_vec, edge_src, edge_dst = edge_vec.to(devices[0][0], non_blocking=True), edge_src.to(devices[0][0], non_blocking=True), edge_dst.to(devices[0][0], non_blocking=True)

        R_list = criterion.get_R_list(edge_vec, edge_src, edge_dst, lattice_vector_torch, position_torch)

        # 随机获得一个k点波函数，并且通过 band_cut_index_torch 指标把空间分为 P & Q 两部分
        kpt_coord, eigenvectors_P_space, eigenvectors_Q_space = criterion.divide_space(tot_kunm_torch, tot_basis_num_torch, kpt_torch, band_cut_index_torch, eigenvectors_enlager_torch)

        # 获取子哈密顿量波函数投影矩阵
        reduce_P_space1, reduce_Q_space1, reduce_PQ_space1  = criterion.cal_wfc_hk_vectorized(edge_src, edge_dst, H_gt, kpt_coord, eigenvectors_P_space, eigenvectors_Q_space)
        reduce_P_space2, reduce_Q_space2, reduce_PQ_space2  = criterion.cal_wfc_hk_vectorized(edge_src, edge_dst, H_pred, kpt_coord, eigenvectors_P_space, eigenvectors_Q_space)

        # 计算 mu 与 mae
        mu = criterion.grep_min_mu(reduce_P_space1, reduce_Q_space1, reduce_P_space2, reduce_Q_space2, 
                                H_gt, H_pred, overlap_tensor_raw, mask_tensor_raw)


        loss_h = criterion.cal_loss(reduce_P_space1, reduce_Q_space1, reduce_PQ_space1,
                                reduce_P_space2, reduce_Q_space2, reduce_PQ_space2,
                                H_gt, H_pred, overlap_tensor_raw).real      


        sample_num += 1

        if torch.isnan(loss_h).any() or torch.isinf(loss_h).any():
            print('nan or inf loss')
            continue

        trace_label = construct_kernel.get_H_trace(delta_H_dp + mu*overlap_tensor)

        trace_mask = construct_kernel.get_H_trace(mask_tensor).to(torch.bool).to(mask_tensor.real.dtype)

        if args.with_trace:
            loss_t = criterion_trace(pred_h_trace, trace_label, trace_mask) 
            loss_all = 0.8 * loss_h + 0.2 * loss_h.item() / loss_t.item() * loss_t
        else:
            loss_t = torch.zeros_like(loss_h)
            loss_all = loss_h
            
        loss_h_all.append(loss_h.item())

        if train:
            for model_idx in range(4):
                optimizers[model_idx].zero_grad()
            loss_all.backward()
            for model_idx in range(4):
                optimizers[model_idx].step()

        loss_metrics['ham'].update(loss_h.item(), n=1)
        loss_metrics['trace'].update(loss_t.item(), n=1)
        if args.with_trace:
            mae_trace = torch.mean(torch.abs(pred_h_trace.detach()-trace_label)).item()
        else:
            mae_trace = 0
        mae_metrics['trace'].update(mae_trace, n=1)

        combined_mask_sum, mae_ham = MAE_metric(H_pred.detach(), H_gt.detach(), overlap_tensor_raw.detach(), mask_tensor_raw.detach())
        _, mae_ham_on_site = MAE_metric(H0_raw.detach(), H_gt.detach(), overlap_tensor_raw.detach(), mask_tensor_raw.detach()*mask_dis_list[0][:,:,None])     
        _, mae_ham_1_2 = MAE_metric(H0_raw.detach(), H_gt.detach(), overlap_tensor_raw.detach(), mask_tensor_raw.detach()*mask_dis_list[1][:,:,None])       
        _, mae_ham_2_4 = MAE_metric(H0_raw.detach(), H_gt.detach(), overlap_tensor_raw.detach(), mask_tensor_raw.detach()*mask_dis_list[2][:,:,None])    
        _, mae_ham_4_6 = MAE_metric(H0_raw.detach(), H_gt.detach(), overlap_tensor_raw.detach(), mask_tensor_raw.detach()*mask_dis_list[3][:,:,None])    

        _, mae_baseline_ham = MAE_metric(H0_raw.detach(), H_gt.detach(), overlap_tensor_raw.detach(), mask_tensor_raw.detach())

        mae_metrics['ham'].update(mae_ham.item(), n=1)
        mae_metrics['ham_on_site'].update(mae_ham_on_site.item(), n=1)
        mae_metrics['ham_1_2'].update(mae_ham_1_2.item(), n=1)
        mae_metrics['ham_2_4'].update(mae_ham_2_4.item(), n=1)
        mae_metrics['ham_4_6'].update(mae_ham_4_6.item(), n=1)
        mae_metrics['baseline_ham'].update(mae_baseline_ham.item(), n=1)

        mae_h_all.append(mae_ham)
    
        # logging
        if train:


            if step % print_freq == 0 or step == len(data_loader) - 1: 
                e = (step + 1) / len(data_loader)
                info_str = 'Epoch: [{epoch}][{step}/{length}] \t'.format(epoch=epoch, step=step, length=len(data_loader))
                info_str +=  'loss_ham: {loss_ham:.9f}, loss_trace: {loss_trace:.9f}, ham_MAE: {ham_mae:.9f},  baseline_ham_MAE: {baseline_ham_mae:.9f}, trace_MAE: {trace_mae:.9f}'.format(
                    loss_ham=loss_metrics['ham'].avg, loss_trace=loss_metrics['trace'].avg, ham_mae=mae_metrics['ham'].avg,  baseline_ham_mae=mae_metrics['baseline_ham'].avg, trace_mae=mae_metrics['trace'].avg, 
                )                
                logger.info(info_str)

            if sample_num % 100 == 0:
                for model_idx in range(4):
                    torch.save(
                        {'state_dict': models[model_idx].state_dict()}, 
                        os.path.join(args.output_dir, 'model_range'+str(model_idx)+'_curr.pth.tar')
                    )
        else:
            if (step % print_freq == 0 or step == len(data_loader) - 1) and print_progress: 
                e = (step + 1) / len(data_loader)
                info_str = '[{step}/{length}] \t'.format(step=step, length=len(data_loader))

                info_str +=  'loss_ham: {loss_ham:.9f}, loss_trace: {loss_trace:.9f}, ham_MAE: {ham_mae:.9f}, ham_on_site: {ham_on_site:.9f}, ham_1_2: {ham_1_2:.9f}, ham_2_4: {ham_2_4:.9f}, ham_4_6: {ham_4_6:.9f}, baseline_ham_MAE: {baseline_ham_mae:.9f}, trace_MAE: {trace_mae:.9f}'.format(
                    loss_ham = loss_metrics['ham'].avg, loss_trace = loss_metrics['trace'].avg, ham_mae = mae_metrics['ham'].avg,  ham_on_site = mae_metrics['ham_on_site'].avg,  ham_1_2 = mae_metrics['ham_1_2'].avg,  ham_2_4 = mae_metrics['ham_2_4'].avg,  ham_4_6 = mae_metrics['ham_4_6'].avg, baseline_ham_mae = mae_metrics['baseline_ham'].avg, trace_mae = mae_metrics['trace'].avg, 
                )

                logger.info(info_str)

    del loss_all, loss_h, pred_h
    torch.cuda.empty_cache()
    return mae_metrics['ham'].avg

        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser('Training equivariant networks on Material Project', parents=[get_args_parser()])
    args = parser.parse_args()  
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)