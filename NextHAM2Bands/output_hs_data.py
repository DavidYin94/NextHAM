import torch
from typing import Optional, Union
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csc_matrix, csr_matrix
from scipy.linalg import block_diag
from ase.io import read, write
import numpy as np
import os
import sys

try:
    from timing_utils import time_execution
except ImportError:
    sys.path.append(str(Path(__file__).parent.parent))
    from timing_utils import time_execution

class transform_abacus_hamiltion_data:
    def __init__(self, 
                 usage: str,
                 nspin: int,
                 stru_file: Union[str, Path],
                 file_read: Union[str, Path],
                 save_path: Union[str, Path]
    ):
        self.usage = usage
        self.nspin = nspin
        self.stru_file = Path(stru_file)
        self.file_read = Path(file_read)
        self.save_path = Path(save_path)

        self.orb_origin = {'H': 5,   'He': 5,  'Li': 7,  'Be': 7,  'B': 13,  'C': 13,  'N': 13,  'O': 13,  'F': 13, 'Ne': 13, 
                           'Na': 15, 'Mg': 15, 'Al': 13, 'Si': 13, 'P': 13,  'S': 13,  'Cl': 13, 'Ar': 13, 'K': 15, 
                           'Sc': 27, 'V': 27,  'Fe': 27, 'Co': 27, 'Ni': 27, 'Cu': 27, 'Zn': 27, 'Ga': 25, 'Ge': 25, 
                           'Br': 13, 'Y': 27,  'Nb': 27, 'Mo': 27, 'Pd': 25, 'Ag': 27, 'Cd': 27, 'In': 25, 'Sn': 25, 
                           'Sb': 25, 'Te': 25, 'I': 13,  'Xe': 13, 'Hf': 27, 'Ta': 27, 'Re': 27, 'Pt': 27, 'Au': 27, 
                           'Hg': 27, 'Tl': 25, 'Pb': 25, 'Bi': 25, 'Ca': 15, 'Ti': 27, 'Cr': 27, 'Mn': 27, 'Kr': 13, 
                           'Rb': 15, 'Sr': 15, 'Zr': 27, 'Tc': 27, 'Ru': 27, 'Rh': 27, 'Cs': 15, 'Ba': 15, 'W': 27, 
                           'Os': 27, 'Ir': 27, 'As': 13, 'Se': 13}
    
    def transform_orb(self, nspin):
        self.abacus2deeph = {}
        self.abacus2deeph[0] = np.eye(1)
        self.abacus2deeph[1] = np.eye(3)[[1, 2, 0]]
        self.abacus2deeph[2] = np.eye(5)[[0, 3, 4, 1, 2]]
        self.abacus2deeph[3] = np.eye(7)[[0, 1, 2, 3, 4, 5, 6]]
        minus_dict = {
            1: [0, 1],
            2: [3, 4],
            3: [1, 2, 5, 6],
        }
        # 应用符号翻转
        for k, v in minus_dict.items():
            self.abacus2deeph[k][v] *= -1

        # 根据 nspin 决定轨道列表
        if nspin == 1:
            orb_list = [0, 0, 0, 0, 1, 1, 2, 2, 3]
        elif nspin == 4:
            orb_list = [0, 0, 0, 0, 1, 1, 2, 2, 3, 0, 0, 0, 0, 1, 1, 2, 2, 3]
        else:
            raise ValueError(f"Unsupported nspin value: {nspin}")

        # 修正调用方式，使用 [] 访问字典
        transofrm_matrix = block_diag(*[self.abacus2deeph[l_number] for l_number in orb_list])

        return transofrm_matrix
    
    @time_execution
    def read_stru(self):
        stru_file = self.stru_file
        # 读取 STRU 文件
        self.atoms = read(stru_file, format='abacus')
        self.atoms.wrap()
        # print(self.atoms.get_positions())
        # 计算轨道数目总和，对于自旋为4情况，矩阵维度进行扩充
        elements = self.atoms.get_chemical_symbols()
        self.stru_dim = sum(self.orb_origin.get(element, 0) for element in elements)
        if self.nspin == 4:
            self.stru_dim = 2 * self.stru_dim 
        # print(self.stru_dim)
        # 对于每一个原子指标 i 生成轨道 patching 字典[1*27] 维度列表
        # 如果考虑 自旋轨道耦合情况，轨道指标会翻倍，每一个原子指标 i 生成轨道 patching 字典[1*54] 维度列表
        # 所有补零矩阵都是按照 4s2p2d1f 形式进行的
        count_element = 0
        count_matrix_dim = -1
        self.index_relation = {}
        for element in elements:
            if self.orb_origin.get(element) == 5:    # 2s1p
                orital_patch0 = [1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
            elif self.orb_origin.get(element) == 7:  # 4s1p
                orital_patch0 = [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
            elif self.orb_origin.get(element) == 13: # 2s2p1d
                orital_patch0 = [1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
            elif self.orb_origin.get(element) == 15: # 4s2p1d
                orital_patch0 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
            elif self.orb_origin.get(element) == 25: # 2s2p2d1f
                orital_patch0 = [1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ]
            elif self.orb_origin.get(element) == 27: # 4s2p2d1f
                orital_patch0 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ]
            # 对自旋等于四轨道进行处理    
            if self.nspin != 4:
                orital_patch = orital_patch0
            else:
                orital_patch = []
                for temp_value in orital_patch0:
                    orital_patch.append(temp_value)
                    orital_patch.append(temp_value)

            orbital_index = []
            for temp_value in orital_patch:
                count_matrix_dim = count_matrix_dim + temp_value
                orbital_index.append(count_matrix_dim)
            self.index_relation[count_element] = [element, orital_patch, orbital_index]
            count_element = count_element +1 

    @time_execution
    def get_hs_data(self):
        Ry_to_eV = 13.605698066
        # 每一个矩阵为27*27维度小方块矩阵，如果考虑自旋需要翻倍
        self.unify_orb_num = 27 
        if self.nspin == 4:
            self.unify_orb_num = 2 * self.unify_orb_num

        # 读取torch数据
        torch_data = torch.load(self.file_read, weights_only=True, map_location=torch.device('cpu'))
        precise_H, precise_H_pred, weak_H, overlap_tensor, mask_tensor, edge_vec, edge_src, edge_dst, ele_list = torch_data
        # 转换为numpy数据格式
        precise_H_pred = precise_H_pred.detach().cpu().numpy()
        mask_tensor = mask_tensor.detach().cpu().numpy()
        edge_vec = edge_vec.detach().cpu().numpy()
        edge_src = edge_src.detach().cpu().numpy()
        edge_dst = edge_dst.detach().cpu().numpy()
        # 数据格式进行转化
        precise_H_pred = precise_H_pred.reshape(-1, 54, 54)
        mask_tensor = mask_tensor.reshape(-1, 54, 54)

        pair_num = len(edge_dst)
        # print(pair_num, flush=True)
        R_tot_list = [] 
        # 确定R_num 数目和数组
        for ii in range(pair_num):
            posit_ii = self.atoms.positions[edge_src[ii]]
            posit_jj = self.atoms.positions[edge_dst[ii]]
            R_dis = edge_vec[ii] - (posit_jj -posit_ii)
            R_temp = np.linalg.inv(self.atoms.cell.T) @ R_dis
            # print(R_temp, flush=True)
            R_tot = np.rint(R_temp)
            # print(R_tot, flush=True)
            
            # 计算绝对误差
            diff = np.abs(R_tot - R_temp)
            # 设置阈值
            threshold = 0.01
            if np.any(diff > threshold):
                print(diff)
                print("转换数据出错，请检查结构或者输入数据")
                sys.exit(2)  # 退出程序，返回错误代码 2
                # 将每次计算得到的R_tot（形状 (3,)）存入列表
            R_tot_list.append(R_tot)

        # 获取每一行的唯一值以及各行出现的次数
        unique_R = np.unique(R_tot_list, axis=0)
        sorted_idx = np.lexsort((unique_R[:, 2], unique_R[:, 1], unique_R[:, 0]))
        sorted_unique_R = unique_R[sorted_idx]
        R_number = len(sorted_unique_R)

        # 创建文件地址，写文件
        precise_H_pred_save = os.path.join(self.save_path, 'predict_hr_cut')
        with open(precise_H_pred_save, 'w') as f_precise_H_pred:
            f_precise_H_pred.write('STEP: 0' + '\n')
            f_precise_H_pred.write(f'Matrix Dimension of H(R): {self.stru_dim}' + '\n')
            f_precise_H_pred.write(f'Matrix number of H(R): {R_number}' + '\n')


            for ii in range(R_number):
                if self.nspin != 4:
                    precise_H_pred_R_local = np.zeros((self.stru_dim, self.stru_dim), dtype=float)

                else:
                    precise_H_pred_R_local = np.zeros((self.stru_dim, self.stru_dim), dtype=complex)
            
            # 对于每一个R_tot 进行遍历
                for jj in range(len(R_tot_list)):
                    if sorted_unique_R[ii][0]== R_tot_list[jj][0] and sorted_unique_R[ii][1]== R_tot_list[jj][1] and sorted_unique_R[ii][2]== R_tot_list[jj][2]:

                        # 对于矩阵进行逆旋转操作，转换到abacus正常基组上
                        if self.nspin == 1:
                            data_precise_H_pred_temp = np.linalg.inv(self.transform_orb(1)) @ precise_H_pred[jj] @ np.linalg.inv(self.transform_orb(1).T)

                            for row in range(self.unify_orb_num):
                                for col in range(self.unify_orb_num):
                                    if self.index_relation[edge_src[jj]][1][row] * self.index_relation[edge_dst[jj]][1][col] == 1:
                                        precise_H_pred_R_local[self.index_relation[edge_src[jj]][2][row], self.index_relation[edge_dst[jj]][2][col]] =  data_precise_H_pred_temp[row, col]
                                        

                        elif self.nspin == 4:
                            data_precise_H_pred_temp = np.linalg.inv(self.transform_orb(4)) @ precise_H_pred[jj] @ np.linalg.inv(self.transform_orb(4).T)
                            reshape_data_precise_H_pred_temp = data_precise_H_pred_temp.reshape((2, 27, 2, 27)).transpose((1, 0, 3, 2)).reshape((2 * 27, 2 * 27))


                            for row in range(self.unify_orb_num):
                                for col in range(self.unify_orb_num):
                                    if self.index_relation[edge_src[jj]][1][row] * self.index_relation[edge_dst[jj]][1][col] == 1:
                                        precise_H_pred_R_local[self.index_relation[edge_src[jj]][2][row], self.index_relation[edge_dst[jj]][2][col]] =  reshape_data_precise_H_pred_temp[row, col]
                                        
                precise_H_pred_R_local = precise_H_pred_R_local/Ry_to_eV

                
                # 转换为稀疏矩阵，写入预测精标签文件中
                sparse_data = csr_matrix(precise_H_pred_R_local)
                # 写入文件
                f_precise_H_pred.write(f'{sorted_unique_R[ii][0]:.0f} {sorted_unique_R[ii][1]:.0f} {sorted_unique_R[ii][2]:.0f} {len(sparse_data.data)}\n')
                if len(sparse_data.data) == 0:
                    pass
                else:
                    # 将 data 数组转换为字符串，每个元素之间用空格分隔，并写入一行
                    if self.nspin == 1:
                        data_str = " ".join(map(str, sparse_data.data))
                        f_precise_H_pred.write(f"{data_str}\n")
                    elif self.nspin == 4:
                        data_str = " ".join("({:.8e},{:.8e})".format(c.real, c.imag) for c in sparse_data.data)
                        f_precise_H_pred.write(f"{data_str}\n")
                    # 将 indices 数组转换为字符串，每个元素之间用空格分隔，并写入一行
                    indices_str = " ".join(map(str, sparse_data.indices))
                    f_precise_H_pred.write(f"{indices_str}\n")
                    # 将 indptr 数组转换为字符串，每个元素之间用空格分隔，并写入一行
                    indptr_str = " ".join(map(str, sparse_data.indptr))
                    f_precise_H_pred.write(f"{indptr_str}\n")


def main():
    pass

if __name__ == "__main__":
    main()