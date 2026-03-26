import os
import re
import torch
import numpy as np
import sys
from ase.io import read, write
import pyatb
from pathlib import Path
try:
    from .timing_utils import time_execution
except ImportError:
    sys.path.append(str(Path(__file__).parent.parent))
    from .timing_utils import time_execution
from pyatb.parallel import op_gather_numpy
from scipy.linalg import block_diag
from pyatb.kpt import kpoint_generator
from pyatb import RANK, COMM, SIZE, OUTPUT_PATH, RUNNING_LOG, timer

class output_wfc: 

    def __init__(self, nspin, stru_file, hr, sr, ecut, fermi_energy, mp_grid, save_path):
        self.nspin = nspin
        self.stru_file = stru_file
        self.hr = hr
        self.sr = sr
        self.ecut = ecut
        self.efermi = fermi_energy
        self.mp_grid = mp_grid
        self.save_path = save_path
        self.orb_origin = {'H': 5,   'He': 5,  'Li': 7,  'Be': 7,  'B': 13,  'C': 13,  'N': 13,  'O': 13,  'F': 13, 'Ne': 13, 
                           'Na': 15, 'Mg': 15, 'Al': 13, 'Si': 13, 'P': 13,  'S': 13,  'Cl': 13, 'Ar': 13, 'K': 15, 
                           'Sc': 27, 'V': 27,  'Fe': 27, 'Co': 27, 'Ni': 27, 'Cu': 27, 'Zn': 27, 'Ga': 25, 'Ge': 25, 
                           'Br': 13, 'Y': 27,  'Nb': 27, 'Mo': 27, 'Pd': 25, 'Ag': 27, 'Cd': 27, 'In': 25, 'Sn': 25, 
                           'Sb': 25, 'Te': 25, 'I': 13,  'Xe': 13, 'Hf': 27, 'Ta': 27, 'Re': 27, 'Pt': 27, 'Au': 27, 
                           'Hg': 27, 'Tl': 25, 'Pb': 25, 'Bi': 25, 'Ca': 15, 'Ti': 27, 'Cr': 27, 'Mn': 27, 'Kr': 13, 
                           'Rb': 15, 'Sr': 15, 'Zr': 27, 'Tc': 27, 'Ru': 27, 'Rh': 27, 'Cs': 15, 'Ba': 15, 'W': 27, 
                           'Os': 27, 'Ir': 27, 'As': 13, 'Se': 13}
        
    def transform_orb(self):
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
        if self.nspin == 1:
            orb_list = [0, 0, 0, 0, 1, 1, 2, 2, 3]
        elif self.nspin == 4:
            orb_list = [0, 0, 0, 0, 1, 1, 2, 2, 3, 0, 0, 0, 0, 1, 1, 2, 2, 3]
        else:
            raise ValueError(f"Unsupported nspin value: {self.nspin}")

        # 修正调用方式，使用 [] 访问字典
        transofrm_matrix = block_diag(*[self.abacus2deeph[l_number] for l_number in orb_list])

        return transofrm_matrix    
    
    @time_execution
    def read_stru(self):
        stru_file = self.stru_file
        # 读取 STRU 文件
        self.atoms = read(stru_file, format='abacus')
        # self.atoms.wrap()
        # 获取晶格常数（单位：Å）
        self.lattice_vector = np.array(self.atoms.get_cell())
        # 计算轨道数目总和，对于自旋为4情况，矩阵维度进行扩充
        elements = self.atoms.get_chemical_symbols()
        self.positions = self.atoms.get_positions()
        self.stru_dim = sum(self.orb_origin.get(element, 0) for element in elements)
        if self.nspin == 4:
            self.stru_dim = 2 * self.stru_dim 
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

        return self.lattice_vector, self.positions

    @time_execution
    def generate_kpoint(self, k_start = np.array([0.0, 0.0, 0.0], dtype=float), 
                              k_vect1 = np.array([1.0, 0.0, 0.0], dtype=float), 
                              k_vect2 = np.array([0.0, 1.0, 0.0], dtype=float), 
                              k_vect3 = np.array([0.0, 0.0, 1.0], dtype=float)):
        max_kpoint_num = 8000
        disp_k = np.random.rand(1, 3)/self.mp_grid[0]
        k_start = k_start + disp_k
        k_vect1 = k_vect1 + disp_k
        k_vect2 = k_vect2 + disp_k
        k_vect3 = k_vect3 + disp_k
        kmesh = kpoint_generator.mp_generator(max_kpoint_num, k_start, k_vect1, k_vect2, k_vect3, self.mp_grid)
        return kmesh

    @time_execution
    def output_wfc(self):
        lattice_vector, positions = self.read_stru()
        kmesh = self.generate_kpoint()
        m_tb = pyatb.init_tb(
                package = 'ABACUS',
                nspin = self.nspin,
                lattice_constant = 1, # unit is Angstrom
                lattice_vector = lattice_vector,
                max_kpoint_num = 8000,
                isSparse = False,
                HR_route = self.hr,
                HR_unit = 'Ry',
                SR_route = self.sr,
                need_rR = False,
                rR_route = None,
                rR_unit = 'Bohr',
        )
        self.tot_basis_num = m_tb.basis_num
        # 计算 gamma点 -10 ~10 区间内能带最大、最小指标
        efermi = self.efermi
        gamma_point = np.array([[0.0, 0.0, 0.0]], dtype=float)
        eigenvalues = m_tb.tb_solver.diago_H_eigenvaluesOnly(gamma_point)[0]
        # 能带相对费米能
        relative_energy = eigenvalues - efermi
        # 找出所有满足条件的 band index
        selected_indices = np.where((relative_energy >= self.ecut))[0]
        if selected_indices.size == 0:
            raise ValueError("没有任何能带落在指定能量窗口内")
        self.band_cut_index = selected_indices[0] - 1 
        print('Band cut index:', self.band_cut_index, flush=True)

        # 计算所有 k 点的波函数
        print('Start to calculate band stru', flush=True)
        for ik in kmesh:
            self.kvec_d = ik
            eigenvectors, eigenvalues = m_tb.tb_solver.diago_H(ik)
        tem_eigenvectors = eigenvectors
        print('Band stru calculation finshed', flush=True)

        self.unify_orb_num = 27 if self.nspin != 4 else 54
        tot_num = self.atoms.get_global_number_of_atoms() 
        self.tot_kunm = kmesh.total_kpoint_num
        # print(f'全部k点数目为:{self.tot_kunm}', flush=True)
        # 原始数组 shape: (tot_kpoint, nbasis, cal_band_num)
        # 先将数组转置为 (tot_kpoint, cal_band_num, nbasis)
        temp_transposed = np.transpose(tem_eigenvectors, (0, 2, 1))
        # 再 reshape 成二维数组，行数为 tot_kpoint * cal_band_num, 列数为 nbasis
        eigenvectors_2d = temp_transposed.reshape(-1, tem_eigenvectors.shape[1])
        # 创建扩充基组之后的波函数系数矩阵
        self.eigenvectors_enlager =np.zeros([eigenvectors_2d.shape[0], self.unify_orb_num * tot_num], dtype=complex ) 
        for ii in range(tot_num):
            # 对于每一个原子指标 ii 进行轨道补零
            temp_wf_part = np.zeros([eigenvectors_2d.shape[0], self.unify_orb_num], dtype=complex)
            for row in range(self.unify_orb_num):
                if self.index_relation[ii][1][row] == 1:
                    temp_wf_part[:, row] = eigenvectors_2d[:, self.index_relation[ii][2][row]]
            # 对于nspin=1矩阵进行轨道旋转
            if self.nspin == 1:
                temp_wf_part_reordered = temp_wf_part @ self.transform_orb().T
            # 对于nspin=4矩阵进行轨道旋转与重新排列
            elif self.nspin == 4:
                up_indices = np.arange(0, self.unify_orb_num, 2)
                down_indices = np.arange(1, self.unify_orb_num, 2)
                temp_wf_part_reordered = np.concatenate((temp_wf_part[:, up_indices], temp_wf_part[:, down_indices]), axis=1)
                temp_wf_part_reordered = temp_wf_part_reordered @ self.transform_orb().T

            self.eigenvectors_enlager[:, ii*self.unify_orb_num : (ii+1)*self.unify_orb_num] = temp_wf_part_reordered 

        sample = (
            torch.tensor(lattice_vector, dtype=torch.float32),
            torch.tensor(positions, dtype=torch.float32),
            self.tot_basis_num,
            torch.tensor(self.band_cut_index, dtype=torch.int64),
            torch.tensor(self.tot_kunm, dtype=torch.int64),
            torch.tensor(self.kvec_d, dtype=torch.float64),
            torch.tensor(self.eigenvectors_enlager, dtype=torch.complex64),
        )
        sample_path = os.path.join(self.save_path, f"wfc.pth")
        torch.save(sample, sample_path)
        # return self.tot_basis_num, self.band_cut_index, self.tot_kunm, self.kvec_d, self.eigenvectors_enlager


def main():
    pass

if __name__ == "__main__":
    main()