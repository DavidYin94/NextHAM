from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix
# from ase.io import read, write
import numpy as np
import os
import re
import sys
import time
import torch

class get_hamiltion_data:
    def __init__(self, nspin, cal_path):
        self.nspin = nspin
        self.cal_path = cal_path
        self.orb_origin = {'H': 5,   'He': 5,  'Li': 7,  'Be': 7,  'B': 13,  'C': 13,  'N': 13,  'O': 13,  'F': 13, 'Ne': 13, 
                           'Na': 15, 'Mg': 15, 'Al': 13, 'Si': 13, 'P': 13,  'S': 13,  'Cl': 13, 'Ar': 13, 'K': 15, 
                           'Sc': 27, 'V': 27,  'Fe': 27, 'Co': 27, 'Ni': 27, 'Cu': 27, 'Zn': 27, 'Ga': 25, 'Ge': 25, 
                           'Br': 13, 'Y': 27,  'Nb': 27, 'Mo': 27, 'Pd': 25, 'Ag': 27, 'Cd': 27, 'In': 25, 'Sn': 25, 
                           'Sb': 25, 'Te': 25, 'I': 13,  'Xe': 13, 'Hf': 27, 'Ta': 27, 'Re': 27, 'Pt': 27, 'Au': 27, 
                           'Hg': 27, 'Tl': 25, 'Pb': 25, 'Bi': 25, 'Ca': 15, 'Ti': 27, 'Cr': 27, 'Mn': 27, 'Kr': 13, 
                           'Rb': 15, 'Sr': 15, 'Zr': 27, 'Tc': 27, 'Ru': 27, 'Rh': 27, 'Cs': 15, 'Ba': 15, 'W': 27, 
                           'Os': 27, 'Ir': 27, 'As': 13, 'Se': 13}
    
    def read_stru(self):
        stru_file = os.path.join(self.cal_path, 'STRU')
        # 读取 STRU 文件
        self.atoms = read(stru_file, format='abacus')
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
    
    def read_data(self):
     
        strong_label_path = os.path.join(self.cal_path, 'data-HR-sparse_SPIN0_strong.csr')
        weak_label_path   = os.path.join(self.cal_path, 'data-HR-sparse_SPIN0_weak.csr')
        # 检查文件是否存在
        if os.path.exists(strong_label_path) and os.path.exists(weak_label_path):
            print(f'{self.cal_path}目录下存在相应精标签、弱标签H矩阵文件')
        else:
            print(f'{self.cal_path}目录下不存在相应精标签、弱标签H矩阵文件, 报错退出')
            sys.exit(1)  # 退出程序，返回错误代码 1

        # 先读取精标签数据，最后读取弱标签数据
        # 创建两个列表，一个存储指标信息[Rx, Ry, Rz, i, j, dx, dy, dz] 1*8 矩阵形式
        # 另一个为3*27*27 维度矩阵，分别存储不同pair对的精、弱标签hamiltion和patching矩阵
        self.index_list = []
        self.matrix_list = []
        self.unify_orb_num = 27 
        if self.nspin == 4:
            self.unify_orb_num = 2 * self.unify_orb_num
        
        # 先预先读取一遍矩阵文件，把所有R指标信息存储下来
        # 读取精标签数据前三行信息
        with open(strong_label_path, 'r') as fread_strong:
            line_strong = fread_strong.readline()
            line_strong = fread_strong.readline()
            basis_num = int(line_strong.split()[-1])
            if basis_num != self.stru_dim:
                print(f" STRU 结构计算矩阵维度和{strong_label_path} 矩阵维度不一致，请检查结果 ")
                sys.exit(2)  # 退出程序，返回错误代码 2
            line_strong = fread_strong.readline()
            R_num_strong = int(line_strong.split()[-1])  
            R_direct_coor_strong = np.zeros([R_num_strong, 3], dtype=int)
            for iR in range(R_num_strong):
                # 获取 R 指标
                line = fread_strong.readline().split()
                R_direct_coor_strong[iR, 0] = int(line[0])
                R_direct_coor_strong[iR, 1] = int(line[1])
                R_direct_coor_strong[iR, 2] = int(line[2])
                data_size = int(line[3])
                if data_size != 0:
                    line_strong = fread_strong.readline()
                    line_strong = fread_strong.readline()
                    line_strong = fread_strong.readline()

        # 读取弱标签数据数据前三行信息
        with open(weak_label_path, 'r') as fread_weak:
            line_weak = fread_weak.readline()
            line_weak = fread_weak.readline()
            basis_num = int(line_weak.split()[-1])
            if basis_num != self.stru_dim:
                print(f" STRU 结构计算矩阵维度和{weak_label_path} 矩阵维度不一致，请检查结果 ")
                sys.exit(2)  # 退出程序，返回错误代码 2
            line_weak = fread_weak.readline()
            R_num_weak = int(line_weak.split()[-1])
            R_direct_coor_weak = np.zeros([R_num_weak, 3], dtype=int)
            for iR in range(R_num_weak):
                line = fread_weak.readline().split()
                R_direct_coor_weak[iR, 0] = int(line[0])
                R_direct_coor_weak[iR, 1] = int(line[1])
                R_direct_coor_weak[iR, 2] = int(line[2])
                data_size = int(line[3])
                if data_size != 0:
                    line_weak = fread_weak.readline()
                    line_weak = fread_weak.readline()
                    line_weak = fread_weak.readline()
        
        ## 筛选出所有重复R指标
        R_coor_both= []
        for iR_strong in range(R_num_strong):
            for iR_weak in range(R_num_weak):
                if R_direct_coor_strong[iR_strong, 0] == R_direct_coor_weak[iR_weak, 0] and R_direct_coor_strong[iR_strong, 1] == R_direct_coor_weak[iR_weak, 1] \
                and R_direct_coor_strong[iR_strong, 2] == R_direct_coor_weak[iR_weak, 2]:
                    R_coor_both.append(R_direct_coor_strong[iR_strong])

        # print(R_coor_both)
        # print(len(R_coor_both))
        # 按照重复 R 指标序列，同时读取精/弱标签哈密顿量矩阵

        with open(strong_label_path, 'r') as fread_strong , open(weak_label_path, 'r') as fread_weak:
            line_strong = fread_strong.readline()
            line_strong = fread_strong.readline()
            line_strong = fread_strong.readline()
            line_weak = fread_weak.readline()
            line_weak = fread_weak.readline()
            line_weak = fread_weak.readline()

            for iR in range(len(R_coor_both)):
                # print(f'iR 指标为{R_coor_both[iR]}' )
                R_stong_temp = np.zeros([3,], dtype=int)
                line_strong = fread_strong.readline().split()
                R_stong_temp[0] = int(line_strong[0])
                R_stong_temp[1] = int(line_strong[1])
                R_stong_temp[2] = int(line_strong[2])
                data_size = int(line_strong[3])
                
                while True:
                    if R_stong_temp[0] == R_coor_both[iR][0] and R_stong_temp[1] == R_coor_both[iR][1] and R_stong_temp[2] == R_coor_both[iR][2]:
                        break
                    else:
                        if data_size != 0:
                            line_strong = fread_strong.readline()
                            line_strong = fread_strong.readline()
                            line_strong = fread_strong.readline()
                        # get new R_num
                        line_strong = fread_strong.readline().split()
                        R_stong_temp[0] = int(line_strong[0])
                        R_stong_temp[1] = int(line_strong[1])
                        R_stong_temp[2] = int(line_strong[2])
                        data_size = int(line_strong[3])
                # print(f'精标签指标为{R_stong_temp}')

                if self.nspin != 4:
                    data = np.zeros((data_size, ), dtype=float)
                else:
                    data = np.zeros((data_size, ), dtype=complex)
                indices = np.zeros((data_size, ), dtype=int)
                indptr = np.zeros((basis_num+1, ), dtype=int)
                
                if data_size != 0:

                    # 创建每一个R指标下临时存放数据矩阵
                    if self.nspin != 4:
                        matrix_R_strong = np.zeros([basis_num, basis_num], dtype=float)
                    else:
                        matrix_R_strong = np.zeros([basis_num, basis_num], dtype=complex)
                    # 稀疏矩阵数据行读取
                    if self.nspin != 4:
                        line_strong = fread_strong.readline().split()
                        for index in range(data_size):
                            data[index] = float(line_strong[index])
                    else:
                        line_strong = re.findall('[(](.*?)[])]', fread_strong.readline())
                        for index in range(data_size):
                            value = line_strong[index].split(',')
                            data[index] = complex( float(value[0]), float(value[1]) )

                    # 稀疏矩阵列指标读取
                    line_strong = fread_strong.readline().split()
                    for index in range(data_size):
                        indices[index] = int(line_strong[index])
    
                    # 稀疏矩阵行偏移量读取
                    line_strong = fread_strong.readline().split()
                    for index in range(basis_num+1):
                        indptr[index] = int(line_strong[index])

                    
                    # 转化为方块矩阵
                    matrix_R_strong = csr_matrix((data, indices, indptr), shape=(basis_num, basis_num)).toarray()
                
                        

                R_weak_temp = np.zeros([3,], dtype=int)
                line_weak= fread_weak.readline().split()
                R_weak_temp[0] = int(line_weak[0])
                R_weak_temp[1] = int(line_weak[1])
                R_weak_temp[2] = int(line_weak[2])
                data_size = int(line_weak[3])

                while True:
                    if R_weak_temp[0] == R_coor_both[iR][0] and R_weak_temp[1] == R_coor_both[iR][1] and R_weak_temp[2] == R_coor_both[iR][2]:
                        break
                    else:
                        if data_size != 0:
                            line_weak = fread_weak.readline()
                            line_weak = fread_weak.readline()
                            line_weak = fread_weak.readline()
                        # get new R_num
                        line_weak= fread_weak.readline().split()
                        R_weak_temp[0] = int(line_weak[0])
                        R_weak_temp[1] = int(line_weak[1])
                        R_weak_temp[2] = int(line_weak[2])
                        data_size = int(line_weak[3])
                # print(f'弱标签指标为{R_stong_temp}')
                # print('\n')
                if self.nspin != 4:
                    data = np.zeros((data_size, ), dtype=float)
                else:
                    data = np.zeros((data_size, ), dtype=complex)
                indices = np.zeros((data_size, ), dtype=int)
                indptr = np.zeros((basis_num+1, ), dtype=int)

                if data_size != 0:
                    # 创建每一个R指标下临时存放数据矩阵
                    if self.nspin != 4:
                        matrix_R_weak= np.zeros([basis_num, basis_num], dtype=float)
                    else:
                        matrix_R_weak = np.zeros([basis_num, basis_num], dtype=complex)
                    # 稀疏矩阵数据行读取
                    if self.nspin != 4:
                        line_weak = fread_weak.readline().split()
                        for index in range(data_size):
                            data[index] = float(line_weak[index])
                    else:
                        line_weak = re.findall('[(](.*?)[])]', fread_weak.readline())
                        for index in range(data_size):
                            value = line_weak[index].split(',')
                            data[index] = complex( float(value[0]), float(value[1]) )

                    # 稀疏矩阵列指标读取
                    line_weak = fread_weak.readline().split()
                    for index in range(data_size):
                        indices[index] = int(line_weak[index])
    
                    # 稀疏矩阵行偏移量读取
                    line_weak = fread_weak.readline().split()
                    for index in range(basis_num+1):
                        indptr[index] = int(line_weak[index])
                    
                    # 转化为方块矩阵
                    matrix_R_weak = csr_matrix((data, indices, indptr), shape=(basis_num, basis_num)).toarray()

            # 获取pair对信息，判定距离
                tot_num = self.atoms.get_global_number_of_atoms() 
                R_cut = 8
                for ii in range(tot_num):
                    for jj in range(tot_num):
                        posit_ii = self.atoms.positions[ii]
                        posit_jj = self.atoms.positions[jj]
                        distance = R_coor_both[iR][0] * self.atoms.cell[0] + R_coor_both[iR][1] * self.atoms.cell[1]  + R_coor_both[iR][2] * self.atoms.cell[2] + posit_jj - posit_ii
                        if np.linalg.norm(distance) < R_cut:
                            # 创建列表指标矩阵，并且添加到 self.index_list 中
                            temp_label = np.zeros( [8, ], dtype=float )
                            if self.nspin != 4:
                                temp_data = np.zeros([3, self.unify_orb_num, self.unify_orb_num], dtype=float)
                            else:
                                temp_data = np.zeros([3, self.unify_orb_num, self.unify_orb_num], dtype=complex)

                            temp_label[0:3] = R_coor_both[iR]
                            temp_label[3] = ii
                            temp_label[4] = jj
                            temp_label[5:8] = distance
                            self.index_list.append(temp_label)
                            for row in range(self.unify_orb_num):
                                for colume in range(self.unify_orb_num):
                                    temp_data[2, row, colume] = self.index_relation[ii][1][row] * self.index_relation[jj][1][colume]
                                    if temp_data[2, row, colume] == 1:
                                        temp_data[0, row, colume] = matrix_R_strong[self.index_relation[ii][2][row], self.index_relation[jj][2][colume]]
                                        temp_data[1, row, colume] = matrix_R_weak[self.index_relation[ii][2][row], self.index_relation[jj][2][colume]]

                               
                            self.matrix_list.append(temp_data)

        self.matrix_tensor = torch.tensor(self.matrix_list, dtype = torch.float32)
        label_tensor = self.matrix_tensor[:,0]-self.matrix_tensor[:,1]
        descriptor_tensor = self.matrix_tensor[:,1]
        mask_tensor = self.matrix_tensor[:,2]
        self.index_tensor = torch.tensor(self.index_list, dtype = torch.float32)
        edge_vec = self.index_tensor[:,5:-1]
        node_dict = {}
        edge_src = []
        edge_dst = []
        for d_idx in range(len(self.index_tensor)):
            rx, ry, rz, i, j = int(self.index_tensor[d_idx][0]), int(self.index_tensor[d_idx][1]), int(self.index_tensor[d_idx][2]), int(self.index_tensor[d_idx][3]), int(self.index_tensor[d_idx][4])
            i_str = str(i)
            j_str = str(rx)+'_'+str(ry)+'_'+str(rz)+'_'+str(j)
            if not i_str in node_dict:
                i_ndx = len(node_dict)
                node_dict[i_str] = i_ndx
            i_ndx = node_dict[i_str]
            edge_src.append(i_ndx)
            if not j_str in node_dict:
                j_idx = len(node_dict)
                node_dict[j_str] = j_idx
            j_ndx = node_dict[j_str]
            edge_dst.append(j_ndx)
        edge_src = torch.tensor(edge_src).long()
        edge_dst = torch.tensor(edge_dst).long()
        self.data = [descriptor_tensor, mask_tensor, edge_vec, edge_src, edge_dst]
        self.label = [label_tensor]
        return self.data, self.label  

def main():
    start_time = time.time()  # 记录起始时间
    path = '/home/ubuntu/code/universal_model/datasets/processed_data/train/mp-24'
    hamilton_data = get_hamiltion_data(1, path)
    hamilton_data.read_stru()
    data, label = hamilton_data.read_data()
    end_time = time.time()    # 记录结束时间
    print(f"函数运行时间为 {end_time - start_time:.4f} 秒")

if __name__ == "__main__":
    main()





