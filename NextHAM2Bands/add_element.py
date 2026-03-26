# 把截断半径以外的数据用弱标签哈密顿量数据补全，获取新的数据
import numpy as np
import os 
import re
from ase.io import read, write
from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix
import sys
from pathlib import Path
try:
    from timing_utils import time_execution
except ImportError:
    sys.path.append(str(Path(__file__).parent.parent))
    from timing_utils import time_execution


class XR_matrix:
    def __init__(self, nspin, XR_fileName):
        self.nspin = nspin
        self.XR_fileName = XR_fileName
        self.read_file()

    @time_execution
    def read_file(self):
        # Auto-detect binary or text mode
        is_binary = False
        import struct
        with open(self.XR_fileName, 'rb') as f:
            header = f.read(4)
            if header != b'STEP':
                is_binary = True
        
        if not is_binary:
            with open(self.XR_fileName, 'r') as fread:
                fread.readline() # STEP
                line = fread.readline()
                self.basis_num = int(line.split()[-1])
                line = fread.readline()
                self.R_num = int(line.split()[-1])
                self.R_direct_coor = np.zeros([self.R_num, 3], dtype=int)
                if self.nspin != 4:
                    self.XR = np.zeros([self.R_num, self.basis_num, self.basis_num], dtype=float)
                else:
                    self.XR = np.zeros([self.R_num, self.basis_num, self.basis_num], dtype=complex)

                for iR in range(self.R_num):
                    line = fread.readline().split()
                    self.R_direct_coor[iR, 0] = int(line[0])
                    self.R_direct_coor[iR, 1] = int(line[1])
                    self.R_direct_coor[iR, 2] = int(line[2])
                    data_size = int(line[3])
                    
                    if self.nspin != 4:
                        data = np.zeros((data_size,), dtype=float)
                    else:
                        data = np.zeros((data_size,), dtype=complex)

                    indices = np.zeros((data_size,), dtype=int)
                    indptr = np.zeros((self.basis_num+1,), dtype=int)

                    if data_size != 0:
                        if self.nspin != 4:
                            line = fread.readline().split()
                            if (len(line) != data_size):
                                print("size = ", len(line), " data_size = ", data_size)
                            for index in range(data_size):
                                data[index] = float(line[index])
                        else:
                            line = re.findall('[(](.*?)[])]', fread.readline())
                            for index in range(data_size):
                                value = line[index].split(',')
                                data[index] = complex( float(value[0]), float(value[1]) ) 

                        line = fread.readline().split()
                        for index in range(data_size):
                            indices[index] = int(line[index])

                        line = fread.readline().split()
                        for index in range(self.basis_num+1):
                            indptr[index] = int(line[index])

                    self.XR[iR] = csr_matrix((data, indices, indptr), shape=(self.basis_num, self.basis_num)).toarray()
        else:
            with open(self.XR_fileName, 'rb') as f:
                # Binary format: step(int), nlocal(int), nR(int)
                raw_header = f.read(12)
                if not raw_header: return
                step, nlocal, nR = struct.unpack('iii', raw_header)
                self.basis_num = nlocal
                self.R_num = nR
                self.R_direct_coor = np.zeros([self.R_num, 3], dtype=int)
                if self.nspin != 4:
                    self.XR = np.zeros([self.R_num, self.basis_num, self.basis_num], dtype=float)
                else:
                    self.XR = np.zeros([self.R_num, self.basis_num, self.basis_num], dtype=complex)
                
                for iR in range(nR):
                    raw_R = f.read(16) # rx, ry, rz, nnz
                    if not raw_R: break
                    rx, ry, rz, nnz = struct.unpack('iiii', raw_R)
                    self.R_direct_coor[iR] = [rx, ry, rz]
                    
                    if nnz == 0:
                        f.read((self.basis_num + 1) * 4)
                        continue
                    
                    if self.nspin == 4:
                        data = np.frombuffer(f.read(nnz * 16), dtype=complex)
                    else:
                        data = np.frombuffer(f.read(nnz * 8), dtype=float)
                    
                    indices = np.frombuffer(f.read(nnz * 4), dtype=np.int32)
                    indptr = np.frombuffer(f.read((self.basis_num + 1) * 4), dtype=np.int32)
                    
                    self.XR[iR] = csr_matrix((data, indices, indptr), shape=(self.basis_num, self.basis_num)).toarray()


class add_hs_matrix:

    def __init__(self, nspin, stru_file, hr1, hr2, save_path):
        self.nspin = nspin
        self.stru_file = stru_file
        self.hr1 = hr1
        self.hr2 = hr2
        self.save_path = save_path
        self.orb_origin = {'H': 5,   'He': 5,  'Li': 7,  'Be': 7,  'B': 13,  'C': 13,  'N': 13,  'O': 13,  'F': 13, 'Ne': 13, 
                           'Na': 15, 'Mg': 15, 'Al': 13, 'Si': 13, 'P': 13,  'S': 13,  'Cl': 13, 'Ar': 13, 'K': 15, 
                           'Sc': 27, 'V': 27,  'Fe': 27, 'Co': 27, 'Ni': 27, 'Cu': 27, 'Zn': 27, 'Ga': 25, 'Ge': 25, 
                           'Br': 13, 'Y': 27,  'Nb': 27, 'Mo': 27, 'Pd': 25, 'Ag': 27, 'Cd': 27, 'In': 25, 'Sn': 25, 
                           'Sb': 25, 'Te': 25, 'I': 13,  'Xe': 13, 'Hf': 27, 'Ta': 27, 'Re': 27, 'Pt': 27, 'Au': 27, 
                           'Hg': 27, 'Tl': 25, 'Pb': 25, 'Bi': 25, 'Ca': 15, 'Ti': 27, 'Cr': 27, 'Mn': 27, 'Kr': 13, 
                           'Rb': 15, 'Sr': 15, 'Zr': 27, 'Tc': 27, 'Ru': 27, 'Rh': 27, 'Cs': 15, 'Ba': 15, 'W': 27, 
                           'Os': 27, 'Ir': 27, 'As': 13, 'Se': 13}        

    @time_execution
    def read_stru(self):
        self.atoms = read(self.stru_file, format='abacus')
        self.atoms.wrap()
        # 创建一个字典，用以存储结构的原子指标和对应的哈密顿量起始和终止指标，字典类型为 ii item {element: [start, end]}
        index_dict = {}
        current_index = 0
        for ii, atom in enumerate(self.atoms):
            element = atom.symbol
            n_orbitals = self.orb_origin[element]
            start = current_index
            if self.nspin == 4:
                n_orbitals = n_orbitals * 2
            end = current_index + n_orbitals
            index_dict[ii] = [start, end]
            current_index = end  # 更新到下一个原子的起始轨道索引
        # print('index_dict:', index_dict)
        return index_dict
    
    @time_execution
    def add_matrxi_element(self):
        index_dict = self.read_stru()
        pred_hr = XR_matrix(4, self.hr1)
        abacus_hr = XR_matrix(4, self.hr2)
        R_coor1 = pred_hr.R_direct_coor
        R_coor2 = abacus_hr.R_direct_coor
        R_num1 = pred_hr.R_num
        R_num2 = abacus_hr.R_num
        basis_num = abacus_hr.basis_num

        add_hr = abacus_hr.XR
        R_cut = 7
        tot_num = self.atoms.get_global_number_of_atoms() 
        for iR2 in range(R_num2):
            # 判断 R_coor2[iR] 指标是否存在 R_coor1 对应，如果不存在对应，全部用弱标签数据补全
            for iR1 in range(R_num1):
                if R_coor2[iR2][0] == R_coor1[iR1][0] and R_coor2[iR2][1] == R_coor1[iR1][1] and R_coor2[iR2][2] == R_coor1[iR1][2]:
                    for ii in range(tot_num):
                        for jj in range(tot_num):
                            posit_ii = self.atoms.positions[ii]
                            posit_jj = self.atoms.positions[jj]
                            distance = R_coor2[iR2][0] * self.atoms.cell[0] + R_coor2[iR2][1] * self.atoms.cell[1]  + R_coor2[iR2][2] * self.atoms.cell[2] + posit_jj - posit_ii
                            if np.linalg.norm(distance) <= R_cut:
                                add_hr[iR2, index_dict[ii][0]:index_dict[ii][1], index_dict[jj][0]:index_dict[jj][1]] = pred_hr.XR[iR1, index_dict[ii][0]:index_dict[ii][1], index_dict[jj][0]:index_dict[jj][1]]
    

        # 文件输出
        hr_add_file = os.path.join(self.save_path, 'hr_tot')        
        with open(hr_add_file, 'w' ) as f1 :
            # 写入文件头信息
            f1.write('STEP: 0' + '\n')
            f1.write(f'Matrix Dimension of H(R): {basis_num}' + '\n')
            f1.write(f'Matrix number of H(R): {R_num2}' + '\n')

            for iR in range(R_num2):
                sparse_data_hr = csr_matrix(add_hr[iR])
                
                # 写文件其它内容信息
                f1.write(f'{R_coor2[iR][0]:.0f} {R_coor2[iR][1]:.0f} {R_coor2[iR][2]:.0f} {len(sparse_data_hr.data)}\n')
                if len(sparse_data_hr.data) == 0:
                    pass
                else:
                    # 将 data 数组转换为字符串，每个元素之间用空格分隔，并写入一行
                    if self.nspin == 1:
                        data_str = " ".join(map(str, sparse_data_hr.data))
                        f1.write(f"{data_str}\n")
                    elif self.nspin == 4:
                        data_str = " ".join("({:.8e},{:.8e})".format(c.real, c.imag) for c in sparse_data_hr.data)
                        f1.write(f"{data_str}\n")
                    
                    # 将 indices 数组转换为字符串，每个元素之间用空格分隔，并写入一行
                    indices_str = " ".join(map(str, sparse_data_hr.indices))
                    f1.write(f"{indices_str}\n")
                    # 将 indptr 数组转换为字符串，每个元素之间用空格分隔，并写入一行
                    indptr_str = " ".join(map(str, sparse_data_hr.indptr))
                    f1.write(f"{indptr_str}\n")
            

def main():
    pass

if __name__ == "__main__":
    main()



# # mp-22120

# init_path = '/home/zujiandai/file-test/24.12.5-HCP/test/25.4.9-test-band/mp-559437'
# data_path = '/home/zujiandai/file-test/24.12.5-HCP/soc_data_collect/soc-data1/mp-559437'
# save_path = init_path
# hr1 = os.path.join(init_path, 'precise_H_pred')
# # hr1 = '/home/zujiandai/file-test/24.12.5-HCP/test/25.4.11-test-rcut-band/soc-data1/mp-10030/hr_cut.csr'
# hr2 = os.path.join(data_path, 'data-HR-sparse_SPIN0_weak.csr')
# sr2 = os.path.join(data_path, 'data-SR-sparse_SPIN0.csr')
# nspin = 4
# stru_name = 'mp-559437'

# my_class = add_hs_matrix(nspin, stru_name, hr1, hr2, sr2, save_path)
# stru_path = my_class.search_stru()
# # print(stru_path)
# my_class.read_stru()
# my_class.add_matrxi_element()

# path_data = '/home/zujiandai/file-test/24.12.5-HCP/test/25.4.16-test-band'
# for filename in os.listdir(path_data):
    
#     # if filename.startswith('mp-'):
#     if filename == 'mp-1001615':
#         stru_name = filename
#         print(stru_name)
#         path_data_now = os.path.join(path_data, stru_name)
#         save_path = path_data_now 
#         hr1 = os.path.join(path_data_now, 'precise_H')
#         hr2 = None
#         sr2 = None
#         nspin = 4
#         # 第一次调用类，获取初始结构文件信息
#         my_class = add_hs_matrix(nspin, stru_name, hr1, hr2, sr2, save_path)
#         stru_path = my_class.search_stru()
#         path_origin = stru_path .rsplit('/', 1)[0]
#         # 第二次调用类，正常进行计算
#         hr2 = os.path.join(path_origin, 'data-HR-sparse_SPIN0_strong.csr')
#         sr2 = os.path.join(path_origin, 'data-SR-sparse_SPIN0.csr')
#         my_class = add_hs_matrix(nspin, stru_name, hr1, hr2, sr2, save_path)
#         stru_path = my_class.search_stru()
#         my_class.read_stru()
#         my_class.add_matrxi_element()
#         print(stru_name,'补充哈密顿量矩阵元完成')
        
