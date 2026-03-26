import pyatb
import numpy as np
import re
import os
from ase.io import read, write
from pyatb.kpt import kpoint_generator
from pyatb import RANK, COMM, SIZE, OUTPUT_PATH, RUNNING_LOG, timer
from pyatb.parallel import op_gather_numpy
from matplotlib import pyplot as plt
import sys
from pathlib import Path
try:
    from timing_utils import time_execution
except ImportError:
    sys.path.append(str(Path(__file__).parent.parent))
    from timing_utils import time_execution




class plot_band: 
    def __init__(self, nspin, stru_file, hr1, sr1, emin, emax, mu, save_path):
        self.nspin = nspin
        self.stru_file = stru_file
        self.hr1 = hr1
        self.sr1 = sr1
        self.emin = emin
        self.emax = emax
        self.mu = mu
        self.save_path = save_path
    
    @time_execution
    def generate_kline(self, kline_density = 0.03, tolerance=5e-4, kpath = None, knum=0 ):
        stru_file = self.stru_file
        # 读取 STRU 文件
        ase_stru = read(stru_file, format='abacus')
        ase_stru.wrap()
        lattice_vector = np.array(ase_stru.get_cell())
        # print(lattice_vector)
        bandpath = ase_stru.cell.bandpath(path = kpath, density=kline_density , eps = tolerance)
        path_label = bandpath.path
        # 使用正则表达式匹配字母和数字组合，例如 'X1'
        pattern = re.findall(r'[A-Z][0-9]*', path_label)
        path_label_array = list(pattern)
        cleaned_labels = [label for label in pattern if label != ',']
        special_points = bandpath.special_points
        # 对每个坐标进行检查和变换 只对不在 [-1, 1] 范围内的坐标应用 mod 1 操作
        shifted_points = {
            key: value if np.all((value >= -1) & (value <= 1)) else np.mod(value, 1)
            for key, value in special_points.items()
        }
        special_points = shifted_points

        kpt_output = []
        kpoint_label = []
        kpoint_num_in_line = []

        rec_lat_cell = ase_stru.cell.reciprocal()
        rec_lat_matrix  = rec_lat_cell[:]
        # print(rec_lat_matrix)
        for i in range(len(path_label_array)):
            # 在这里使用 i 和 path_label_array[i] 进行操作
            label = path_label_array[i]
            label_next = path_label_array[i+1] if i+1 < len(path_label_array) else None
            coordinates = special_points.get(label)
            coordinates_next = special_points.get(label_next) if label_next else None  # 如果获取不到值也置为None
            if coordinates is not None and coordinates_next is not None:
                if knum == 0: # 如果knum为0，则计算两点之间的k点数
                    # 计算两点之间的距离
                    k_real = coordinates @ rec_lat_matrix
                    k_real_next = coordinates_next @ rec_lat_matrix
                    distance_in_reciprocal = np.linalg.norm(k_real - k_real_next)
                    distance = distance_in_reciprocal
                    # print(f"Distance in reciprocal space is {distance_in_reciprocal}")
                    # 计算两点之间的k点数，如果小于3则置为3
                    density = max(int(distance * (2* np.pi) / kline_density ), 3)
                    kpt_output.append(f"{'  '.join([f'{coord: .10f}' for coord in coordinates])}  {format(density, '<4')}   # {label}")
                    kpoint_label.append(f"{label}   ")
                    kpoint_num_in_line.append(f"{density}  ")
                else: # 如果knum不为0，则直接使用knum设置k点数
                    kpt_output.append(f"{'  '.join([f'{coord: .10f}' for coord in coordinates])}  {format(knum, '<4')}   # {label}")
                    kpoint_label.append(f"{label}   ")
                    kpoint_num_in_line.append(f"{knum}  ")
            elif coordinates is None:
                # print("no coord this line")
                pass
            else:
                kpt_output.append(f"{'  '.join([f'{coord: .10f}' for coord in coordinates])}  {format('1', '<4')}   # {label}")
                kpoint_label.append(f"{label}   ")
                kpoint_num_in_line.append(f"{ format('1', '<4') }")
        
        kpt_output.append(f' kpoint_label{" " * 20}{",".join(cleaned_labels)}')
        
        high_symm_num = len(kpoint_label)
        kpoint_num_in_line_list = []
        high_symmetry_kpoint_list = []
        for ii in range(len(path_label_array)):
            label = path_label_array[ii]
            coordinates = special_points.get(label)
            high_symmetry_kpoint_list.append(coordinates)
            kpoint_num_in_line_list.append(int(kpoint_num_in_line[ii]))
        high_symmetry_kpoint = np.stack(high_symmetry_kpoint_list, axis=0)
        kpoint_num_in_line = np.array(kpoint_num_in_line_list)
        # print(high_symmetry_kpoint, flush=True)
        # print(kpoint_label, flush=True)
        # print(kpoint_num_in_line_list)
        kline = kpoint_generator.line_generator(8000, high_symmetry_kpoint, kpoint_num_in_line)
        # print(kpt_output[0])
        # print(kpoint_label)
        return kpt_output, kpoint_label, kpoint_num_in_line, kline, lattice_vector
    
    @time_execution
    def cal_band(self):
        kpt_output, kpoint_label, kpoint_num_in_line, kline, lattice_vector = self.generate_kline()

        m_tb1 = pyatb.init_tb(
                package = 'ABACUS',
                nspin = self.nspin,
                lattice_constant = 1, # unit is Angstrom
                lattice_vector = lattice_vector,
                max_kpoint_num = 8000,
                isSparse = False,
                HR_route = self.hr1,
                HR_unit = 'Ry',
                SR_route = self.sr1,
                need_rR = False,
                rR_route = None,
                rR_unit = 'Bohr',
        )

        if RANK == 0:
            print('Start to calculate band structure', flush=True)
        COMM.Barrier()

        for ik in kline:
            # time_start = time.time()

            ik_process = kpoint_generator.kpoints_in_different_process(SIZE, RANK, ik)
            kpoint_num = ik_process.k_direct_coor_local.shape[0]

            if kpoint_num:
                eigenvalues1= m_tb1.tb_solver.diago_H_eigenvaluesOnly(ik_process.k_direct_coor_local)
            else:
                eigenvalues1 = np.zeros((0, m_tb1.tb_solver.basis_num), dtype=np.float64)

        band1 = COMM.reduce(eigenvalues1, root=0, op=op_gather_numpy)

    
        if RANK == 0:
            print('Band structure calculated finished', flush=True)
            band1_name = os.path.join(self.save_path, 'band1.txt')
            np.savetxt(band1_name, band1)
        COMM.Barrier()    
                
        return band1
    
    def set_fig(self, fig, ax, bwidth=1.0, width=1, mysize=10):
        ax.spines['top'].set_linewidth(bwidth)
        ax.spines['right'].set_linewidth(bwidth)
        ax.spines['left'].set_linewidth(bwidth)
        ax.spines['bottom'].set_linewidth(bwidth)
        ax.tick_params(length=5, width=width, labelsize=mysize)

    @time_execution
    def plot_pic(self, kline_density = 0.02):
        band1 = np.loadtxt(os.path.join(self.save_path, 'band1.txt'))
        kpt_output, kpoint_label, kpoint_num_in_line, kline, lattice_vector = self.generate_kline()
        band_data1 = band1 - self.mu
        y_min = self.emin # eV
        y_max = self.emax # eV
        fig_name = os.path.join(self.save_path, 'band1.pdf')
        k_num = band1.shape[0]
        k_length = k_num * kline_density
        x_coor_array = np.linspace(0, k_length, k_num)
        high_symmetry_kpoint_labels = kpoint_label
        high_symmetry_kpoint_x_coor = []
        for ii in range(len(high_symmetry_kpoint_labels)):
            high_symmetry_kpoint_x_coor.append(sum(kpoint_num_in_line[:ii]*kline_density))

        mysize=10
        fig, ax = plt.subplots(1, 1, tight_layout=True)
        self.set_fig(fig, ax,  mysize=mysize)
        linewidth = [1.0, 1.0]
        color = ['red', 'blue']
        linestyle = ['-', '--']  

        ax.plot(x_coor_array, band_data1, color=color[0], linewidth=linewidth[0], linestyle=linestyle[0])
        label = ['pred_tot']
        ax.plot([], [], color=color[0], linewidth=linewidth[0], linestyle=linestyle[0], label=label[0])
        ax.legend(loc="upper right")
        ax.set_title('Band Structure', fontsize=mysize)
        # ax.set_xlabel('High Symmetry Points', fontsize=mysize)
        ax.set_ylabel('E - E$_F$ (eV)', fontsize=mysize)
        ax.set_xlim(0, x_coor_array[-1])
        ax.set_ylim(y_min, y_max)
        plt.xticks(high_symmetry_kpoint_x_coor, high_symmetry_kpoint_labels)     
        for i in high_symmetry_kpoint_x_coor:
            plt.axvline(i, color ="grey", alpha = 0.5, lw = 1, linestyle='--') # draw vertical lines at each kpoints

            ax.axhline(0.0, color ="black", alpha = 1, lw = 1, linestyle='--')

        plt.savefig(fig_name)
        plt.close('all')   