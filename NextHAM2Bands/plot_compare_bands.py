import numpy as np
import os
import argparse
from matplotlib import pyplot as plt
from ase.io import read
import re

def generate_kline(stru_file, kline_density=0.03, tolerance=5e-4, kpath=None, knum=0):
    ase_stru = read(stru_file, format='abacus')
    ase_stru.wrap()
    lattice_vector = np.array(ase_stru.get_cell())
    bandpath = ase_stru.cell.bandpath(path=kpath, density=kline_density, eps=tolerance)
    path_label = bandpath.path
    pattern = re.findall(r'[A-Z][0-9]*', path_label)
    path_label_array = list(pattern)
    cleaned_labels = [label for label in pattern if label != ',']
    special_points = bandpath.special_points
    
    shifted_points = {
        key: value if np.all((value >= -1) & (value <= 1)) else np.mod(value, 1)
        for key, value in special_points.items()
    }
    special_points = shifted_points

    kpt_output = []
    kpoint_label = []
    kpoint_num_in_line = []

    rec_lat_cell = ase_stru.cell.reciprocal()
    rec_lat_matrix = rec_lat_cell[:]
    
    for i in range(len(path_label_array)):
        label = path_label_array[i]
        label_next = path_label_array[i+1] if i+1 < len(path_label_array) else None
        coordinates = special_points.get(label)
        coordinates_next = special_points.get(label_next) if label_next else None
        if coordinates is not None and coordinates_next is not None:
            if knum == 0:
                k_real = coordinates @ rec_lat_matrix
                k_real_next = coordinates_next @ rec_lat_matrix
                distance = np.linalg.norm(k_real - k_real_next)
                density = max(int(distance * (2 * np.pi) / kline_density), 3)
                kpoint_label.append(f"{label}   ")
                kpoint_num_in_line.append(f"{density}  ")
            else:
                kpoint_label.append(f"{label}   ")
                kpoint_num_in_line.append(f"{knum}  ")
        elif coordinates is None:
            pass
        else:
            kpoint_label.append(f"{label}   ")
            kpoint_num_in_line.append(f"{format('1', '<4')}")
            
    return kpoint_label, [int(x) for x in kpoint_num_in_line], lattice_vector

def set_fig(fig, ax, bwidth=1.0, width=1, mysize=10):
    ax.spines['top'].set_linewidth(bwidth)
    ax.spines['right'].set_linewidth(bwidth)
    ax.spines['left'].set_linewidth(bwidth)
    ax.spines['bottom'].set_linewidth(bwidth)
    ax.tick_params(length=5, width=width, labelsize=mysize)

def plot_bands(band1_file, band2_file, stru_file, out_file, mu, emin, emax, 
               kline_density=0.02, adaptive_align=True):
    """
    绘制能带对比图 (包含自适应平移与防乱线处理)
    """
    # 1. 加载数据
    band1 = np.loadtxt(band1_file)
    band2 = np.loadtxt(band2_file)
    
    # 【修复 1】：强制按能量排序，彻底解决连线像条形码一样乱飞的问题
    band1 = np.sort(band1, axis=1)
    band2 = np.sort(band2, axis=1)
    
    # 2. 处理高对称点路径坐标 (直接调用你环境里原有的 generate_kline 函数)
    kpoint_label, kpoint_num_in_line, lattice_vector = generate_kline(stru_file)
    
    high_symmetry_kpoint_x_coor = []
    for ii in range(len(kpoint_label)):
        segment_points_sum = sum(np.array(kpoint_num_in_line[:ii]))
        high_symmetry_kpoint_x_coor.append(segment_points_sum * kline_density)

    # 3. 计算自适应平移量
    shift = 0.0
    if adaptive_align:
        if band1.shape == band2.shape:
            # 【修复 2】：找到费米能级 (mu) 以下的占据带数量，避免高能空带干扰平移方向
            occ_bands_count = np.sum(band1[0, :] <= (mu + 0.1))
            
            if 0 < occ_bands_count < band1.shape[1]:
                # 仅用价带计算修正量 = True - Pred
                shift = np.mean(band1[:, :occ_bands_count] - band2[:, :occ_bands_count])
                print(f"[*] 自适应平移 (基于价带对齐): 预测值需修正 {shift:+.4f} eV")
            else:
                # 备用退化方案
                shift = np.mean(band1 - band2)
                print(f"[*] 自适应平移 (全局对齐): 预测值需修正 {shift:+.4f} eV")
        else:
            print("[!] 警告: 能带维度不匹配，偏移量设为 0。")

    # 4. 数据标准化：预测值加上修正量，然后统一以真值 mu 为 0 点
    band_data1 = band1 - mu
    band_data2 = (band2 + shift) - mu
    
    # 5. 准备 X 轴物理坐标
    k_num = band1.shape[0]
    k_length = (k_num - 1) * kline_density 
    x_coor_array = np.linspace(0, k_length, k_num)

    # 6. 开始绘图
    mysize = 12
    fig, ax = plt.subplots(1, 1, tight_layout=True, figsize=(8, 6))
    
    # 绘制真值矩阵 (蓝色实线) —— 坚决不加 label，防止图例爆炸
    ax.plot(x_coor_array, band_data1, color='blue', linewidth=1.5, linestyle='-', alpha=0.5)
    
    # 绘制预测值矩阵 (红色虚线) —— 同样不加 label
    ax.plot(x_coor_array, band_data2, color='red', linewidth=1.5, linestyle='--', alpha=0.9)
    
    # 【修复 3】：“空线占位法”生成干净图例，只显示预测值的信息
    pred_label = f'Pred (Shifted: {shift:+.2f} eV)' if adaptive_align else 'Predicted Band'
    ax.plot([], [], color='red', linewidth=1.5, linestyle='--', label=pred_label)
    
    # 7. 装饰与设置
    ax.set_title('Band Structure Comparison', fontsize=mysize + 2)
    ax.set_ylabel('E - E$_F$ (eV)', fontsize=mysize)
    ax.set_xlim(0, x_coor_array[-1])
    ax.set_ylim(emin, emax)
    
    # 【修复 4】：安全的 X 轴刻度设置法，避免 ValueError
    ax.set_xticks(high_symmetry_kpoint_x_coor)
    ax.set_xticklabels([l.strip() for l in kpoint_label])
    
    for x in high_symmetry_kpoint_x_coor:
        ax.axvline(x, color="grey", alpha=0.4, lw=0.8, linestyle='--')
    ax.axhline(0.0, color="black", alpha=0.8, lw=1.2, linestyle='-')

    # 渲染图例
    ax.legend(loc="upper right", frameon=False, fontsize=mysize-2)

    # 保存图片
    plt.savefig(out_file, dpi=300)
    plt.close('all')
    print(f"[OK] 能带对比图已保存: {out_file}")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot and compare two band1.txt files.")
    parser.add_argument("--band1", type=str, required=True, help="Path to first band file")
    parser.add_argument("--band2", type=str, required=True, help="Path to second band file")
    parser.add_argument("--stru", type=str, required=True, help="Path to STRU file for k-points")
    parser.add_argument("--out", type=str, default="compare_bands.pdf", help="Output PDF path")
    parser.add_argument("--mu", type=float, default=2.1700249049, help="Fermi level (mu)")
    parser.add_argument("--emin", type=float, default=-10, help="Min energy limit")
    parser.add_argument("--emax", type=float, default=10, help="Max energy limit")
    
    args = parser.parse_args()
    
    plot_bands(args.band1, args.band2, args.stru, args.out, args.mu, args.emin, args.emax)
