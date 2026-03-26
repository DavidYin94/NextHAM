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

def plot_bands(band1_file, band2_file, stru_file, out_file, mu, emin, emax, kline_density=0.02):
    band1 = np.loadtxt(band1_file)
    band2 = np.loadtxt(band2_file)
    
    kpoint_label, kpoint_num_in_line, lattice_vector = generate_kline(stru_file)
    
    band_data1 = band1 - mu
    band_data2 = band2 - mu
    
    k_num = band1.shape[0]
    k_length = k_num * kline_density
    x_coor_array = np.linspace(0, k_length, k_num)
    
    high_symmetry_kpoint_labels = kpoint_label
    high_symmetry_kpoint_x_coor = []
    for ii in range(len(high_symmetry_kpoint_labels)):
        high_symmetry_kpoint_x_coor.append(sum(np.array(kpoint_num_in_line[:ii]) * kline_density))

    mysize = 10
    fig, ax = plt.subplots(1, 1, tight_layout=True, figsize=(8, 6))
    set_fig(fig, ax, mysize=mysize)
    
    # Plot band 1
    ax.plot(x_coor_array, band_data1, color='blue', linewidth=1.5, linestyle='-', alpha=0.7)
    # Plot band 2
    ax.plot(x_coor_array, band_data2, color='red', linewidth=1.5, linestyle='--', alpha=0.7)
    
    # Custom legends
    ax.plot([], [], color='blue', linewidth=1.5, linestyle='-', label='Band 1')
    ax.plot([], [], color='red', linewidth=1.5, linestyle='--', label='Band 2')
    ax.legend(loc="upper right")
    
    ax.set_title('Band Structure Comparison', fontsize=mysize)
    ax.set_ylabel('E - E$_F$ (eV)', fontsize=mysize)
    ax.set_xlim(0, x_coor_array[-1])
    ax.set_ylim(emin, emax)
    plt.xticks(high_symmetry_kpoint_x_coor, [l.strip() for l in high_symmetry_kpoint_labels])
    
    for i in high_symmetry_kpoint_x_coor:
        plt.axvline(i, color="grey", alpha=0.5, lw=1, linestyle='--')
        ax.axhline(0.0, color="black", alpha=1, lw=1, linestyle='--')

    plt.savefig(out_file)
    plt.close('all')
    print(f"Comparison plot saved to {out_file}")

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
