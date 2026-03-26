import os
import sys
import torch
import argparse


from output_hs_data import transform_abacus_hamiltion_data
from add_element import add_hs_matrix
from plot_band import plot_band
from plot_compare_bands import plot_bands  # 现成的双能带画图函数

def process_single_pipeline(pth_file, stru_file, weak_file, overlap_file, mu, work_dir, nspin):
    """流水线：调用现成模块完成 -> 还原基组 -> 补全弱标签 -> 求能带"""
    os.makedirs(work_dir, exist_ok=True)
    
    # 步骤 A：逆转 E(3) 格式，还原为 abacus 矩阵
    transform_data = transform_abacus_hamiltion_data("inference", nspin, stru_file, pth_file, work_dir)
    transform_data.read_stru()
    transform_data.get_hs_data() # 生成 predict_hr_cut
    
    # 步骤 B：补充截断半径外矩阵元
    predict_hr_cut = os.path.join(work_dir, 'hr_cut')
    add_elem = add_hs_matrix(nspin, stru_file, predict_hr_cut, weak_file, work_dir)
    add_elem.read_stru()
    add_elem.add_matrxi_element() # 生成 predict_hr_tot
    
    # 步骤 C：使用 pyatb 计算能带，并保存为 band1.txt
    predict_hr_tot = os.path.join(work_dir, 'hr_tot')
    band_calculator = plot_band(nspin, stru_file, predict_hr_tot, overlap_file, -10, 10, mu, work_dir)
    band_calculator.cal_band() # 内部会自动生成 band1.txt
    
    return os.path.join(work_dir, 'band1.txt')

def main(args):
    # =======================================================
    # 模块零：智能推导路径 & 读取费米能级
    # =======================================================
    # 自动拼接出 abacus 的基础文件路径
    stru_file = os.path.join(args.case_dir, "abacus_scf/STRU")
    weak_file = os.path.join(args.case_dir, "abacus_geths/OUT.ABACUS/hrs1_nao.csr")
    overlap_file = os.path.join(args.case_dir, "abacus_geths/OUT.ABACUS/srs1_nao.csr")
    
    print(f">>> [0/3] 正在从 {args.efermi_pth} 读取费米能级...")
    # 加载费米能级文件
    mu = float(torch.load(args.efermi_pth, map_location='cpu'))
    print(f">>> 成功获取费米能级: {mu} eV")

    os.makedirs(args.out_dir, exist_ok=True)
    pred_dir = os.path.join(args.out_dir, "pred_band")
    gt_dir = os.path.join(args.out_dir, "gt_band")
    
    # =======================================================
    # 模块一：提取并计算网络预测能带 (Pred)
    # =======================================================
    print("\n>>> [1/3] 调用现成代码处理 Pred 哈密顿量...")
    pred_band_txt = process_single_pipeline(
        args.pth, stru_file, weak_file, overlap_file, mu, pred_dir, args.nspin
    )
    
    # =======================================================
    # 模块二：提取并计算真实物理能带 (Ground Truth)
    # =======================================================
    print("\n>>> [2/3] 调用现成代码处理 Ground Truth 哈密顿量...")
    # 把 H_gt (index 0) 覆盖到 index 1 上，存个临时文件以复用代码
    torch_data = torch.load(args.pth, weights_only=True, map_location='cpu')
    fake_tuple = list(torch_data)
    fake_tuple[1] = fake_tuple[0]  
    temp_gt_pth = os.path.join(args.out_dir, "temp_gt.pth")
    torch.save(tuple(fake_tuple), temp_gt_pth)
    
    gt_band_txt = process_single_pipeline(
        temp_gt_pth, stru_file, weak_file, overlap_file, mu, gt_dir, args.nspin
    )
    os.remove(temp_gt_pth) # 用完清理掉临时文件
    
    # =======================================================
    # 模块三：调用现成的 plot_compare_bands.py 画对比图
    # =======================================================
    print("\n>>> [3/3] 调用 plot_compare_bands 生成双线对比图...")
    out_pdf = os.path.join(args.out_dir, "comparison_bands.pdf")
    
    plot_bands(
        band1_file=gt_band_txt, 
        band2_file=pred_band_txt, 
        stru_file=stru_file, 
        out_file=out_pdf, 
        mu=mu, 
        emin=args.emin, 
        emax=args.emax
    )
    print(f"\n>>> 流水线执行完毕，对比能带图已存至: {out_pdf}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assemble existing scripts to plot compare bands")
    parser.add_argument("--pth", type=str, required=True, help="NextHAM 推理出的 .pth 路径")
    parser.add_argument("--case-dir", type=str, required=True, help="ABACUS 计算所在的 case 文件夹路径")
    parser.add_argument("--efermi-pth", type=str, required=True, help="保存费米能级的 efermi.pth 路径")
    parser.add_argument("--out-dir", type=str, default="./Output_Bands", help="输出目录")
    parser.add_argument("--nspin", type=int, default=4)
    parser.add_argument("--emin", type=float, default=-10.0)
    parser.add_argument("--emax", type=float, default=10.0)
    args = parser.parse_args()
    main(args)