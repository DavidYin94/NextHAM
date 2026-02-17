import re
import matplotlib.pyplot as plt

# 定义从日志文件中提取数据的函数
def extract_data(filename):
    epochs = []
    values = []

    # 正则表达式提取 epoch 和 val_ham_MAE
    pattern = r'Epoch: \[(\d+)\].*val_ham_MAE: ([\d\.]+)'

    with open(filename, 'r') as file:
        for line in file:
            match = re.search(pattern, line)
            if match:
                epoch = int(match.group(1))
                value = float(match.group(2))
                epochs.append(epoch)
                values.append(value)

    return epochs, values

# 从四个日志文件中提取数据
epochs_MG, values_MG = extract_data('alpha_elementwise_bs1.log')
epochs_MG_bs1, values_MG_bs1 = extract_data('alpha_elementwise_bs3.log')
epochs_MG_wo, values_MG_wo = extract_data('sacada_alpha_morepara_1e-3.log')
epochs_MG_wo_bs1, values_MG_wo_bs1 = extract_data('sa_wo_bs1_lrfix1e-3_combined_lesspara.log')

# 创建图表
plt.figure(figsize=(10, 6))

# 绘制四组数据，并设置线宽为0.8
plt.plot(epochs_MG, values_MG, label='alpha_elementwise_bs1', linewidth=0.8)
plt.plot(epochs_MG_bs1, values_MG_bs1, label='alpha_elementwise_bs3', linewidth=0.8)
plt.plot(epochs_MG_wo, values_MG_wo, label='sacada_alpha_morepara_1e-3', linewidth=0.8)
plt.plot(epochs_MG_wo_bs1, values_MG_wo_bs1, label='sa_wo_bs1_lrfix1e-3_combined_lesspara.log', linewidth=0.8)

# 设置对数坐标轴
plt.yscale('log')

# 添加图表标题和标签
plt.title('val_ham_MAE Over Epochs (Log Scale)')
plt.xlabel('Epoch')
plt.ylabel('val_ham_MAE')

# 添加图例
plt.legend()

# 添加网格
plt.grid(True, which="both", ls="--")

# 保存图表为图片文件
plt.savefig('val_ham_MAE_plot_log_scale_thin_lines.png', dpi=300)

# 如果不需要显示图表，可以省略 plt.show()
# plt.show()
