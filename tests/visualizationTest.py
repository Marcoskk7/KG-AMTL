import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

from data.cwru_loader import _default_file_mapping, _find_channel_key

# ========== 基本配置（可按需修改，结构与原始示例保持相似） ==========
root_dir = os.path.join("D:\KG-AMTL","data", "CRWU")  # CWRU 原始 .mat 文件目录
channel = "DE"  # 使用 DE 通道，可改为 "FE"
sampling_rate = 12000  # 采样率 (Hz)
num_samples = 2048  # 每条信号截取前 2048 个点
max_signals = 10  # 最多可视化 10 个不同工况（5×2 子图）
output_dir = "CWRU信号示意图"  # 输出图片目录

# ========== 创建输出目录 ==========
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# ========== Matplotlib 全局样式设置（与原始代码一致风格） ==========
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"

# ========== 使用项目内的默认文件映射，替代手动文件名列表 ==========
file_mapping = _default_file_mapping()
items = sorted(file_mapping.items())[:max_signals]
if not items:
    raise RuntimeError("默认 CWRU 文件映射为空，请检查 data/CRWU 目录结构。")

# ========== 绘制时域图（结构仿照你最初提供的 5×2 子图代码） ==========
plt.figure(figsize=(12, 10))  # 设置画布大小
n_rows, n_cols = 5, 2

for i, (label_name, rel_path) in enumerate(items):
    file_path = os.path.join(root_dir, rel_path)
    if not os.path.exists(file_path):
        print(f"[跳过] 文件不存在: {file_path}")
        continue

    try:
        mat_data = sio.loadmat(file_path)
    except Exception as e:
        print(f"[跳过] 读取失败 {file_path}: {e}")
        continue

    # 使用已有逻辑自动找到 DE/FE 通道 key，而不是手写变量名
    key = _find_channel_key(mat_data, channel)
    if key is None:
        print(f"[跳过] 未找到 {channel} 通道信号: {file_path}")
        continue

    # 获取对应的信号数据，并转为一维
    signal_data = np.ravel(mat_data[key])
    if len(signal_data) < num_samples:
        print(f"[跳过] 信号过短 {file_path}, 长度 {len(signal_data)} < {num_samples}")
        continue

    signal_data = signal_data[:num_samples]

    # 创建时间轴（秒）
    time = np.arange(num_samples) / float(sampling_rate)

    # 子图位置（5 行 2 列）
    subplot_idx = i + 1
    if subplot_idx > n_rows * n_cols:
        break

    plt.subplot(n_rows, n_cols, subplot_idx)
    plt.plot(time, signal_data, color="#2632cd")
    plt.title(label_name, fontsize=12)
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("Amplitude", fontsize=12)

# 调整布局并保存多种格式
plt.tight_layout()
base_name = f"CWRU_Time_domain_signals_{channel}"
plt.savefig(os.path.join(output_dir, f"{base_name}.png"), dpi=600)
plt.savefig(os.path.join(output_dir, f"{base_name}.tif"), dpi=300)
plt.savefig(os.path.join(output_dir, f"{base_name}.svg"))
plt.savefig(os.path.join(output_dir, f"{base_name}.pdf"))

# 显示图形
plt.show()
