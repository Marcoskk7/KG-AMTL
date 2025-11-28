import numpy as np
import matplotlib.pyplot as plt
from vmdpy import VMD  # 导入VMD库
from PyEMD import EMD  # 导入EMD库

# --- 第一步：再造那碗八宝粥（信号生成） ---
t = np.linspace(0, 1, 1000)
# 信号A：5Hz (低频基波)
signal_low = np.sin(2 * np.pi * 5 * t)
# 信号B：50Hz (高频干扰/故障特征)
signal_high = 0.5 * np.sin(2 * np.pi * 50 * t)
# 混合信号
signal_mixed = signal_low + signal_high

# --- 第二步：EMD 分解 (经验模态分解) ---
# EMD 不需要设置参数，它自己算
emd = EMD()
emd_imfs = emd.emd(signal_mixed)
# 通常EMD会分解出很多层，我们只取前两层主要的来看
emd_imf1 = emd_imfs[0] # 第1层 (通常是高频)
emd_imf2 = emd_imfs[1] # 第2层 (通常是低频)

# --- 第三步：VMD 分解 (变分模态分解) ---
# VMD 需要预先设定一些参数，这正是它严谨的地方
alpha = 2000       # 惩罚因子 (决定带宽，越大越窄)
tau = 0            # 噪声容忍度 (0代表不容忍噪声)
K = 2              # 模态数量 (我们知道有两个成分，所以设为2)
DC = 0             # 是否有直流分量
init = 1           # 初始化方式 (1代表均匀初始化)
tol = 1e-7         # 收敛准则

# 运行VMD
# u是分解出的模态，u_hat是频域结果，omega是中心频率
u, u_hat, omega = VMD(signal_mixed, alpha, tau, K, DC, init, tol)
vmd_imf1 = u[0]    # VMD分解出的第1个分量
vmd_imf2 = u[1]    # VMD分解出的第2个分量

# --- 第四步：画图对比 ---
plt.figure(figsize=(12, 10))

# 1. 原始信号
plt.subplot(3, 1, 1)
plt.plot(t, signal_mixed, 'g', linewidth=1.5)
plt.title("原始混合信号 (5Hz + 50Hz)", fontproperties="SimHei", fontsize=14)
plt.grid(True)

# 2. EMD 分解结果
plt.subplot(3, 2, 3)
plt.plot(t, emd_imf1, 'b')
plt.title("EMD - IMF1 (高频分量)", fontproperties="SimHei")
plt.grid(True)
plt.subplot(3, 2, 4)
plt.plot(t, emd_imf2, 'b')
plt.title("EMD - IMF2 (低频分量)", fontproperties="SimHei")
plt.grid(True)

# 3. VMD 分解结果
plt.subplot(3, 2, 5)
plt.plot(t, vmd_imf1, 'r')
plt.title("VMD - 模态1", fontproperties="SimHei")
plt.grid(True)
plt.subplot(3, 2, 6)
plt.plot(t, vmd_imf2, 'r')
plt.title("VMD - 模态2", fontproperties="SimHei")
plt.grid(True)

plt.tight_layout()
plt.show()