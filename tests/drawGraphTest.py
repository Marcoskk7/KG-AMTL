import numpy as np
import matplotlib.pyplot as plt

# 1. 设置时间，假设咱们录了1秒钟的声音
t = np.linspace(0, 1, 1000)

# 2. 制造食材（成分）
# 食材A：慢悠悠的低音（比如大鼓），每秒震动5下 (5Hz)
signal_low = np.sin(2 * np.pi * 5 * t)

# 食材B：急匆匆的高音（比如胡琴），每秒震动50下 (50Hz)
signal_high = 0.5 * np.sin(2 * np.pi * 50 * t)

# 3. 熬成八宝粥（混合信号）
# 机器实际上发出的声音，是上面两个加在一起
signal_mixed = signal_low + signal_high

# --- 开始绘图给太奶看 ---
plt.figure(figsize=(10, 8))

# 第一张图：看食材A
plt.subplot(3, 1, 1)
plt.plot(t, signal_low, color='blue', linewidth=2)
plt.title("【食材A】低频信号（像大鼓，慢，劲儿大）", fontproperties="SimHei", fontsize=14)
plt.grid(True)

# 第二张图：看食材B
plt.subplot(3, 1, 2)
plt.plot(t, signal_high, color='orange', linewidth=2)
plt.title("【食材B】高频信号（像胡琴，快，劲儿小）", fontproperties="SimHei", fontsize=14)
plt.grid(True)

# 第三张图：看八宝粥
plt.subplot(3, 1, 3)
plt.plot(t, signal_mixed, color='green', linewidth=2)
plt.title("【如果不处理】机器实际发出的声音（乱糟糟的波浪）", fontproperties="SimHei", fontsize=14)
plt.xlabel("时间 (秒)", fontproperties="SimHei")
plt.grid(True)

plt.tight_layout()
plt.show()


# 对刚才那个乱糟糟的绿线（signal_mixed）做傅里叶变换
fft_values = np.fft.fft(signal_mixed)
freqs = np.fft.fftfreq(len(signal_mixed), 1/1000)

# 只取正半边的频率来看（因为对称的）
half_len = len(signal_mixed) // 2
freqs_half = freqs[:half_len]
fft_magnitude = np.abs(fft_values)[:half_len] / half_len

# --- 绘图 ---
plt.figure(figsize=(10, 5))
plt.plot(freqs_half, fft_magnitude, color='red', linewidth=2)
plt.title("【照妖镜】频域图：一眼看穿配方表", fontproperties="SimHei", fontsize=16)
plt.xlabel("频率 (赫兹/Hz) - 代表震动的快慢", fontproperties="SimHei", fontsize=12)
plt.ylabel("幅度 - 代表劲儿有多大", fontproperties="SimHei", fontsize=12)
plt.xlim(0, 60) # 只看前60Hz，后面没东西
plt.grid(True)

# 给太奶标注一下重点
plt.annotate('这根柱子在5Hz\n那是刚才的“大鼓”！', xy=(5, 1), xytext=(10, 1.2),
             arrowprops=dict(facecolor='black', shrink=0.05), fontproperties="SimHei", fontsize=12)

plt.annotate('这根柱子在50Hz\n那是刚才的“胡琴”！', xy=(50, 0.5), xytext=(30, 0.7),
             arrowprops=dict(facecolor='black', shrink=0.05), fontproperties="SimHei", fontsize=12)

plt.show()