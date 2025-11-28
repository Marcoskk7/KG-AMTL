import numpy as np
from scipy.fft import fft


FREQ_FEATURE_NAMES = [
    "Freq-Mean",
    "Freq-Var",
    "Freq-Skewness",
    "Freq-Kurtosis",
    "Freq-Center",
    "Freq-RMS",
    "Freq-Std",
    "Freq-4th-RMS",
    "Freq-Shape-Factor",
    "Freq-Skew-Factor",
    "Freq-Skewness-2",
    "Freq-Kurtosis-2",
]


def extract_frequency_domain_features(signal: np.ndarray, fs: int = 12000) -> list[float]:
    """
    提取频域特征（12 个），公式与 reference/KG_2_Final.py 完全一致。

    Parameters
    ----------
    signal : np.ndarray
        1D 振动信号。
    fs : int, default 12000
        采样频率。

    Returns
    -------
    list[float]
        长度为 12 的特征列表，对应 FREQ_FEATURE_NAMES 的顺序。
    """
    if signal.ndim != 1:
        raise ValueError("频域特征提取仅支持 1D 信号")

    n = len(signal)
    fft_vals = fft(signal)
    amplitude_spectrum = np.abs(fft_vals[: n // 2]) / n
    freqs = np.fft.fftfreq(n, 1 / fs)[: n // 2]

    L = len(amplitude_spectrum)
    if L == 0:
        raise ValueError("频谱长度为 0，无法计算频域特征")

    s = amplitude_spectrum

    p12 = np.sum(s) / L  # 频谱均值
    p13 = np.sum((s - p12) ** 2) / (L - 1)  # 频谱方差

    # 防止除 0
    if p13 <= 0:
        p13 = 1e-12

    p14 = np.sum((s - p12) ** 3) / (L * (np.sqrt(p13)) ** 3)  # 频谱偏度
    p15 = np.sum((s - p12) ** 4) / (L * p13**2)  # 频谱峰度
    p16 = np.sum(freqs * s) / np.sum(s)  # 频率重心
    p17 = np.sqrt(np.sum(freqs**2 * s) / np.sum(s))  # 均方频率
    p18 = np.sqrt(np.sum((freqs - p16) ** 2 * s) / L)  # 频率标准差
    p19 = np.sqrt(np.sum(freqs**4 * s) / np.sum(freqs**2 * s))  # 频率均方根

    numerator = np.sum(freqs**2 * s)
    denominator = np.sum(s) * np.sum(freqs**4 * s)
    if denominator <= 0:
        denominator = 1e-12
    p20 = np.sqrt(numerator / denominator)  # 频率形状因子

    if p16 == 0:
        p16 = 1e-12
    if p18 == 0:
        p18 = 1e-12

    p21 = p18 / p16  # 频率偏度因子
    p22 = np.sum((freqs - p16) ** 3 * s) / (L * p18**3)  # 频率偏度因子2
    p23 = np.sum((freqs - p16) ** 4 * s) / (L * p18**4)  # 频率峰度2

    return [p12, p13, p14, p15, p16, p17, p18, p19, p20, p21, p22, p23]


__all__ = ["FREQ_FEATURE_NAMES", "extract_frequency_domain_features"]


