import numpy as np

TIME_FEATURE_NAMES = [
    "Mean",
    "Std",
    "Sq-Mean-of-Sqrt",
    "RMS",
    "Peak",
    "Skewness",
    "Kurtosis",
    "Waveform-Factor",
    "Peak-Factor",
    "Impulse-Factor",
    "Crest-Factor",
]


def extract_time_domain_features(signal: np.ndarray) -> list[float]:
    """
    提取时域特征（11 个），严格按论文 TABLE I 公式计算：

    p1 = (1/N) * sum x(n)
    p2 = sqrt( sum (x(n)-p1)^2 / (N-1) )
    p3 = ( (1/N) * sum sqrt(|x(n)|) )^2
    p4 = sqrt( sum x(n)^2 / N )
    p5 = max |x(n)|
    p6 = sum (x(n)-p1)^3 / ( (N-1) * p2^3 )
    p7 = sum (x(n)-p1)^4 / ( (N-1) * p2^4 )
    p8 = p5 / p4
    p9 = p5 / p3
    p10 = p4 / ( (1/N) * sum |x(n)| )
    p11 = p5 / ( (1/N) * sum |x(n)| )
    """
    if signal.ndim != 1:
        raise ValueError("时域特征提取仅支持 1D 信号")

    N = len(signal)
    if N < 2:
        raise ValueError("信号长度至少为 2，才能计算方差与高阶矩")

    abs_signal = np.abs(signal)
    sqrt_abs = np.sqrt(abs_signal)

    mean_val = np.mean(signal)  # p1
    # 与 TABLE I 一致，使用 N-1 归一化
    std_val = np.sqrt(np.sum((signal - mean_val) ** 2) / (N - 1))  # p2

    mean_abs = np.mean(abs_signal)
    mean_sqrt_abs = np.mean(sqrt_abs)

    p1 = mean_val
    p2 = std_val
    p3 = mean_sqrt_abs**2  # 平方幅值的均值
    p4 = np.sqrt(np.mean(signal**2))  # RMS
    p5 = np.max(abs_signal)  # 峰值

    if p2 == 0:
        # 全零或常量信号时，偏度/峰度按 0 处理，避免除零
        p6 = 0.0
        p7 = 0.0
    else:
        centered = signal - mean_val
        p6 = np.sum(centered**3) / ((N - 1) * p2**3)  # 偏度
        p7 = np.sum(centered**4) / ((N - 1) * p2**4)  # 峰度

    p8 = p5 / p4 if p4 != 0 else 0.0  # 峰值因子
    p9 = p5 / p3 if p3 != 0 else 0.0  # 裕度因子
    p10 = p4 / mean_abs if mean_abs != 0 else 0.0  # 波形因子
    p11 = p5 / mean_abs if mean_abs != 0 else 0.0  # 脉冲因子

    return [
        float(p1),
        float(p2),
        float(p3),
        float(p4),
        float(p5),
        float(p6),
        float(p7),
        float(p8),
        float(p9),
        float(p10),
        float(p11),
    ]


__all__ = ["TIME_FEATURE_NAMES", "extract_time_domain_features"]
