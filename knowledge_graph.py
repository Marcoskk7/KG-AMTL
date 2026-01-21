import numpy as np
from scipy.stats import kurtosis, skew
from vmdpy import VMD
from sklearn.preprocessing import MinMaxScaler
import os
from scipy.fft import fft
import numpy as np

# 以仓库现有数据组织结构为准
from preprocess_cwru import load_CWRU_dataset, dataname_dict
from utils import extract_dict_data

# 特征名称定义
TIME_FEATURE_NAMES = [
    "Mean",             # p1
    "Std",              # p2
    "Sq-Mean-of-Sqrt",  # p3 (方根幅值)
    "RMS",              # p4
    "Peak",             # p5
    "Skewness",         # p6
    "Kurtosis",         # p7
    "Crest-Factor",     # p8: p5/p4
    "Clearance-Factor", # p9: p5/p3
    "Waveform-Factor",  # p10: p4/mean_abs
    "Impulse-Factor",   # p11: p5/mean_abs
]

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

VMD_FEATURE_NAMES = [
    "VMD-Energy-1",
    "VMD-Energy-2",
    "VMD-Energy-3",
    "VMD-Energy-4",
    "VMD-SVD-1",
    "VMD-SVD-2",
    "VMD-SVD-3",
    "VMD-SVD-4",
]
# 合并所有特征名称（共31个）
FULL_FEATURE_NAMES = TIME_FEATURE_NAMES + FREQ_FEATURE_NAMES + VMD_FEATURE_NAMES

def time_domain_features(signal):
    """提取时域特征（11个特征，严格按论文TABLE I公式计算）"""
    N = len(signal)
    mean_val = np.mean(signal)  # p1
    std_val = np.std(signal, ddof=1)  # p2

    # 计算中间量
    abs_signal = np.abs(signal)
    sqrt_abs = np.sqrt(abs_signal)
    mean_abs = np.mean(abs_signal)  # 用于多个特征计算
    mean_sqrt_abs = np.mean(sqrt_abs)  # 方根幅值

    # 严格按照论文公式计算
    p1 = mean_val
    p2 = std_val
    p3 = np.square(np.mean(sqrt_abs))  # 平方幅值的均值
    p4 = np.sqrt(np.mean(signal**2))  # RMS
    p5 = np.max(abs_signal)  # 峰值
    p6 = skew(signal)  # 偏度
    p7 = kurtosis(signal)  # 峰度
    p8 = p5 / p4  # 峰值因子
    p9 = p5 / np.square(mean_sqrt_abs)  # 裕度因子
    p10 = p4 / mean_abs  # 波形因子
    p11 = p5 / mean_abs  # 脉冲因子

    return [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11]

def frequency_domain_features(signal, fs=12000):
    """提取频域特征（12个特征，严格按论文TABLE II公式计算）"""
    n = len(signal)
    fft_vals = fft(signal)
    amplitude_spectrum = np.abs(fft_vals[: n // 2]) / n  # 幅度谱
    freqs = np.fft.fftfreq(n, 1 / fs)[: n // 2]
    L = len(amplitude_spectrum)
    s = amplitude_spectrum  # 使用s表示频谱，与论文一致
    p12 = np.sum(s) / L  # 频谱均值
    p13 = np.sum((s - p12) ** 2) / (L - 1)  # 频谱方差（N-1归一化）
    p14 = np.sum((s - p12) ** 3) / (L * (np.sqrt(p13)) ** 3)  # 频谱偏度
    p15 = np.sum((s - p12) ** 4) / (L * p13**2)  # 频谱峰度
    p16 = np.sum(freqs * s) / np.sum(s)  # 频率重心
    p17 = np.sqrt(np.sum(freqs**2 * s) / np.sum(s))  # 均方频率
    p18 = np.sqrt(np.sum((freqs - p16) ** 2 * s) / L)  # 频率标准差
    p19 = np.sqrt(np.sum(freqs**4 * s) / np.sum(freqs**2 * s))  # 频率均方根
    numerator = np.sum(freqs**2 * s)
    denominator = np.sum(s) * np.sum(freqs**4 * s)
    p20 = np.sqrt(numerator / denominator)  # 频率形状因子
    p21 = p18 / p16  # 频率偏度因子
    p22 = np.sum((freqs - p16) ** 3 * s) / (L * p18**3)  # 频率偏度因子2
    p23 = np.sum((freqs - p16) ** 4 * s) / (L * p18**4)  # 频率峰度2

    return [p12, p13, p14, p15, p16, p17, p18, p19, p20, p21, p22, p23]

def vmd_features(signal, alpha=1000, tau=0, K=4, DC=0, init=1, tol=1e-7):
    """提取VMD时频域特征（4个能量特征 + 4个奇异值特征）"""
    u, _, _ = VMD(signal, alpha, tau, K, DC, init, tol)
    # 确保分解得到4个模态
    if u.shape[0] != 4:
        raise ValueError(f"VMD分解应得到4个模态，但实际得到{u.shape[0]}个")
    # 1. 能量特征（4个
    energy_features = [np.sum(np.square(mode)) for mode in u]
    # 2. 奇异值特征（4个）
    mode_matrix = np.array(u)  # 4×N的模态矩阵
    U, S, Vh = np.linalg.svd(mode_matrix, full_matrices=False)
    svd_features = S.tolist()
    # 如果奇异值少于4个，用0填充
    while len(svd_features) < 4:
        svd_features.append(0.0)
    return energy_features + svd_features


def extract_31d_features(
    signal,
    fs=12000,
    vmd_alpha=1000,
    vmd_tau=0,
    vmd_K=4,
    vmd_DC=0,
    vmd_init=1,
    vmd_tol=1e-7,
):
    """
    提取单条时域信号的 31 维特征（时域11 + 频域12 + VMD 8），并按顺序拼接。

    Args:
        signal: (L,) 或 (1, L) 或 (L, 1) 的一维信号
        fs: 采样频率
        vmd_*: VMD 参数
    Returns:
        feat: shape = (31,) 的 np.ndarray
    """
    x = np.asarray(signal, dtype=float)
    x = np.squeeze(x)
    if x.ndim != 1:
        raise ValueError(f"signal 期望为一维数组，但得到形状 {x.shape}")

    f_time = time_domain_features(x)
    f_freq = frequency_domain_features(x, fs=fs)
    f_vmd = vmd_features(x, alpha=vmd_alpha, tau=vmd_tau, K=vmd_K, DC=vmd_DC, init=vmd_init, tol=vmd_tol)
    feat = np.asarray(f_time + f_freq + f_vmd, dtype=float)

    if feat.shape[0] != len(FULL_FEATURE_NAMES):
        raise ValueError(f"特征维度不匹配：得到 {feat.shape[0]} 维，期望 {len(FULL_FEATURE_NAMES)} 维")
    return feat


def signals_to_features(signals, fs=12000, vmd_alpha=1000, vmd_tau=0, vmd_K=4, vmd_DC=0, vmd_init=1, vmd_tol=1e-7):
    """
    将时域信号样本转换为 31 维特征矩阵。

    Args:
        signals: (N, L) 或 (N, 1, L) 的样本矩阵
        fs: 采样频率
    Returns:
        X: (N, 31) 特征矩阵
    """
    sig = np.asarray(signals)
    if sig.ndim == 3 and sig.shape[1] == 1:
        sig = sig[:, 0, :]
    if sig.ndim != 2:
        raise ValueError(f"signals 期望形状为 (N,L) 或 (N,1,L)，但得到 {sig.shape}")

    feats = []
    for i in range(sig.shape[0]):
        feats.append(
            extract_31d_features(
                sig[i],
                fs=fs,
                vmd_alpha=vmd_alpha,
                vmd_tau=vmd_tau,
                vmd_K=vmd_K,
                vmd_DC=vmd_DC,
                vmd_init=vmd_init,
                vmd_tol=vmd_tol,
            )
        )
    X = np.asarray(feats, dtype=float)

    if X.shape[1] != len(FULL_FEATURE_NAMES):
        raise ValueError(f"特征维度不匹配：得到 {X.shape[1]} 维，期望 {len(FULL_FEATURE_NAMES)} 维")
    return X


def compute_feature_fault_weights(X, y, num_classes, membership_soft=None, eps=1e-12):
    """
    计算 feature-fault 关联权重矩阵 W，对应知识图谱中的 (feature -> fault) 边权重。

    - v_{ik} = sum_j u_{ij} x_{jk} / sum_j u_{ij}
    - sigma_{ik} = sum_j u_{ij} (x_{jk}-v_{ik})^2
    - w_{ik} ∝ (sigma_{ik})^{-1/2}，并在 k 维归一化

    Args:
        X: (N, D)
        y: (N,)
        num_classes: 类别数（你当前需要 10 类）
        membership_soft: (N, C) 可选软隶属度；不提供则使用 one-hot
    Returns:
        v: (C, D)
        sigma: (C, D)
        W: (C, D)
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int).reshape(-1)
    N, D = X.shape
    C = int(num_classes)

    if membership_soft is not None:
        U = np.asarray(membership_soft, dtype=float)
        if U.shape != (N, C):
            raise ValueError(f"membership_soft 期望形状 {(N, C)}，但得到 {U.shape}")
        U = np.clip(U, 0.0, 1.0)
    else:
        U = np.zeros((N, C), dtype=float)
        U[np.arange(N), y] = 1.0

    denom = U.sum(axis=0).reshape(C, 1)
    denom = np.maximum(denom, eps)
    v = (U.T @ X) / denom  # (C, D)

    diff2 = (X[None, :, :] - v[:, None, :]) ** 2  # (C, N, D)
    sigma = (U.T[:, :, None] * diff2).sum(axis=1)  # (C, D)
    sigma = np.maximum(sigma, eps)

    inv_sqrt = 1.0 / np.sqrt(sigma)
    W = inv_sqrt / np.maximum(inv_sqrt.sum(axis=1, keepdims=True), eps)
    return v, sigma, W


def build_fault_transition_matrix(num_classes, group_ids=None, y=None, smoothing=1e-3):
    """
    注：当前构建的 P 矩阵，对角线几乎为1，让模型认为只能自转移合理，跨故障转移为0。
    构建故障演化(转移)矩阵 P，输出形状 (C, C)。

    说明：
    - CWRU 默认缺少天然的时间退化序列信息，因此在未提供 group_ids 时，
      这里使用“自环为主 + 拉普拉斯平滑”的保守先验。
    - 若你后续有按设备/运行时序组织的 group_ids（同组内按索引顺序），可传入统计转移。
    """
    C = int(num_classes)
    P = np.zeros((C, C), dtype=float)

    if group_ids is not None and y is not None:
        from collections import defaultdict

        y = np.asarray(y, dtype=int).reshape(-1)
        groups = defaultdict(list)
        for idx, g in enumerate(group_ids):
            groups[g].append(idx)
        for _, idxs in groups.items():
            idxs = sorted(idxs)
            for a, b in zip(idxs[:-1], idxs[1:]):
                P[y[a], y[b]] += 1.0
        P = P + float(smoothing)
        P = P / P.sum(axis=1, keepdims=True)
    else:
        P[:] = float(smoothing)
        np.fill_diagonal(P, 1.0)
        P = P / P.sum(axis=1, keepdims=True)
    return P


def build_kg_cwru(domain, dir_path, time_steps=1024, overlap_ratio=0.5, normalization=False, random_seed=42,
                 fs=12000, scale_features=True, smoothing=1e-3, save_dir="./data/kg"):
    """
    按当前仓库结构直接构建 CWRU 知识图谱所需的矩阵输出：
      - W: (10, 31) feature->fault 权重矩阵
      - P: (10, 10) fault->fault 转移矩阵（保守先验/可选统计）

    数据加载以 preprocess_cwru.load_CWRU_dataset 为准。
    """
    dataset = load_CWRU_dataset(
        domain=domain,
        dir_path=dir_path,
        time_steps=time_steps,
        overlap_ratio=overlap_ratio,
        normalization=normalization,
        random_seed=random_seed,
        raw=True,     # 让样本统一为 (1, L)，便于 extract_dict_data 组装
        fft=False,    # 这里构图基于时域信号的 31 维物理特征，不做 FFT 输入
    )

    signals, y = extract_dict_data(dataset)  # signals: (N, L), y: (N,)
    X = signals_to_features(signals, fs=fs)

    if scale_features:
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)

    num_classes = 10
    v, sigma, W = compute_feature_fault_weights(X, y, num_classes=num_classes)
    P = build_fault_transition_matrix(num_classes=num_classes, smoothing=smoothing)

    # 类名参考 preprocess_cwru 的 fault 编号划分
    class_names = [str(fid) for fid in dataname_dict[int(domain)]]

    os.makedirs(save_dir, exist_ok=True)
    np.savez(
        os.path.join(save_dir, f"kg_domain{int(domain)}_W_P.npz"),
        W=W,
        P=P,
        v=v,
        sigma=sigma,
        feature_names=np.array(FULL_FEATURE_NAMES),
        class_names=np.array(class_names),
        domain=int(domain),
        time_steps=int(time_steps),
        overlap_ratio=float(overlap_ratio),
        normalization=bool(normalization),
        fs=int(fs),
        smoothing=float(smoothing),
        scale_features=bool(scale_features),
        random_seed=int(random_seed),
    )
    return W, P


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build CWRU KG matrices (W,P) using current repo loader + 31D features.")
    parser.add_argument("--domain", type=int, default=0, help="CWRU working condition domain: 0/1/2/3")
    parser.add_argument("--data_dir", type=str, default="./data", help="Data root directory (contains CWRU_12k/...)")
    parser.add_argument("--time_steps", type=int, default=1024)
    parser.add_argument("--overlap_ratio", type=float, default=0.5)
    parser.add_argument("--normalization", action="store_true")
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--fs", type=int, default=12000)
    parser.add_argument("--no_scale", action="store_true", help="Disable MinMax scaling for 31D features.")
    parser.add_argument("--smoothing", type=float, default=1e-3)
    parser.add_argument("--save_dir", type=str, default="./data/kg")

    args = parser.parse_args()

    W, P = build_kg_cwru(
        domain=args.domain,
        dir_path=args.data_dir,
        time_steps=args.time_steps,
        overlap_ratio=args.overlap_ratio,
        normalization=args.normalization,
        random_seed=args.random_seed,
        fs=args.fs,
        scale_features=not args.no_scale,
        smoothing=args.smoothing,
        save_dir=args.save_dir,
    )

    print("KG matrices saved. Shapes:", "W", W.shape, "P", P.shape)