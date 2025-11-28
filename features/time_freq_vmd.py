import numpy as np
from vmdpy import VMD


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


def extract_vmd_features(
    signal: np.ndarray,
    alpha: float = 1000,
    tau: float = 0,
    K: int = 4,
    DC: int = 0,
    init: int = 1,
    tol: float = 1e-7,
) -> list[float]:
    """
    提取基于 VMD 的时频特征：4 个模态能量 + 4 个奇异值特征。

    实现与 reference/KG_2_Final.py 中的 extract_vmd_features 保持一致。
    """
    if signal.ndim != 1:
        raise ValueError("VMD 特征提取仅支持 1D 信号")

    u, _, _ = VMD(signal, alpha, tau, K, DC, init, tol)

    if u.shape[0] != 4:
        raise ValueError(f"VMD 分解应得到 4 个模态，但实际得到 {u.shape[0]} 个")

    # 1. 能量特征
    energy_features = [np.sum(np.square(mode)) for mode in u]

    # 2. 奇异值特征
    mode_matrix = np.array(u)
    _, S, _ = np.linalg.svd(mode_matrix, full_matrices=False)
    svd_features = S.tolist()

    while len(svd_features) < 4:
        svd_features.append(0.0)

    return energy_features + svd_features[:4]


__all__ = ["VMD_FEATURE_NAMES", "extract_vmd_features"]


