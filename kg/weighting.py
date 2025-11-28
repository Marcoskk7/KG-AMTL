from typing import Tuple

import numpy as np

from .fuzzy_c_means import fuzzy_c_means


def compute_correlation_matrix(
    features: np.ndarray,
    num_classes: int,
    m: float = 2.0,
    max_iter: int = 100,
    tol: float = 1e-6,
    random_state: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    根据 Steps / Doc5 中的公式计算特征-故障相关矩阵 W。

    实现与 reference/KG_2_Final.py 中 compute_correlation_matrix 保持一致：
    - 先对所有样本做 Fuzzy C-Means 得到隶属度矩阵 u；
    - 再按类 i 计算特征中心 v_ik 与方差 σ_ik；
    - 最后根据公式 (2) 归一化得到 w_ik。

    Parameters
    ----------
    features : np.ndarray
        形状为 (N, D) 的特征矩阵。
    num_classes : int
        故障类别数量 C。
    m, max_iter, tol, random_state :
        FCM 聚类的超参数。

    Returns
    -------
    W : np.ndarray
        形状为 (C, D) 的相关矩阵。
    u : np.ndarray
        形状为 (C, N) 的隶属度矩阵（便于后续分析）。
    """
    N, D = features.shape
    C = num_classes
    if C <= 0:
        raise ValueError("num_classes 必须为正整数")

    W = np.zeros((C, D), dtype=float)

    # Fuzzy C-Means 聚类
    u, _ = fuzzy_c_means(
        features,
        n_clusters=num_classes,
        m=m,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
    )

    for i in range(C):
        u_ij = u[i]  # (N,)
        sum_u = np.sum(u_ij)
        if sum_u < 1e-10:
            continue

        # 特征中心 v_ik
        v_ik = np.zeros(D, dtype=float)
        for k in range(D):
            v_ik[k] = np.sum(u_ij * features[:, k]) / sum_u

        # 方差 σ_ik
        sigma_ik = np.zeros(D, dtype=float)
        for k in range(D):
            sigma_ik[k] = np.sum(u_ij * (features[:, k] - v_ik[k]) ** 2)

        sigma_ik = np.maximum(sigma_ik, 1e-10)

        # 权重 w_ik
        w_i = sigma_ik ** (-0.5)
        w_i /= np.sum(w_i)

        W[i] = w_i

    return W, u


__all__ = ["compute_correlation_matrix"]


