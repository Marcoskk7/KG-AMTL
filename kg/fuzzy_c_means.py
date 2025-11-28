from typing import Tuple

import numpy as np


def fuzzy_c_means(
    X: np.ndarray,
    n_clusters: int,
    m: float = 2.0,
    max_iter: int = 100,
    tol: float = 1e-6,
    random_state: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    模糊 C 均值聚类（Fuzzy C-Means）。

    实现与 reference/KG_2_Final.py 中的 fuzzy_c_means 保持一致，
    但增加了 random_state 以便复现。

    Parameters
    ----------
    X : np.ndarray
        形状为 (N, D) 的特征矩阵。
    n_clusters : int
        聚类个数 C。
    m : float, default 2.0
        模糊系数 m > 1。
    max_iter : int, default 100
        最大迭代次数。
    tol : float, default 1e-6
        收敛阈值。
    random_state : int | None
        随机种子。

    Returns
    -------
    u : np.ndarray
        形状为 (C, N) 的隶属度矩阵。
    v : np.ndarray
        形状为 (C, D) 的聚类中心矩阵。
    """
    if m <= 1:
        raise ValueError("模糊系数 m 必须大于 1")

    n_samples, _ = X.shape

    rng = np.random.default_rng(random_state)

    # 初始化隶属度矩阵
    u = rng.random((n_clusters, n_samples))
    u /= np.sum(u, axis=0, keepdims=True)

    for _ in range(max_iter):
        u_prev = u.copy()

        # 计算聚类中心
        um = u**m
        v = um @ X / np.sum(um, axis=1, keepdims=True)

        # 更新隶属度矩阵
        for i in range(n_clusters):
            for j in range(n_samples):
                dist = np.linalg.norm(X[j] - v[i])
                if dist < 1e-10:
                    u[i, j] = 1.0
                    continue
                ratio_sum = 0.0
                for k in range(n_clusters):
                    dist_k = np.linalg.norm(X[j] - v[k])
                    if dist_k < 1e-10:
                        dist_k = 1e-10
                    ratio_sum += (dist / dist_k) ** (2.0 / (m - 1.0))
                u[i, j] = 1.0 / ratio_sum

        if np.linalg.norm(u - u_prev) < tol:
            break

    return u, v


__all__ = ["fuzzy_c_means"]


