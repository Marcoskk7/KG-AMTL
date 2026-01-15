from __future__ import annotations

"""
Knowledge-Aware Initialization (原型加权融合) 工具模块。

本模块实现 `reference/markdown/3.md` 第 3.1 节中与知识空间相关的核心公式，
用于在给定：
    - 源域知识图谱矩阵  W^{(s)}  (每一行是一个“知识原型” w_c^{(s)})
    - 目标任务少量支持集样本提取到的特征重要性向量 w^{(t)}
时，计算：
    - 原型融合权重 γ_c
    - 基于 γ_c 的加权融合知识向量  w_fused

注意：
    - 这里聚焦于「知识向量」层面的原型加权（即 w 空间），不直接约束具体网络参数 θ。
      在实际使用中，可以将 w_fused 映射到：
          - 动态特征加权层的相关矩阵（如替换/融合某一类的 W[i]）；
          - 或用于构造更高层的参数原型（例如 class-wise head 的加权融合）。
    - 若需要对完整模型参数做原型加权，可在上层调用中使用本模块返回的 γ_c，
      对若干 prototype state_dict 进行逐参数加权求和即可。
"""

from typing import Tuple

import numpy as np

from kg.weighting import compute_correlation_matrix


def compute_target_knowledge_vector(
    support_features: np.ndarray,
    random_state: int | None = None,
) -> np.ndarray:
    """
    根据少量支持集样本，计算目标任务的特征重要性向量 w^{(t)}。

    实现与 `kg.weighting.compute_correlation_matrix` 保持一致，只是将
    num_classes 固定为 1（视为单一“簇/故障类”），然后取返回矩阵的第 0 行。

    Parameters
    ----------
    support_features : np.ndarray, shape [N_supp, D]
        支持集样本在物理特征空间的表示。
    random_state : Optional[int]
        Fuzzy C-Means 聚类的随机种子。

    Returns
    -------
    w_t : np.ndarray, shape [D]
        目标任务的知识向量 w^{(t)}。
    """

    if support_features.ndim != 2:
        raise ValueError(
            f"support_features 期望形状为 [N, D]，但收到 {support_features.shape}"
        )
    if support_features.shape[0] < 2:
        # 样本过少时，直接使用简单均值近似（避免 FCM 数值不稳定）
        return support_features.mean(axis=0)

    W_t, _ = compute_correlation_matrix(
        support_features,
        num_classes=1,
        random_state=random_state,
    )
    # 只取唯一一类的相关性向量
    return W_t[0]


def fuse_knowledge_prototypes(
    W_source: np.ndarray,
    w_target: np.ndarray,
    lambda_temp: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    根据源域知识原型 W^{(s)} 与目标知识向量 w^{(t)} 计算：

        γ_c = softmax\left( -λ · d(w_c^{(s)}, w^{(t)}) \right)
        w_fused = \sum_c γ_c · w_c^{(s)}

    其中 d(·,·) 为欧氏距离。

    Parameters
    ----------
    W_source : np.ndarray, shape [C_s, D]
        源域知识矩阵，每一行对应一个源故障类的特征重要性向量 w_c^{(s)}。
    w_target : np.ndarray, shape [D]
        目标任务知识向量 w^{(t)}。
    lambda_temp : float, default 1.0
        温度系数 λ，控制 softmax 分布的尖锐程度。

    Returns
    -------
    w_fused : np.ndarray, shape [D]
        融合后的知识向量。
    gammas : np.ndarray, shape [C_s]
        对应每个源原型的融合权重 γ_c，满足 sum γ_c = 1。
    """

    if W_source.ndim != 2:
        raise ValueError(
            f"W_source 期望形状为 [C_s, D]，但收到 {W_source.shape}"
        )
    if w_target.ndim != 1:
        raise ValueError(
            f"w_target 期望形状为 [D]，但收到 {w_target.shape}"
        )
    if W_source.shape[1] != w_target.shape[0]:
        raise ValueError(
            f"W_source 与 w_target 维度不匹配："
            f"W_source.shape={W_source.shape}, w_target.shape={w_target.shape}"
        )

    # 欧氏距离 d_c = || w_c^{(s)} - w^{(t)} ||_2
    diffs = W_source - w_target[None, :]
    dists = np.linalg.norm(diffs, axis=1)  # [C_s]

    # softmax(-λ d_c)
    logits = -lambda_temp * dists
    # 数值稳定性处理
    logits = logits - logits.max()
    exp_logits = np.exp(logits)
    gammas = exp_logits / (exp_logits.sum() + 1e-12)

    # 原型加权融合
    w_fused = (gammas[:, None] * W_source).sum(axis=0)
    return w_fused, gammas


__all__ = [
    "compute_target_knowledge_vector",
    "fuse_knowledge_prototypes",
]


