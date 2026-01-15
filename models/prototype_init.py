from __future__ import annotations

"""
Prototype-based Knowledge-Aware Initialization utilities.

本模块实现 SRP Guidance 中 5.4.1 节的核心思想：

    θ0 = Σ_c γ_c · θ^proto_c

其中 γ_c 由源域知识向量 w_c 与目标任务知识向量 w_target 的距离决定。

为了避免与 `models.meta_transfer` 形成循环依赖，本文件不直接导入
`KGMetaClassifier`，而是只依赖其通用的 forward 接口：

    logits, feats = model(x_signal, x_feat, y)

和标准的 `state_dict()` / `load_state_dict()` 行为。
"""

from collections import OrderedDict
from typing import Dict, Iterable

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def _get_device_from_model(model: nn.Module) -> torch.device:
    """从模型参数推断当前设备。"""

    try:
        first_param = next(model.parameters())
        return first_param.device
    except StopIteration:
        return torch.device("cpu")


def train_class_prototypes(
    base_model: nn.Module,
    signals: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    finetune_epochs: int = 10,
    finetune_lr: float = 1e-3,
    batch_size: int = 256,
) -> Dict[int, OrderedDict]:
    """
    Phase 1: 为每个类别训练原型参数 θ^proto_c。

    流程（与计划说明一致）：
      1) 以 base_model 的当前参数为起点；
      2) 对每个类 c，用该类的数据在少量 epoch 内做微调；
      3) 返回 {c: θ^proto_c} 字典（包含 backbone + classifier 全量参数）。

    参数
    ----
    base_model :
        已构建好的分类模型（例如 KGMetaClassifier），要求 forward(x_signal, x_feat, y)
        返回 (logits, feats)。
    signals : (N, T) np.ndarray
        时域信号片段。
    features : (N, D) np.ndarray
        源域特征矩阵（通常是 31 维物理特征）。
    labels : (N,) np.ndarray[int]
        源域硬标签。
    num_classes : int
        类别数 C。
    finetune_epochs : int
        每一类原型微调的 epoch 数。
    finetune_lr : float
        原型微调阶段的学习率。
    batch_size : int
        微调时的 batch 大小。
    """

    if signals.ndim != 2:
        raise ValueError(f"signals 期望形状 [N, T]，但收到 {signals.shape}")
    if features.ndim != 2:
        raise ValueError(f"features 期望形状 [N, D]，但收到 {features.shape}")
    if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
        raise ValueError(
            f"labels 期望形状 [N] 且与 features 样本数一致，"
            f"但收到 labels.shape={labels.shape}, features.shape={features.shape}"
        )
    if signals.shape[0] != features.shape[0]:
        raise ValueError(
            f"signals 与 features 样本数不一致：signals={signals.shape[0]}, "
            f"features={features.shape[0]}"
        )

    device = _get_device_from_model(base_model)

    S_all = torch.tensor(signals, dtype=torch.float32, device=device)
    X_all = torch.tensor(features, dtype=torch.float32, device=device)
    y_all = torch.tensor(labels, dtype=torch.long, device=device)

    prototypes: Dict[int, OrderedDict] = {}

    for c in range(num_classes):
        mask = y_all == c
        num_samples_c = int(mask.sum().item())

        if num_samples_c == 0:
            # 若某一类在当前数据中没有样本，则直接使用 base_model 的参数作为其原型。
            prototypes[c] = OrderedDict(
                (k, v.detach().clone()) for k, v in base_model.state_dict().items()
            )
            continue

        S_c = S_all[mask]
        X_c = X_all[mask]
        y_c = y_all[mask]

        dataset = TensorDataset(S_c, X_c, y_c)
        loader = DataLoader(dataset, batch_size=min(batch_size, num_samples_c), shuffle=True)
        # 使用 base_model 的深拷贝作为原型初始点，避免依赖具体构造函数签名
        proto_model = copy.deepcopy(base_model).to(device)
        proto_model.train()

        optimizer = torch.optim.Adam(proto_model.parameters(), lr=finetune_lr)

        for _ in range(max(finetune_epochs, 1)):
            for sb, xb, yb in loader:
                logits, _ = proto_model(sb, xb, yb)
                loss = F.cross_entropy(logits, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        prototypes[c] = OrderedDict(
            (k, v.detach().clone()) for k, v in proto_model.state_dict().items()
        )

    return prototypes


def fuse_prototype_params(
    prototypes: Dict[int, OrderedDict],
    W_source: np.ndarray,
    w_target: np.ndarray,
    lambda_temp: float = 1.0,
) -> OrderedDict:
    """
    Phase 2: 根据 w_target 与 W_source 计算 γ_c，融合得到 θ0。

    公式：
        γ_c = softmax(-λ · ||W_source[c] - w_target||²)
        θ0  = Σ_c γ_c · θ^proto_c  （逐参数加权求和）

    参数
    ----
    prototypes :
        {类索引 -> 该类原型参数 state_dict}。
    W_source : (C, D) np.ndarray
        源域知识矩阵 W_real，每行对应一个类的知识向量 w_c。
    w_target : (D,) np.ndarray
        目标任务的知识向量 w_target。
    lambda_temp : float
        softmax 温度 λ，越大则对距离差异越敏感。
    """

    if not prototypes:
        raise ValueError("prototypes 字典为空，无法进行参数融合。")
    if W_source.ndim != 2:
        raise ValueError(f"W_source 期望形状 [C, D]，但收到 {W_source.shape}")
    if w_target.ndim != 1:
        raise ValueError(f"w_target 期望形状 [D]，但收到 {w_target.shape}")
    if W_source.shape[1] != w_target.shape[0]:
        raise ValueError(
            f"W_source 与 w_target 维度不匹配："
            f"W_source.shape={W_source.shape}, w_target.shape={w_target.shape}"
        )

    # 只对存在原型的那些类进行加权融合
    cls_indices: Iterable[int] = sorted(prototypes.keys())
    cls_indices_list = list(cls_indices)
    W_sub = W_source[cls_indices_list]  # (C_sub, D)

    # 欧氏距离的平方 ||w_c - w_target||²
    diffs = W_sub - w_target[None, :]
    dists_sq = np.sum(diffs**2, axis=1)  # (C_sub,)

    logits = -lambda_temp * dists_sq
    logits = logits - logits.max()
    exp_logits = np.exp(logits)
    gammas = exp_logits / (exp_logits.sum() + 1e-12)  # (C_sub,)

    # 逐参数加权求和
    first_proto = next(iter(prototypes.values()))
    fused_state = OrderedDict()

    for name in first_proto.keys():
        fused_param = None
        for gamma, c in zip(gammas, cls_indices_list):
            param_c = prototypes[c][name]
            if fused_param is None:
                fused_param = gamma * param_c
            else:
                fused_param = fused_param + gamma * param_c
        assert fused_param is not None
        fused_state[name] = fused_param

    return fused_state


__all__ = ["train_class_prototypes", "fuse_prototype_params"]


