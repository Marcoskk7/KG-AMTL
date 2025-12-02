from __future__ import annotations

"""
Step 6: Dynamic Feature Weighting（动态特征加权层）

对应 `SRP Guidance.pdf` / reference/Steps.md 中 2.3 - Knowledge-Guided Meta-Transfer Learning
里的公式：

    f_w = φ(x; θ) ⊙ σ(W_i)

含义：
    - φ(x; θ) : 由基学习器 / 主干网络提取的特征向量（例如 31 维物理特征，或更高维深度特征）；
    - W_i     : 知识图谱中对应故障类别 i 的特征相关性向量（来自 kg.weighting.compute_correlation_matrix）；
    - σ(·)    : Sigmoid 非线性，将相关性权重规范到 (0, 1) 区间；
    - ⊙       : 按元素相乘，对特征进行逐维缩放。

本文件仅实现「特征加权算子」本身，不绑定具体的特征提取网络或元学习训练流程，
便于在后续 Step 7 / Step 8 中直接复用。
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class DynamicFeatureWeighterConfig:
    """
    Step 6 动态特征加权层的基础配置。

    Parameters
    ----------
    num_classes : int
        故障类别数 C，对应相关矩阵 W 的行数。
    feature_dim : int
        特征维度 D，要求与 φ(x; θ) 输出维度一致。
    apply_sigmoid_to_w : bool, default True
        是否对 W_i 先做 Sigmoid，再与特征相乘；若为 False，则直接使用 W_i。
    """

    num_classes: int
    feature_dim: int
    apply_sigmoid_to_w: bool = True


class DynamicFeatureWeighter(nn.Module):
    """
    Knowledge-Guided Dynamic Feature Weighting Layer.

    实现公式：
        f_w = φ(x; θ) ⊙ σ(W_i)

    用法示例
    --------
    >>> cfg = DynamicFeatureWeighterConfig(num_classes=4, feature_dim=31)
    >>> layer = DynamicFeatureWeighter(cfg)
    >>> layer.register_knowledge_weights(W_tensor)  # W_tensor 形状为 [C, D]
    >>> f = backbone(x)          # [B, D]，例如 31 维物理特征
    >>> y = labels               # [B]，故障类别索引
    >>> f_w = layer(f, y)        # [B, D]，动态加权后的特征

    也可以在前向时显式传入 W（便于做 ablation 或不同知识源对比）：

    >>> f_w = layer(f, y, W_override)
    """

    def __init__(self, config: DynamicFeatureWeighterConfig) -> None:
        super().__init__()
        self.config = config

        self.num_classes = config.num_classes
        self.feature_dim = config.feature_dim
        self.apply_sigmoid_to_w = config.apply_sigmoid_to_w

        # 可选：将 W 注册为 buffer，默认情况下由外部通过 register_knowledge_weights 提供。
        self.register_buffer("_W", None, persistent=False)

    @property
    def has_knowledge(self) -> bool:
        """当前层是否已经加载了知识图谱相关矩阵 W。"""

        return self._W is not None

    def register_knowledge_weights(self, W: torch.Tensor) -> None:
        """
        注册来自知识图谱的相关矩阵 W（不会参与梯度更新）。

        Parameters
        ----------
        W : torch.Tensor
            形状为 [C, D] 的张量，其中 C = num_classes, D = feature_dim。
        """

        if W.dim() != 2:
            raise ValueError(f"W 期望形状为 [C, D]，但收到 {tuple(W.shape)}")
        if W.size(0) != self.num_classes or W.size(1) != self.feature_dim:
            raise ValueError(
                f"W 形状与配置不匹配：期望 ({self.num_classes}, {self.feature_dim})，"
                f"实际为 {tuple(W.shape)}"
            )

        # 作为 buffer 存储，确保在 .to(device) / .eval() 时自动迁移，但不参与梯度更新
        self._W = W.detach()

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        W: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播：对特征做按类别的知识引导加权。

        Parameters
        ----------
        features : torch.Tensor
            [B, D]，来自主干网络 / 手工特征管线的特征向量 φ(x; θ)。
        labels : torch.Tensor
            [B]，每个样本的故障类别索引（0 ~ C-1）。
        W : Optional[torch.Tensor]
            若提供，则使用该相关矩阵 W（形状 [C, D]，优先级高于内部 buffer）；
            若为 None，则使用先前通过 register_knowledge_weights 注册的 _W。

        Returns
        -------
        weighted_features : torch.Tensor
            [B, D]，按类别使用知识权重缩放后的特征向量 f_w。
        """

        if features.dim() != 2:
            raise ValueError(
                f"features 期望形状为 [B, D]，但收到 {tuple(features.shape)}"
            )
        if features.size(1) != self.feature_dim:
            raise ValueError(
                f"features 第二维应为 feature_dim={self.feature_dim}，"
                f"但收到 {features.size(1)}"
            )

        if labels.dim() != 1:
            raise ValueError(f"labels 期望形状为 [B]，但收到 {tuple(labels.shape)}")
        if labels.size(0) != features.size(0):
            raise ValueError(
                f"labels 与 features 批量大小不一致："
                f"features={features.size(0)}, labels={labels.size(0)}"
            )

        if W is None:
            if not self.has_knowledge:
                raise RuntimeError(
                    "未提供 W，且当前 DynamicFeatureWeighter 也尚未通过 "
                    "`register_knowledge_weights` 注册知识矩阵。"
                )
            W = self._W

        if W.dim() != 2:
            raise ValueError(f"W 期望形状为 [C, D]，但收到 {tuple(W.shape)}")
        if W.size(0) != self.num_classes or W.size(1) != self.feature_dim:
            raise ValueError(
                f"W 形状与配置不匹配：期望 ({self.num_classes}, {self.feature_dim})，"
                f"实际为 {tuple(W.shape)}"
            )

        # 根据标签选出每个样本对应的 W_i，形状 [B, D]
        if labels.min() < 0 or labels.max() >= self.num_classes:
            raise ValueError(
                f"labels 中存在越界类别索引，合法范围应为 [0, {self.num_classes - 1}]，"
                f"但 labels.min={int(labels.min())}, labels.max={int(labels.max())}"
            )

        class_weights = W[labels]  # [B, D]
        if self.apply_sigmoid_to_w:
            class_weights = torch.sigmoid(class_weights)

        weighted_features = features * class_weights
        return weighted_features


__all__ = ["DynamicFeatureWeighterConfig", "DynamicFeatureWeighter"]
