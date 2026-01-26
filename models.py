import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN1D(nn.Module):
    """
    一维卷积神经网络模型，用于处理FFT预处理后的时序数据
    """
    def __init__(self, output_size, use_kg: bool = False):
        """
        初始化CNN1D模型
        
        Args:
            output_size: 输出类别数（分类任务的类别数量）
            use_kg: 是否启用KG动态门控
        """
        # 调用父类构造函数
        super(CNN1D, self).__init__()
        self.output_size = int(output_size)
        self.use_kg = bool(use_kg)
        # 第一层卷积块：输入通道1，输出通道32
        self.layer1 = nn.Sequential(
            nn.Conv1d(1,32,kernel_size=3,padding=1),  # 一维卷积：1输入通道，32输出通道，卷积核大小3，填充1保持尺寸
            nn.BatchNorm1d(32),  # 一维批量归一化，32个通道
            nn.ReLU(),  # ReLU激活函数
            nn.MaxPool1d(kernel_size=2, padding=0)  # 一维最大池化，池化核大小2，将特征图尺寸减半
            )
        # 第二层卷积块：输入通道32，输出通道64
        self.layer2 = nn.Sequential(
            nn.Conv1d(32,64,kernel_size=3,padding=1),  # 一维卷积：32输入通道，64输出通道
            nn.BatchNorm1d(64),  # 一维批量归一化，64个通道
            nn.ReLU(),  # ReLU激活函数
            nn.MaxPool1d(kernel_size=2, padding=0)  # 一维最大池化
            )
        # 第三层卷积块：输入通道64，输出通道64
        self.layer3 = nn.Sequential(
            nn.Conv1d(64,64,kernel_size=3,padding=1),  # 一维卷积：64输入通道，64输出通道
            nn.BatchNorm1d(64),  # 一维批量归一化，64个通道
            nn.ReLU(),  # ReLU激活函数
            nn.MaxPool1d(kernel_size=2, padding=0)  # 一维最大池化
            )
        # 自适应平均池化：将特征图池化到固定长度64
        self.avgpool = nn.AdaptiveAvgPool1d(64) # 输出形状 (batch, 64, 64)
        if self.use_kg:
            # FFT -> 31维瓶颈 + KG gate
            self.proj = nn.Linear(64*64, 31)
            self.classifier = nn.Linear(31, self.output_size)
            # KG matrices as buffers (moved with .to(device))
            self.register_buffer("W", torch.zeros(self.output_size, 31), persistent=True)  # (10,31)
            self.register_buffer("P", torch.eye(self.output_size), persistent=True)       # (10,10)
            # # KG gate strength (learnable) + schedule scale
            # # 用 sigmoid 映射到 (0,1)，避免 alpha 过大导致不稳定
            # self.kg_alpha = nn.Parameter(torch.tensor(0.0))
            # self.register_buffer("kg_alpha_scale", torch.tensor(1.0), persistent=True)
            # # Gate temperature (optional, to avoid saturation)
            # self.register_buffer("kg_temp", torch.tensor(1.0), persistent=True)
            # fast_adapt 会用这个 flag 决定是否在 support 传入真值标签进行门控
            self.supports_label_gate = True
        else:
            # 全连接层：将特征展平后映射到输出类别数
            self.fc = nn.Linear(64*64, self.output_size)  # 输入维度64*64，输出维度output_size
            self.supports_label_gate = False

    @staticmethod
    def _load_kg_npz(kg_npz_path: str) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        if not os.path.exists(kg_npz_path):
            raise FileNotFoundError(f"KG npz not found: {kg_npz_path}")
        data = np.load(kg_npz_path, allow_pickle=True)
        W = data["W"]  # expected (10,31) in this repo
        P = data["P"]  # expected (10,10)
        feature_names = data["feature_names"] if "feature_names" in data.files else None
        return W, P, feature_names

    @torch.no_grad()
    def set_kg(self, W: np.ndarray, P: np.ndarray, feature_names: Optional[np.ndarray] = None):
        """
        在训练循环中可按 domain 动态切换 KG。

        W 允许两种形状：
          - (10,31) 本仓库 build_kg_cwru 的默认输出
          - (31,10) 若外部按“列为类别”保存，可自动转置为 (10,31)
        """
        if not self.use_kg:
            return
        W = np.asarray(W, dtype=np.float32)
        P = np.asarray(P, dtype=np.float32)
        if W.shape == (31, self.output_size):
            W = W.T
        if W.shape != (self.output_size, 31):
            raise ValueError(f"W shape must be ({self.output_size},31) or (31,{self.output_size}), got {W.shape}")
        if P.shape != (self.output_size, self.output_size):
            raise ValueError(f"P shape must be ({self.output_size},{self.output_size}), got {P.shape}")

        # 索引一致性校验：KG 文件 feature_names vs 代码定义 FULL_FEATURE_NAMES
        if feature_names is not None:
            try:
                from knowledge_graph import FULL_FEATURE_NAMES
                feature_names = np.asarray(feature_names).astype(str).tolist()
                if feature_names != list(FULL_FEATURE_NAMES):
                    raise ValueError(
                        "KG feature_names 与 knowledge_graph.FULL_FEATURE_NAMES 不一致，"
                        "这会导致 W 与 31维输入特征在索引上错位。"
                    )
            except Exception:
                # 若导入失败/校验失败，抛出以避免静默错位
                raise

        self.W.copy_(torch.from_numpy(W))
        self.P.copy_(torch.from_numpy(P))

    def forward(self, x, y: Optional[torch.Tensor] = None):
        """
        前向传播
        
        Args:
            x: 输入张量，形状为(batch_size, 1, sequence_length)
        Returns:
            输出张量，形状为(batch_size, output_size)
        """
        # 通过第一层卷积块
        x = self.layer1(x)
        # 通过第二层卷积块
        x = self.layer2(x)
        # 通过第三层卷积块
        x = self.layer3(x)
        # 自适应平均池化，输出形状变为 [batch, 64, 64]
        x = self.avgpool(x)  # [batch, 64, 64]
        if not self.use_kg:
            # 展平特征图：将(batch, 64, 64)变为(batch, 64*64)
            x = x.view(x.size(0), -1)  # x.size(0)是批次大小，-1表示自动计算展平后的维度
            # 通过全连接层得到最终分类结果
            x = self.fc(x)
            # 返回分类结果
            return x

        feat = x
        flatten = feat.view(feat.size(0), -1)
        z = self.proj(flatten)  # (B,31)

        if y is not None:
            if y.dim() != 1:
                y = y.view(-1)
            W_y = self.W[y.long()]
            gate = torch.sigmoid(W_y)
        else:
            logits0 = self.classifier(z)
            p = F.softmax(logits0, dim=-1)  # (B,10)
            W_hat = p @ self.W  # (B,31)
            gate = torch.sigmoid(W_hat)

        z_gated = z * gate
        logits = self.classifier(z_gated)
        return logits

class KG_MLP(nn.Module):
    """
    KG-MLP（KG-Physical-Net）

    输入是 31 维“人工物理特征”（严格顺序见 knowledge_graph.FULL_FEATURE_NAMES）。
    知识矩阵 W / P 来自 ./data/kg/*.npz：
      - W: shape (10, 31) 其中 W[c, k] 表示第 c 类故障与第 k 个物理特征的相关性
      - P: shape (10, 10) 故障状态转移矩阵（此实现中作为 buffer 接入，默认不参与前向）

    关键：索引一致性（Feature ↔ W 的对齐）
      - 本仓库的 31 维特征提取函数 `knowledge_graph.signals_to_features` 使用
        `knowledge_graph.FULL_FEATURE_NAMES` 规定的固定顺序输出特征向量 x[k]。
      - KG 文件里也保存了 `feature_names`，我们在加载时做一致性校验，确保
        W 的第 k 行/列（这里是第 k 个特征维度）与输入 x[k] 表示同一个物理特征。
    """
    def __init__(self, output_size):
        super().__init__()
        self.output_size = int(output_size)

        # Feature Alignment Layer: 31 -> 64 -> 31
        self.align = nn.Sequential(
            nn.Linear(31, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 31),
        )

        # 最终分类器（同一个线性层会被用于“无标签软门控”的预估）
        self.classifier = nn.Linear(31, self.output_size)

        # KG matrices as buffers (moved with .to(device))
        self.register_buffer("W", torch.zeros(self.output_size, 31), persistent=True)  # (10,31)
        self.register_buffer("P", torch.eye(self.output_size), persistent=True)       # (10,10)

        # fast_adapt 会用这个 flag 决定是否在 support 传入真值标签进行门控
        self.supports_label_gate = True

    @staticmethod
    def _load_kg_npz(kg_npz_path: str) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        if not os.path.exists(kg_npz_path):
            raise FileNotFoundError(f"KG npz not found: {kg_npz_path}")
        data = np.load(kg_npz_path, allow_pickle=True)
        W = data["W"]  # expected (10,31) in this repo
        P = data["P"]  # expected (10,10)
        feature_names = data["feature_names"] if "feature_names" in data.files else None
        return W, P, feature_names

    @classmethod
    def from_kg_file(cls, output_size: int, kg_npz_path: str) -> "KG_MLP":
        model = cls(output_size=output_size)
        W, P, feature_names = cls._load_kg_npz(kg_npz_path)
        model.set_kg(W=W, P=P, feature_names=feature_names)
        return model

    @torch.no_grad()
    def set_kg(self, W: np.ndarray, P: np.ndarray, feature_names: Optional[np.ndarray] = None):
        """
        在训练循环中可按 domain 动态切换 KG。

        W 允许两种形状：
          - (10,31) 本仓库 build_kg_cwru 的默认输出
          - (31,10) 若外部按“列为类别”保存，可自动转置为 (10,31)
        """
        W = np.asarray(W, dtype=np.float32)
        P = np.asarray(P, dtype=np.float32)
        if W.shape == (31, self.output_size):
            W = W.T
        if W.shape != (self.output_size, 31):
            raise ValueError(f"W shape must be ({self.output_size},31) or (31,{self.output_size}), got {W.shape}")
        if P.shape != (self.output_size, self.output_size):
            raise ValueError(f"P shape must be ({self.output_size},{self.output_size}), got {P.shape}")

        # 索引一致性校验：KG 文件 feature_names vs 代码定义 FULL_FEATURE_NAMES
        if feature_names is not None:
            try:
                from knowledge_graph import FULL_FEATURE_NAMES
                feature_names = np.asarray(feature_names).astype(str).tolist()
                if feature_names != list(FULL_FEATURE_NAMES):
                    raise ValueError(
                        "KG feature_names 与 knowledge_graph.FULL_FEATURE_NAMES 不一致，"
                        "这会导致 W 与 31维输入特征在索引上错位。"
                    )
            except Exception:
                # 若导入失败/校验失败，抛出以避免静默错位
                raise

        self.W.copy_(torch.from_numpy(W))
        self.P.copy_(torch.from_numpy(P))
    
    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None):
        """
        前向传播

        - 训练时（support set 适应阶段）：传入 y（真值标签），门控 Gate=Sigmoid(W_y)
        - 推理/评估时（query set）：不传 y，使用“软门控”避免标签泄漏：
            p = softmax(classifier(aligned_x))
            W_hat = p @ W
            Gate = sigmoid(W_hat)
        """
        if x.dim() != 2 or x.size(-1) != 31:
            raise ValueError(f"KG_MLP expects x shape (B,31), got {tuple(x.shape)}")

        # Feature Alignment (加残差更稳，避免把输入破坏掉)
        aligned = x + self.align(x)

        if y is not None:
            # y: (B,) -> W[y]: (B,31)
            if y.dim() != 1:
                y = y.view(-1)
            W_y = self.W[y.long()]
            gate = torch.sigmoid(W_y)
        else:
            # 无标签：先用未门控的 aligned 做一次粗预测，得到 p，再求期望门控
            logits0 = self.classifier(aligned)
            p = F.softmax(logits0, dim=-1)  # (B,10)
            W_hat = p @ self.W  # (B,31)
            gate = torch.sigmoid(W_hat)

        gated = aligned * gate
        logits = self.classifier(gated)
        return logits