"""
神经网络模型定义
统一管理所有网络结构，便于扩展

架构设计原则：
1. 模块化：编码器、分类器、关系网络分离
2. 可配置：通过参数控制网络结构
3. 可复用：基础组件可在不同方法间共享
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List


class CNN1dEncoder(nn.Module):
    """1D卷积编码器 - 基础特征提取（所有方法共享）

    论文依据：基于原文中统一的1D卷积网络设计
    核心思想：使用大卷积核捕捉振动信号的全局相关性

    网络结构：[1, 1024] → [64, 25] 特征图
    - 4个卷积块，通道数均为64
    - 大首层卷积核（size=10）捕捉长程依赖
    - 自适应池化确保输入长度灵活性
    """

    def __init__(self, feature_dim: int = 64, flatten: bool = False):
        """
        Args:
            feature_dim: 输出特征维度 - 控制模型容量
            flatten: 是否展平输出 - 适应不同任务需求
                    True: 分类任务（FTN/DTN/MAML）
                    False: 关系学习（MRN）
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.flatten = flatten
        # 第1层：大卷积核捕捉全局特征
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=10, stride=3),  # 大核：10，步长：3
            nn.BatchNorm1d(64, momentum=1, affine=True),  # BN：稳定训练
            nn.ReLU(),
            nn.MaxPool1d(2)  # 池化：降维
        )
        # 第2层：特征抽象
        self.layer2 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3),   # 标准3x3卷积
            nn.BatchNorm1d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        # 第3-4层：深层特征提取
        self.layer3 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64, momentum=1, affine=True),
            nn.ReLU()  # 无池化
        )

        self.layer4 = nn.Sequential(
            nn.Conv1d(64, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(feature_dim, momentum=1, affine=True),
            nn.ReLU()
        )
        # 自适应池化：统一输出长度
        self.adaptive_pool = nn.AdaptiveMaxPool1d(25)  # 固定输出25个时间点
        # 输出: [feature_dim, 25]

    def forward(self, x):
        """
        Args:
            x: [batch, 1, 1024] - 输入信号（FFT后）
        Returns:
            features:
                flatten=False: [batch, feature_dim, 25] - 空间特征（MRN）
                flatten=True:  [batch, feature_dim*25]  - 展平特征（分类）
        """
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.adaptive_pool(out)

        if self.flatten:
            out = out.view(out.size(0), -1)  # [batch, feature_dim*25] 展平用于分类器

        return out

    def get_layer_groups(self) -> List[nn.Module]:
        """返回各层（用于选择性冻结）- FTN微调的关键

        理论意义：支持分层迁移学习
        - 浅层特征通用，深层特征特定
        - 不同任务可能需要解冻不同层数
        """
        return [self.layer1, self.layer2, self.layer3, self.layer4,
                self.adaptive_pool]


class RelationNetwork1d(nn.Module):
    """1D关系网络 - 用于度量学习（MRN专用）

    理论基础：学习样本间的关系度量而非直接分类
    输入：查询样本特征 + 支持集原型特征的拼接
    输出：关系分数（相似度）

    网络结构：[128, 25] → [1] 关系分数
    """

    def __init__(self, input_dim: int = 64, hidden_dim: int = 8):
        super().__init__()
        # 关系特征提取
        self.layer1 = nn.Sequential(
            nn.Conv1d(input_dim * 2, input_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(input_dim, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv1d(input_dim, input_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(input_dim, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.fc1 = nn.Linear(input_dim * 6, hidden_dim)  # 全连接降维
        self.fc2 = nn.Linear(hidden_dim, 1)  # 输出关系分数

    def forward(self, x):
        """
        Args:
            x: [batch, input_dim*2, 25] 拼接的特征对
                前input_dim维是查询特征，后input_dim维是支持特征
        Returns:
            relations: [batch, 1] 关系分数（0-1之间）
        """
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = torch.sigmoid(self.fc2(out))
        return out


class LinearClassifier(nn.Module):
    """线性分类器 - 简单有效的分类头

    设计理念：保持分类器简单，让编码器学习有用特征
    常用于：FTN的微调阶段、DTN的直接分类、MAML的基础分类器
    """

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)
        self.fc.bias.data.fill_(0)  # 零初始化偏置

    def forward(self, x):
        return self.fc(x)


def freeze_layers(model: nn.Module, num_unfrozen: int):
    """
    冻结模型层 - FTN微调的核心机制

    理论基础：迁移学习中的分层适应
    - 浅层特征通用，应保持冻结
    - 深层特征特定，可进行微调
    - 解冻层数应与目标域数据量匹配

    Args:
        model: 要冻结的模型
        num_unfrozen: 从后往前保持解冻的层数
                    0: 全冻结（特征迁移）
                    4: 全解冻（接近从头训练）
    """
    # 先全部冻结
    for param in model.parameters():
        param.requires_grad = False

    # 解冻最后num_unfrozen层
    if num_unfrozen > 0:
        children = list(model.children())
        for layer in children[-num_unfrozen:]:
            for param in layer.parameters():
                param.requires_grad = True


def init_weights(model: nn.Module):
    """权重初始化 - 确保训练稳定性

    使用He初始化（ReLU激活函数的推荐初始化）
    理论依据：保持前向传播中的方差稳定
    """
    for m in model.modules():
        if isinstance(m, nn.Conv1d):
            # He初始化：从N(0, sqrt(2/fan_in))采样
            n = m.kernel_size[0] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm1d):
            # BN层：权重1，偏置0
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            # 线性层：小随机初始化
            m.weight.data.normal_(0, 0.01)
            if m.bias is not None:
                m.bias.data.fill_(1)

