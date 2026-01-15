from __future__ import annotations

"""
InceptionTime backbone for 1D time-series classification.

参考实现：
    - InceptionTime: Finding AlexNet for Time Series Classification (Fawaz et al.)

该实现只提供特征提取（global average pooling 输出），方便与上层分类头解耦。
"""

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class InceptionTimeConfig:
    in_channels: int = 1
    num_blocks: int = 3
    out_channels: int = 32
    bottleneck_channels: int = 32
    kernel_sizes: Sequence[int] = (41, 21, 11)
    use_residual: bool = True
    dropout: float = 0.1


class InceptionModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: Iterable[int],
        bottleneck_channels: int,
        use_bottleneck: bool,
    ) -> None:
        super().__init__()
        kernel_sizes = tuple(kernel_sizes)
        self.use_bottleneck = use_bottleneck and in_channels > 1

        if self.use_bottleneck:
            self.bottleneck = nn.Conv1d(
                in_channels, bottleneck_channels, kernel_size=1, bias=False
            )
            conv_in = bottleneck_channels
        else:
            self.bottleneck = None
            conv_in = in_channels

        self.conv_layers = nn.ModuleList(
            [
                nn.Conv1d(
                    conv_in,
                    out_channels,
                    kernel_size=k,
                    padding=k // 2,
                    bias=False,
                )
                for k in kernel_sizes
            ]
        )
        self.max_pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.conv_pool = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)

        self.bn = nn.BatchNorm1d(out_channels * (len(kernel_sizes) + 1))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_bottleneck:
            x_b = self.bottleneck(x)
        else:
            x_b = x

        conv_outs = [conv(x_b) for conv in self.conv_layers]
        pool_out = self.conv_pool(self.max_pool(x))
        x_cat = torch.cat(conv_outs + [pool_out], dim=1)
        x_cat = self.bn(x_cat)
        return self.relu(x_cat)


class InceptionBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: Iterable[int],
        bottleneck_channels: int,
        num_modules: int = 3,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        kernel_sizes = tuple(kernel_sizes)
        self.use_residual = use_residual
        self.num_modules = num_modules

        modules = []
        channels = in_channels
        for _ in range(num_modules):
            modules.append(
                InceptionModule(
                    in_channels=channels,
                    out_channels=out_channels,
                    kernel_sizes=kernel_sizes,
                    bottleneck_channels=bottleneck_channels,
                    use_bottleneck=True,
                )
            )
            channels = out_channels * (len(kernel_sizes) + 1)
        self.modules_list = nn.ModuleList(modules)

        if self.use_residual:
            self.residual = nn.Sequential(
                nn.Conv1d(in_channels, channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(channels),
            )
        else:
            self.residual = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for module in self.modules_list:
            out = module(out)

        if self.use_residual:
            out = F.relu(out + self.residual(x), inplace=True)
        return out


class InceptionTimeBackbone(nn.Module):
    def __init__(self, config: InceptionTimeConfig) -> None:
        super().__init__()
        self.config = config

        blocks = []
        in_channels = config.in_channels
        for _ in range(config.num_blocks):
            block = InceptionBlock(
                in_channels=in_channels,
                out_channels=config.out_channels,
                kernel_sizes=config.kernel_sizes,
                bottleneck_channels=config.bottleneck_channels,
                num_modules=3,
                use_residual=config.use_residual,
            )
            blocks.append(block)
            in_channels = config.out_channels * (len(tuple(config.kernel_sizes)) + 1)
        self.blocks = nn.Sequential(*blocks)

        self.dropout = nn.Dropout(p=config.dropout)
        self.output_dim = in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"InceptionTime 期望输入形状 (B,C,T)，但收到 {x.shape}")

        x = self.blocks(x)
        x = x.mean(dim=-1)
        x = self.dropout(x)
        return x


__all__ = ["InceptionTimeConfig", "InceptionTimeBackbone"]
