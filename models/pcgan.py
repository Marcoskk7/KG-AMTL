from __future__ import annotations

"""
Physics-Constrained Generative Augmentation (PC-GAN)

本模块主要来源于 `reference/code/CGanByMa.ipynb` 的 Part 2：
    - 条件提供器 `ConditionProvider`
    - 可微特征提取器 `DifferentiableFeatures`
    - 可微四频带能量 `FourBandEnergy`
    - 条件生成器 `CondGenerator1D`
    - 条件判别器 `CondProjectionDiscriminator1D`
    - 物理约束损失 `PhysicsConstraintLoss`
    - 数据集封装 `SignalsByClass`
    - 训练器 `PCG_Trainer`

本文件不负责数据加载与特征/KG 计算，只接受上游预先计算好的：
    - v_real: (C, D_feat) 类特征中心
    - w_real: (C, D_feat) 特征-故障相关权重
    - E_c   : (C, 4)      每类四频带能量比例
    - P     : (C, C)      可选，故障转移矩阵，仅作为 D 的条件先验
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter


# =============== 0. 实用函数 & 归一化 ==================
def to_tensor(x, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def minmax_scale_np(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    按样本把时域信号缩放到 [-1, 1]，匹配生成器 Tanh 输出。

    支持形状:
        - (N, T)
        - (N, 1, T)
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        xmax = np.max(np.abs(x), axis=1, keepdims=True) + eps
        x = x / xmax
    elif x.ndim == 3:  # (N,1,T)
        xmax = np.max(np.abs(x), axis=(1, 2), keepdims=True) + eps
        x = x / xmax
    return x


# =============== 1. 条件提供器 ==================
class ConditionProvider:
    """
    根据类别标签生成条件向量，向 G / D 提供先验信息。

    - G 条件：concat[w_c, E_c]
    - D 条件：concat[w_c, E_c] 或 concat[w_c, E_c, P_c] （若提供转移矩阵 P）
    """

    def __init__(
        self,
        class_names,
        w_real: np.ndarray,
        E_c: np.ndarray,
        P: Optional[np.ndarray] = None,
    ):
        self.class_names = list(class_names)
        self.C = len(class_names)
        self.w = np.asarray(w_real, dtype=np.float32)
        self.E = np.asarray(E_c, dtype=np.float32)
        self.P = None if P is None else np.asarray(P, dtype=np.float32)

        if self.w.shape[0] != self.C:
            raise ValueError("w_real 行数必须等于类别数 C")
        if self.E.shape[0] != self.C or self.E.shape[1] != 4:
            raise ValueError("E_c 形状应为 (C,4)")
        if self.P is not None and self.P.shape != (self.C, self.C):
            raise ValueError("P 形状应为 (C,C)")

    # --- 提供给生成器的条件 ---
    def get_cond_vectors_G(self, y: np.ndarray) -> np.ndarray:
        """G 的条件：concat[w[y], E_c[y]] -> (N, D_feat + 4)"""
        y = np.asarray(y, dtype=np.int64)
        w_take = self.w[y]  # (N, D_feat)
        E_take = self.E[y]  # (N, 4)
        return np.concatenate([w_take, E_take], axis=1)

    # --- 提供给判别器的条件 ---
    def get_cond_vectors_D(self, y: np.ndarray) -> np.ndarray:
        """D 的条件：concat[w[y], E_c[y], (可选)P[y]]"""
        y = np.asarray(y, dtype=np.int64)
        w_take = self.w[y]
        E_take = self.E[y]
        if self.P is None:
            return np.concatenate([w_take, E_take], axis=1)
        else:
            P_take = self.P[y]
            return np.concatenate([w_take, E_take, P_take], axis=1)


# =============== 2. 可微特征提取器 φ(x) ==================
class DifferentiableFeatures(nn.Module):
    """
    用 rFFT 幅谱的若干维度近似 31 维物理特征，保持可微性以回传到 G。

    实际实现：
        - 对 rFFT 幅度谱做线性插值采样到 d_feat 维。
        - 仅作为物理约束中 w_gen 估计的近似特征。
    """

    def __init__(self, sample_len: int = 2400, d_feat: int = 31):
        super().__init__()
        self.sample_len = sample_len
        self.d_feat = d_feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            形状 (B, 1, T) 的时域信号。
        Returns
        -------
        feats : torch.Tensor
            形状 (B, d_feat) 的频域幅值近似特征。
        """
        if x.dim() != 3 or x.size(1) != 1:
            raise ValueError(f"x 期望形状为 (B,1,T)，但收到 {tuple(x.shape)}")

        X = torch.fft.rfft(x.squeeze(1), dim=-1)  # (B, T//2+1)
        mag = torch.abs(X)

        B, n_bins = mag.shape

        # 若频点数少于 d_feat，右侧零填充
        if n_bins < self.d_feat:
            pad = self.d_feat - n_bins
            mag = F.pad(mag, (0, pad))
            n_bins = mag.shape[1]

        # 在线性坐标上插值采样到 d_feat 维
        idx = torch.linspace(0, n_bins - 1, self.d_feat, device=mag.device)
        idx0 = idx.floor().long().clamp(0, n_bins - 2)
        alpha = (idx - idx0.float())  # (d_feat,)

        mag0 = mag[:, idx0]  # (B, d_feat)
        mag1 = mag[:, idx0 + 1]
        feats = mag0 * (1.0 - alpha) + mag1 * alpha
        return feats


# =============== 2.1 可微四频带能量比例（近似 VMD 能量） ==================
class FourBandEnergy(nn.Module):
    """
    用 STFT 幅度谱在 4 个频带上积分，得到能量比例 (B,4)。

    若 learnable_bands=True，则频带上界以单调参数化方式可学习。
    """

    def __init__(
        self,
        T: int,
        fs: int = 12000,
        nfft: int = 512,
        win_len: int = 256,
        hop: int = 128,
        bands=None,
        learnable_bands: bool = False,
    ):
        super().__init__()
        self.fs = fs
        self.nfft = nfft
        self.hop = hop
        self.win_len = win_len
        self.register_buffer("window", torch.hann_window(win_len))

        if bands is None:
            # 默认四频带（可根据转速 / 特征频自行调整）
            bands = [(0, 600), (600, 1800), (1800, 3600), (3600, 6000)]
        self.learnable_bands = learnable_bands
        if learnable_bands:
            # 用递增的累积 softplus 参数化边界，确保 0 < b1 < b2 < b3 < fs/2
            init = torch.tensor(
                [bands[0][1], bands[1][1], bands[2][1], bands[3][1]],
                dtype=torch.float32,
            )
            self.b_raw = nn.Parameter(
                torch.log(torch.exp(init / (fs / 2 - 1e-3)) - 1.0)
            )
        else:
            self.bands = bands

    def _get_edges(self, device: torch.device) -> torch.Tensor:
        if not self.learnable_bands:
            edges = torch.tensor(
                [b[1] for b in self.bands], dtype=torch.float32, device=device
            )
        else:
            sp = F.softplus(self.b_raw)  # 正数
            cum = torch.cumsum(sp, dim=0)
            edges = (self.fs / 2) * (cum / (cum[-1] + 1e-8))
        e0 = torch.tensor([0.0], device=device)
        edges = torch.cat([e0, edges], dim=0)  # (5,) -> [0,b1,b2,b3,b4]
        return edges

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,1,T)
        返回 (B,4) 四频带能量比例向量。
        """
        if x.dim() != 3 or x.size(1) != 1:
            raise ValueError(f"x 期望形状为 (B,1,T)，但收到 {tuple(x.shape)}")

        X = torch.stft(
            x.squeeze(1),
            n_fft=self.nfft,
            hop_length=self.hop,
            win_length=self.win_len,
            window=self.window.to(x.device),
            return_complex=True,
            center=True,
            pad_mode="reflect",
        )  # (B, F, T')
        mag2 = X.real**2 + X.imag**2  # (B,F,T') 能量谱
        Fbins = mag2.size(1)
        freqs = torch.linspace(0, self.fs / 2, Fbins, device=x.device)  # (F,)

        edges = self._get_edges(x.device)  # (5,)
        Es = []
        for k in range(4):
            f0, f1 = edges[k], edges[k + 1]
            mask = (freqs >= f0) & (freqs < f1)  # (F,)
            if mask.sum() == 0:
                band_E = mag2.new_zeros((x.size(0),))
            else:
                band_E = mag2[:, mask, :].sum(dim=(1, 2))
            Es.append(band_E)
        E = torch.stack(Es, dim=1)  # (B,4)
        E_ratio = E / (E.sum(dim=1, keepdim=True) + 1e-8)
        return E_ratio


# =============== 3. 生成器 G(z|cond) ==================
@dataclass
class PCGANGeneratorConfig:
    """
    生成器结构配置，便于保存到 checkpoint 并在推理时重建。
    """

    cond_dim: int
    z_dim: int = 128
    out_len: int = 2400
    base_ch: int = 128
    emb_dim: int = 16
    num_classes: int = 4


class CondGenerator1D(nn.Module):
    """
    条件 1D 生成器：
        - 输入: 噪声 z, 条件向量 cond_vec, 类别标签 y
        - 输出: (B,1,T) 归一化到 [-1,1] 的时域信号
    """

    def __init__(self, config: PCGANGeneratorConfig):
        super().__init__()
        self.config = config
        self.out_len = config.out_len
        self.z_dim = config.z_dim

        self.emb = nn.Embedding(config.num_classes, config.emb_dim)
        self.cond_proj = nn.Linear(config.cond_dim + config.emb_dim, config.z_dim)

        # 设计固定的上采样链：75 -> 150 -> 300 -> 600 -> 1200 -> 2400
        self.init_len = 75
        self.fc = nn.Linear(config.z_dim * 2, config.base_ch * self.init_len)

        ch = config.base_ch
        self.net = nn.Sequential(
            nn.ConvTranspose1d(ch, ch // 2, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(ch // 2, ch // 4, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(ch // 4, ch // 8, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(ch // 8, ch // 16, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(ch // 16, 1, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(
        self, z: torch.Tensor, cond_vec: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        if z.dim() != 2:
            raise ValueError(f"z 期望形状为 [B, z_dim]，但收到 {tuple(z.shape)}")
        if cond_vec.dim() != 2:
            raise ValueError(
                f"cond_vec 期望形状为 [B, cond_dim]，但收到 {tuple(cond_vec.shape)}"
            )
        if y.dim() != 1 or y.size(0) != z.size(0):
            raise ValueError("y 形状应为 [B] 且与 z 批大小一致")

        emb = self.emb(y)  # (B, emb_dim)
        cond_full = torch.cat([cond_vec, emb], dim=1)
        c = self.cond_proj(cond_full)  # (B, z_dim)
        h = torch.cat([z, c], dim=1)  # (B, 2*z_dim)
        h = self.fc(h)  # (B, base_ch*init_len)

        B = h.size(0)
        C = h.numel() // (B * self.init_len)
        h = h.view(B, C, self.init_len)
        x = self.net(h)  # (B,1,out_len)
        return x


# =============== 4. 判别器 D(x|cond)：Projection ==================
class CondProjectionDiscriminator1D(nn.Module):
    def __init__(self, cond_dim: int, base_ch: int = 64):
        super().__init__()
        self.cond_proj = nn.Linear(cond_dim, base_ch * 4)
        self.backbone = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv1d(1, base_ch, 9, 2, 4)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv1d(base_ch, base_ch * 2, 9, 2, 4)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv1d(base_ch * 2, base_ch * 4, 9, 2, 4)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.uncond_head = nn.utils.spectral_norm(
            nn.Conv1d(base_ch * 4, 1, 7, padding=3)
        )

    def forward(
        self, x: torch.Tensor, cond_vec: torch.Tensor, noise_std: float = 0.02
    ) -> torch.Tensor:
        """
        x: (B,1,T)
        cond_vec: (B, cond_dim)
        """
        if x.dim() != 3 or x.size(1) != 1:
            raise ValueError(f"x 期望形状为 (B,1,T)，但收到 {tuple(x.shape)}")
        if cond_vec.dim() != 2 or cond_vec.size(0) != x.size(0):
            raise ValueError("cond_vec 形状应为 (B,cond_dim) 且批大小与 x 一致")

        if noise_std > 0:
            x = x + noise_std * torch.randn_like(x)
        h = self.backbone(x)  # (B, C, T')
        g = h.mean(dim=-1)  # (B, C)
        logits_uncond = self.uncond_head(h).mean(dim=-1).squeeze(1)  # (B,)
        e = self.cond_proj(cond_vec)  # (B, C)
        logits = logits_uncond + (g * e).sum(dim=1)
        return logits


# =============== 5. 物理约束损失 L_phy（含类内 EMA） ==================
class PhysicsConstraintLoss(nn.Module):
    """
    L_phy = ||w_gen - w_real||_2^2 + alpha_E * D_ratio(E_gen, E_c)

    其中：
      - w_gen：用 φ(x_gen) 的类内统计与 v_real 对齐后，按 EMA 聚合估计；
      - E_gen：可微四频带能量比例（近似 VMD）；
      - D_ratio：默认 L1，可选 'kl' 或 'logcosh'。
    """

    def __init__(
        self,
        v_real: np.ndarray,
        w_real: np.ndarray,
        E_c: np.ndarray,
        feat_func: nn.Module,
        band_energy: nn.Module,
        eps: float = 1e-8,
        alpha_E: float = 1.0,
        ratio_metric: str = "l1",
        ema_momentum: float = 0.1,
    ):
        super().__init__()
        v_real = np.asarray(v_real, dtype=np.float32)  # (C,D)
        w_real = np.asarray(w_real, dtype=np.float32)  # (C,D)
        E_c = np.asarray(E_c, dtype=np.float32)  # (C,4)
        C, D = v_real.shape
        if w_real.shape != (C, D):
            raise ValueError("w_real shape mismatch!")

        self.C, self.D = C, D
        self.register_buffer("v", torch.tensor(v_real))
        self.register_buffer("w_target", torch.tensor(w_real))
        self.register_buffer("E_target", torch.tensor(E_c))
        self.phi = feat_func
        self.band_energy = band_energy
        self.eps = eps
        self.alpha_E = alpha_E
        self.ratio_metric = ratio_metric
        self.m = ema_momentum

        # 类内 EMA 的二阶中心矩（围绕 v_real 的平方差）缓存
        self.register_buffer("ema_var", torch.ones(C, D))
        self.register_buffer("ema_ready", torch.zeros(C))  # 标记是否初始化

    def _ratio_distance(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        if self.ratio_metric == "l1":
            return F.l1_loss(p, q)
        elif self.ratio_metric == "kl":
            p = p + 1e-8
            q = q + 1e-8
            return 0.5 * (
                (p * (p / q).log()).sum(dim=1).mean()
                + (q * (q / p).log()).sum(dim=1).mean()
            )
        elif self.ratio_metric == "logcosh":
            return torch.log(torch.cosh(p - q)).mean()
        else:
            return F.l1_loss(p, q)

    def forward(self, x_gen: torch.Tensor, y: torch.Tensor):
        # ---- 权重项：用 φ(x_gen) 的类内方差（围绕 v_real） → w_gen，并做 EMA 聚合 ----
        feats = self.phi(x_gen)  # (B, D_phi)
        if feats.shape[1] != self.D:
            # 线性投影/切片到 D 维
            feats = (
                feats[:, : self.D]
                if feats.shape[1] > self.D
                else F.pad(feats, (0, self.D - feats.shape[1]))
            )

        loss_w = 0.0
        classes = y.unique()
        for i in classes:
            i_int = int(i.item())
            mask = y == i_int
            Fi = feats[mask]  # (B_i, D)
            if Fi.numel() == 0:
                continue
            vi = self.v[i_int]  # (D,)
            cur_var = ((Fi - vi.unsqueeze(0)) ** 2).mean(dim=0)  # (D,)

            if self.ema_ready[i_int] < 0.5:
                self.ema_var[i_int] = cur_var.detach()
                self.ema_ready[i_int] = 1.0
            else:
                self.ema_var[i_int] = (1 - self.m) * self.ema_var[
                    i_int
                ] + self.m * cur_var.detach()
            invsqrt = torch.rsqrt(self.ema_var[i_int] + self.eps)
            wi_gen = invsqrt / (invsqrt.sum() + self.eps)
            wi_tgt = self.w_target[i_int]  # (D,)
            loss_w = loss_w + F.mse_loss(wi_gen, wi_tgt)
        loss_w = loss_w / max(len(classes), 1)

        # ---- 能量比例项：可微四频带能量 vs. 目标 E_c ----
        Ec_gen = self.band_energy(x_gen)  # (B,4)
        Ec_tgt = self.E_target[y]  # (B,4)
        loss_E = self._ratio_distance(Ec_gen, Ec_tgt)

        return loss_w + self.alpha_E * loss_E, loss_w.detach(), loss_E.detach()


# =============== 6. 数据集包装 ==================
class SignalsByClass(Dataset):
    """
    将原始信号和标签打包为 Dataset，内部自动完成逐段 min-max 归一化。
    """

    def __init__(self, X_signals: np.ndarray, y: np.ndarray, do_minmax: bool = True):
        X = np.asarray(X_signals, dtype=np.float32)
        if do_minmax:
            X = minmax_scale_np(X)
        if X.ndim == 2:
            X = X[:, None, :]  # (N,1,T)
        self.X = X
        self.y = np.asarray(y, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.X[idx]), torch.tensor(
            self.y[idx], dtype=torch.long
        )


# =============== 7. 训练器 ==================
class PCG_Trainer:
    """
    PC-GAN 训练器：HingeGAN + ProjectionD + 物理约束损失。
    """

    def __init__(
        self,
        X_signals: np.ndarray,
        y: np.ndarray,
        class_names,
        v_real: np.ndarray,  # (C,D_feat)
        w_real: np.ndarray,  # (C,D_feat)
        E_c: Optional[np.ndarray] = None,  # (C,4)，若 None 则由 estimate_Ec_from_real 估计
        P: Optional[np.ndarray] = None,  # (C,C) 可选，仅喂 D
        batch_size: int = 64,
        z_dim: int = 128,
        lr_g: float = 2e-4,
        lr_d: float = 1e-4,
        lambda_phys: float = 2.0,
        alpha_E: float = 1.0,
        device: Optional[str] = None,
        fs: int = 12000,
        n_critic: int = 3,
        ratio_metric: str = "l1",
        ema_momentum: float = 0.1,
        learnable_bands: bool = False,
        lambda_warmup_steps: int = 1000,
        log_dir: Optional[str] = None,
        use_tensorboard: bool = True,
        base_ch: int = 128,
        emb_dim: int = 16,
    ):
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.dataset = SignalsByClass(X_signals, y, do_minmax=True)
        self.loader = DataLoader(
            self.dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )

        # TensorBoard & 训练历史
        self.use_tensorboard = use_tensorboard
        self.log_dir = log_dir
        self.writer: Optional[SummaryWriter] = None
        if self.use_tensorboard:
            if self.log_dir is None:
                self.log_dir = "runs/PCG_Trainer"
            self.writer = SummaryWriter(log_dir=self.log_dir)

        self.history: Dict[str, list] = {
            "step": [],
            "epoch": [],
            "d_loss": [],
            "g_loss": [],
            "gan_loss": [],
            "phy_loss": [],
            "loss_w": [],
            "loss_E": [],
            "D_real_acc": [],
            "D_fake_acc": [],
            "lambda_phys": [],
        }

        # 条件 / 先验
        if E_c is None:
            E_c = estimate_Ec_from_real(X_signals, y, fs=fs)
        self.cond_provider = ConditionProvider(class_names, w_real, E_c, P=P)
        dummy_G = self.cond_provider.get_cond_vectors_G(np.array([0]))
        dummy_D = self.cond_provider.get_cond_vectors_D(np.array([0]))
        cond_dim_G = dummy_G.shape[1]
        cond_dim_D = dummy_D.shape[1]

        # 保存知识先验，便于外部访问 / 保存 checkpoint
        self.v_real = np.asarray(v_real, dtype=np.float32)
        self.w_real = np.asarray(w_real, dtype=np.float32)
        self.E_c = np.asarray(E_c, dtype=np.float32)
        self.class_names = list(class_names)

        # 生成器配置（用于 checkpoint）
        T = self.dataset.X.shape[-1]
        num_classes = len(class_names)
        self.gen_config = PCGANGeneratorConfig(
            cond_dim=cond_dim_G,
            z_dim=z_dim,
            out_len=T,
            base_ch=base_ch,
            emb_dim=emb_dim,
            num_classes=num_classes,
        )

        # 模型
        self.G = CondGenerator1D(self.gen_config).to(self.device)
        self.D = CondProjectionDiscriminator1D(cond_dim=cond_dim_D).to(self.device)
        self.phi = DifferentiableFeatures(sample_len=T, d_feat=w_real.shape[1]).to(
            self.device
        )
        self.band_energy = FourBandEnergy(
            T=T, fs=fs, learnable_bands=learnable_bands
        ).to(self.device)
        self.phys_loss = PhysicsConstraintLoss(
            v_real=v_real,
            w_real=w_real,
            E_c=E_c,
            feat_func=self.phi,
            band_energy=self.band_energy,
            alpha_E=alpha_E,
            ratio_metric=ratio_metric,
            ema_momentum=ema_momentum,
        ).to(self.device)

        # 优化器（TTUR）
        self.optG = torch.optim.Adam(self.G.parameters(), lr=lr_g, betas=(0.5, 0.999))
        self.optD = torch.optim.Adam(self.D.parameters(), lr=lr_d, betas=(0.5, 0.999))

        self.lambda_phys_base = float(lambda_phys)
        self.lambda_phys = 0.0
        self.lambda_warmup_steps = int(lambda_warmup_steps)
        self.n_critic = int(n_critic)
        self.step = 0

    # ---- 内部工具函数 ----
    def _lambda_update(self):
        if self.lambda_warmup_steps <= 0:
            self.lambda_phys = self.lambda_phys_base
        else:
            t = min(self.step / self.lambda_warmup_steps, 1.0)
            self.lambda_phys = (0.1 + 0.9 * t) * self.lambda_phys_base

    def sample_noise(self, B: int, z_dim: int) -> torch.Tensor:
        return torch.randn(B, z_dim, device=self.device)

    def make_cond(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        condG_np = self.cond_provider.get_cond_vectors_G(y.cpu().numpy())
        condD_np = self.cond_provider.get_cond_vectors_D(y.cpu().numpy())
        return to_tensor(condG_np, self.device), to_tensor(condD_np, self.device)

    @staticmethod
    def d_hinge_loss(
        real_logits: torch.Tensor, fake_logits: torch.Tensor
    ) -> torch.Tensor:
        return F.relu(1 - real_logits).mean() + F.relu(1 + fake_logits).mean()

    @staticmethod
    def g_hinge_loss(fake_logits: torch.Tensor) -> torch.Tensor:
        return -fake_logits.mean()

    # ---- 训练主循环 ----
    def fit(self, epochs: int = 50, log_every: int = 50, d_noise_std: float = 0.02):
        for ep in range(epochs):
            for xb, yb in self.loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                B = xb.size(0)
                condG, condD = self.make_cond(yb)

                # ---- 训练判别器 D (n_critic 次) ----
                for _ in range(self.n_critic):
                    self.optD.zero_grad()
                    real_logits = self.D(xb, condD, noise_std=d_noise_std)
                    z = self.sample_noise(B, self.G.z_dim)
                    x_fake = self.G(z, condG, yb).detach()
                    fake_logits = self.D(x_fake, condD, noise_std=d_noise_std)
                    d_loss = self.d_hinge_loss(real_logits, fake_logits)
                    d_loss.backward()
                    self.optD.step()

                # ---- 训练生成器 G ----
                self.optG.zero_grad()
                z = self.sample_noise(B, self.G.z_dim)
                x_fake = self.G(z, condG, yb)
                fake_logits = self.D(x_fake, condD, noise_std=0.0)
                gan_loss = self.g_hinge_loss(fake_logits)
                phy, loss_w_det, loss_E_det = self.phys_loss(x_fake, yb)

                self._lambda_update()
                g_loss = gan_loss + self.lambda_phys * phy
                g_loss.backward()
                self.optG.step()

                # 统计
                with torch.no_grad():
                    D_real = (real_logits > 0).float().mean().item()
                    D_fake = (fake_logits < 0).float().mean().item()

                self.history["step"].append(self.step)
                self.history["epoch"].append(ep)
                self.history["d_loss"].append(float(d_loss.item()))
                self.history["g_loss"].append(float(g_loss.item()))
                self.history["gan_loss"].append(float(gan_loss.item()))
                self.history["phy_loss"].append(float(phy.item()))
                self.history["loss_w"].append(float(loss_w_det.mean().item()))
                self.history["loss_E"].append(float(loss_E_det.mean().item()))
                self.history["D_real_acc"].append(float(D_real))
                self.history["D_fake_acc"].append(float(D_fake))
                self.history["lambda_phys"].append(float(self.lambda_phys))

                if self.writer is not None:
                    self.writer.add_scalar("loss/D", d_loss.item(), self.step)
                    self.writer.add_scalar("loss/G_total", g_loss.item(), self.step)
                    self.writer.add_scalar("loss/G_gan", gan_loss.item(), self.step)
                    self.writer.add_scalar("loss/L_phy", phy.item(), self.step)
                    self.writer.add_scalar(
                        "loss/L_w", loss_w_det.mean().item(), self.step
                    )
                    self.writer.add_scalar(
                        "loss/L_E", loss_E_det.mean().item(), self.step
                    )
                    self.writer.add_scalar("D/real>0", D_real, self.step)
                    self.writer.add_scalar("D/fake<0", D_fake, self.step)
                    self.writer.add_scalar("lambda/phys", self.lambda_phys, self.step)

                if self.step % log_every == 0:
                    print(
                        f"[ep {ep:03d} step {self.step:06d}] "
                        f"D: {d_loss.item():.4f} | G: {g_loss.item():.4f} | GAN: {gan_loss.item():.4f} "
                        f"| λ_phy:{self.lambda_phys:.3f} | L_phy:{phy.item():.4f} "
                        f"(w:{loss_w_det.mean().item():.4f}, E:{loss_E_det.mean().item():.4f}) "
                        f"| D_real>0:{D_real:.3f} | D_fake<0:{D_fake:.3f}"
                    )

                self.step += 1

    # ---- 合成样本 ----
    @torch.no_grad()
    def synthesize(self, y: np.ndarray, num_per_class: int = 10) -> np.ndarray:
        """
        根据给定类别 id 列表，为每类生成若干条合成信号。

        Returns
        -------
        Xg : np.ndarray
            形状 (len(y) * num_per_class, T) 的生成信号矩阵。
        """
        y = np.asarray(y, dtype=np.int64)
        all_out = []
        for yi in y:
            condG_np = self.cond_provider.get_cond_vectors_G(np.array([yi]))
            condG = to_tensor(condG_np, self.device).repeat(num_per_class, 1)
            z = self.sample_noise(num_per_class, self.G.z_dim)
            xg = self.G(
                z,
                condG,
                torch.full(
                    (num_per_class,), yi, dtype=torch.long, device=self.device
                ),
            )
            all_out.append(xg.squeeze(1).cpu().numpy())
        Xg = np.concatenate(all_out, axis=0)
        return Xg


# =============== 8. 从真实信号估计 E_c（如未提供） ==================
def estimate_Ec_from_real(
    X_signals: np.ndarray, y: np.ndarray, fs: int = 12000
) -> np.ndarray:
    """
    使用 FourBandEnergy 在真实样本上估计每类的平均四频带能量比例 E_c。
    """
    from collections import defaultdict

    tmp_ds = SignalsByClass(X_signals, y, do_minmax=True)
    loader = DataLoader(tmp_ds, batch_size=128, shuffle=False, drop_last=False)
    band = FourBandEnergy(T=tmp_ds.X.shape[-1], fs=fs)
    by_cls: Dict[int, list] = defaultdict(list)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    band = band.to(device)
    for xb, yb in loader:
        xb = xb.to(device)
        e = band(xb).cpu().numpy()  # (B,4)
        for ei, yi in zip(e, yb.numpy().tolist()):
            by_cls[yi].append(ei)
    C = len(np.unique(y))
    Ec = np.zeros((C, 4), dtype=np.float32)
    for c in range(C):
        if by_cls[c]:
            Ec[c] = np.mean(np.stack(by_cls[c], axis=0), axis=0)
    return Ec


__all__ = [
    "to_tensor",
    "minmax_scale_np",
    "ConditionProvider",
    "DifferentiableFeatures",
    "FourBandEnergy",
    "PCGANGeneratorConfig",
    "CondGenerator1D",
    "CondProjectionDiscriminator1D",
    "PhysicsConstraintLoss",
    "SignalsByClass",
    "PCG_Trainer",
    "estimate_Ec_from_real",
]


