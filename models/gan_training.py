from __future__ import annotations

"""
基于 notebook `CGanByMa.ipynb` 的新版 Physics-Constrained GAN 训练封装。

本文件作为「脚本级入口」，严格对齐 notebook Part 1 + Part 2 中的 2.2 实验设置：
    - 负责：
        * 加载 CWRU 信号
        * 提取 31 维物理特征
        * 按 notebook 中 `compute_feature_fault_weights` 的公式，基于硬标签计算：
              - 类特征中心 v_real
              - 特征-故障相关权重 W_real
        * 物理能量先验 E_c 由 `PCG_Trainer` 内部基于 FourBandEnergy 自动估计
        * 构造并调用 `models.pcgan.PCG_Trainer` 完成对抗 + 物理约束训练
        * 保存训练好的生成器及先验信息到 checkpoint
    - PC-GAN 具体网络结构与损失定义见 `models/pcgan.py`。

与脚本的对齐：
    - `scripts/run_augmentation_and_meta.py` 仍通过
      `from models.gan_training import GANTrainConfig, train_gan_with_physics`
      调用本文件中的接口。
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from data.cwru_loader import CATEGORY_ORDER, load_cwru_signals
from features.pipeline import FULL_FEATURE_NAMES, batch_extract_features
from models.pcgan import (
    CondGenerator1D,
    ConditionProvider,
    PCG_Trainer,
    PCGANGeneratorConfig,
)
from utils.visualization import (
    plot_time_series_compare,
    plot_vmd_energy_compare,
    setup_chinese_font,
    tsne_and_plot_ax,
)

# 与 notebook 中一致的中间结果保存目录（可选，用于调试/可视化）
KG_SAVE_DIR = "./knowledge_graphs"


def compute_feature_fault_weights(
    X: np.ndarray,
    y: np.ndarray,
    class_names: List[str],
    feature_names: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    按 notebook `CGanByMa.ipynb` 中 Cell 16 的实现计算特征-故障相关权重：

    参数
    ----
    X : (N, D)
        31 维特征矩阵。
    y : (N,)
        硬标签 [0..C-1]。
    class_names : list[str]
        类别名称列表，长度为 C。
    feature_names : list[str]
        特征名称列表，长度为 D。

    返回
    ----
    v : (C, D)
        类中心 v_{ik}。
    sigma : (C, D)
        类内离散度 sigma_{ik} = sum_j u_{ij}(x_{jk}-v_{ik})^2。
    w : (C, D)
        相关性权重 w_{ik} ∝ (sigma_{ik})^{-1/2}，并在 k 上归一。
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    N, D = X.shape
    C = len(class_names)

    if N != y.shape[0]:
        raise ValueError(f"X 样本数 {N} 与 y 长度 {y.shape[0]} 不一致")
    if D != len(feature_names):
        raise ValueError(
            f"特征维度 D={D} 与 feature_names 长度 {len(feature_names)} 不一致"
        )

    # 成员度 U (N, C) —— 此处与 notebook 一致，使用硬标签 one-hot
    U = np.zeros((N, C), dtype=float)
    U[np.arange(N), y] = 1.0

    # v_{ik}
    denom = U.sum(axis=0, keepdims=True).T  # (C,1)
    denom[denom == 0] = 1e-12
    v = (U.T @ X) / denom  # (C,D)

    # sigma_{ik}
    sigma = np.zeros((C, D), dtype=float)
    for i in range(C):
        diff = X - v[i]  # (N,D)
        sigma[i] = (U[:, i][:, None] * (diff**2)).sum(axis=0)
    sigma = np.maximum(sigma, 1e-12)

    # w_{ik} ∝ (sigma_{ik})^{-1/2}，并在 k 维上归一
    inv_sqrt = 1.0 / np.sqrt(sigma)
    w = inv_sqrt / inv_sqrt.sum(axis=1, keepdims=True)

    # 便于检查：每类权重之和应为 1
    if not np.allclose(w.sum(axis=1), 1.0, atol=1e-6):
        raise RuntimeError("w 每行的权重和未能归一到 1，请检查实现。")

    # 与 notebook 一致：可选地将中间结果保存到磁盘，便于后续可视化/复用
    try:
        os.makedirs(KG_SAVE_DIR, exist_ok=True)
        np.savez(
            os.path.join(KG_SAVE_DIR, "kg_step2_w_v_sigma.npz"),
            w=w,
            v=v,
            sigma=sigma,
            feature_names=np.array(feature_names),
            class_names=np.array(class_names),
        )
    except Exception as e:  # 保存失败不影响训练流程
        print(f"[警告] 保存 kg_step2_w_v_sigma.npz 失败: {e}")

    return v, sigma, w


@dataclass
class GANTrainConfig:
    """
    PC-GAN 训练配置。

    主要参数与之前版本保持一致，方便从命令行或脚本调用。
    """

    # 数据
    root_dir: str = "data/CWRU"
    sample_length: int = 2400
    # 与 notebook 中示例一致：默认每个文件裁剪 100 段
    num_samples_per_file: int = 100
    channel: str = "DE"
    # 是否使用 cwru_samples_record.json 固定裁剪起点，保证多次运行完全一致
    use_record: bool = True

    # 训练（对齐 notebook：batch_size=64, epochs=500, lr_g=2e-4, lr_d=1e-4）
    batch_size: int = 64
    num_epochs: int = 500
    lr_g: float = 2e-4
    lr_d: float = 1e-4
    z_dim: int = 128
    beta1: float = 0.5
    beta2: float = 0.999
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # 物理约束
    lambda_phys: float = 0.75
    alpha_E: float = 1.0
    ratio_metric: str = "l1"
    ema_momentum: float = 0.1
    lambda_warmup_steps: int = 1000

    # 判别器 / 训练细节
    n_critic: int = 3
    d_noise_std: float = 0.02
    learnable_bands: bool = False

    # 特征提取 / VMD 参数
    vmd_params: Dict | None = None
    fs: int = 12000
    n_jobs: int = -1

    # 日志
    log_dir: str | None = None
    use_tensorboard: bool = True
    log_every: int = 50


def train_gan_with_physics(config: GANTrainConfig) -> None:
    """
    基于 `models.pcgan.PCG_Trainer` 的物理约束 GAN 训练主入口。

    流程：
        1) 加载 CWRU 信号片段
        2) 提取 31 维物理特征
        3) 计算 W_real (kg.weighting) 与类特征中心 v_real
        4) 计算每类 VMD 能量分布 E_real_rel
        5) 构建 PCG_Trainer 并执行训练
        6) 保存生成器 checkpoint 及先验信息
    """

    if config.vmd_params is None:
        # 与项目中其他位置使用的 VMD 默认参数保持一致
        config.vmd_params = dict(alpha=1000, tau=0, K=4, DC=0, init=1, tol=1e-7)

    device = torch.device(config.device)
    print(f"[设备] 使用设备: {device}")

    # ---------- 1. 加载信号 ----------
    signals_np, labels_np, label_names = load_cwru_signals(
        root_dir=config.root_dir,
        sample_length=config.sample_length,
        num_samples_per_file=config.num_samples_per_file,
        channel=config.channel,
        use_record=config.use_record,
    )
    print(f"[数据] 信号形状: {signals_np.shape}, 标签形状: {labels_np.shape}")
    print(f"[数据] 类别顺序 (CATEGORY_ORDER): {CATEGORY_ORDER}")

    num_classes = len(label_names)

    # ---------- 2. 提取 31 维特征 ----------
    features_np, scaler = batch_extract_features(
        signals_np,
        fs=config.fs,
        vmd_params=config.vmd_params,
        scaler=None,
        fit_scaler=True,
        n_jobs=config.n_jobs,
    )
    print(f"[特征] 特征矩阵形状: {features_np.shape} (预期 31 维)")

    # ---------- 3. 计算 v_real 与 W_real（严格对齐 notebook 中的 compute_feature_fault_weights） ----------
    v_real_np, sigma_np, W_real_np = compute_feature_fault_weights(
        features_np,
        labels_np,
        class_names=label_names,
        feature_names=FULL_FEATURE_NAMES,
    )
    print(f"[KG] v_real 形状: {v_real_np.shape}")
    print(f"[KG] W_real 形状: {W_real_np.shape}")
    print(f"[KG] 每行权重和 (应接近 1): {W_real_np.sum(axis=1)}")

    # ---------- 4. 构建 PCG_Trainer 并训练 ----------
    trainer = PCG_Trainer(
        X_signals=signals_np,
        y=labels_np,
        class_names=label_names,
        v_real=v_real_np,
        w_real=W_real_np,
        E_c=None,  # 与 notebook 中示例一致：由 PCG_Trainer 内部估计 E_c
        P=None,  # 如需在 D 中加入故障转移先验，可在此处接入 fault_evolution 模块
        batch_size=config.batch_size,
        z_dim=config.z_dim,
        lr_g=config.lr_g,
        lr_d=config.lr_d,
        lambda_phys=config.lambda_phys,
        alpha_E=config.alpha_E,
        device=str(device),
        fs=config.fs,
        n_critic=config.n_critic,
        ratio_metric=config.ratio_metric,
        ema_momentum=config.ema_momentum,
        learnable_bands=config.learnable_bands,
        lambda_warmup_steps=config.lambda_warmup_steps,
        log_dir=config.log_dir,
        use_tensorboard=config.use_tensorboard,
    )

    trainer.fit(
        epochs=config.num_epochs,
        log_every=config.log_every,
        d_noise_std=config.d_noise_std,
    )

    # ---------- 5. 训练结束后可视化（默认开启） ----------
    try:
        setup_chinese_font()

        # 5.1 t-SNE：重用训练数据和生成器
        print("[可视化] 开始生成 t-SNE 图像 (real vs fake)...")
        device = torch.device(config.device)
        G: CondGenerator1D = trainer.G
        gen_cfg: PCGANGeneratorConfig = trainer.gen_config
        G.eval()

        # 为每个类别生成与真实相同数量的样本
        C = num_classes
        X_real = signals_np
        y_real = labels_np

        # 为了画图，按每类最多 20 条采样，避免 t-SNE 过慢
        num_per_class_vis = min(20, config.num_samples_per_file * 4)
        real_indices: List[int] = []
        for c in range(C):
            idx_c = np.where(y_real == c)[0]
            if len(idx_c) == 0:
                continue
            if len(idx_c) > num_per_class_vis:
                idx_c = idx_c[:num_per_class_vis]
            real_indices.extend(idx_c.tolist())
        real_indices = np.array(real_indices, dtype=int)

        X_real_sub = X_real[real_indices]
        y_real_sub = y_real[real_indices]

        # 生成对应数量的 fake 样本
        cond_provider = ConditionProvider(
            class_names=label_names,
            w_real=W_real_np,
            E_c=trainer.E_c,
            P=None,
        )
        X_fake_list: List[np.ndarray] = []
        y_fake_list: List[np.ndarray] = []
        for cls_idx in range(C):
            z = torch.randn(num_per_class_vis, gen_cfg.z_dim, device=device)
            y_cls = torch.full(
                (num_per_class_vis,), cls_idx, dtype=torch.long, device=device
            )
            condG_np = cond_provider.get_cond_vectors_G(y_cls.cpu().numpy())
            condG = torch.tensor(condG_np, dtype=torch.float32, device=device)
            with torch.no_grad():
                xg = G(z, condG, y_cls).squeeze(1).cpu().numpy()
            X_fake_list.append(xg)
            y_fake_list.append(np.full(num_per_class_vis, cls_idx, dtype=int))

        X_fake = np.concatenate(X_fake_list, axis=0)
        y_fake = np.concatenate(y_fake_list, axis=0)

        # 段内 min-max，与 notebook 一致
        def _segment_minmax(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
            X = np.asarray(X, dtype=np.float32)
            max_abs = np.max(np.abs(X), axis=1, keepdims=True) + eps
            return X / max_abs

        X_real_scaled = _segment_minmax(X_real_sub)
        X_fake_scaled = _segment_minmax(X_fake)

        t_real, f_real, v_real, all_real = [], [], [], []
        t_fake, f_fake, v_fake, all_fake = [], [], [], []

        from features.freq_domain import extract_frequency_domain_features
        from features.time_domain import extract_time_domain_features
        from features.time_freq_vmd import extract_vmd_features

        for sig in X_real_scaled:
            sig = np.asarray(sig, dtype=np.float64)
            tfeats = extract_time_domain_features(sig)
            ffeats = extract_frequency_domain_features(sig, fs=config.fs)
            vfeats = extract_vmd_features(sig, **config.vmd_params)
            t_real.append(tfeats)
            f_real.append(ffeats)
            v_real.append(vfeats)
            all_real.append(tfeats + ffeats + vfeats)

        for sig in X_fake_scaled:
            sig = np.asarray(sig, dtype=np.float64)
            tfeats = extract_time_domain_features(sig)
            ffeats = extract_frequency_domain_features(sig, fs=config.fs)
            vfeats = extract_vmd_features(sig, **config.vmd_params)
            t_fake.append(tfeats)
            f_fake.append(ffeats)
            v_fake.append(vfeats)
            all_fake.append(tfeats + ffeats + vfeats)

        t_real = np.asarray(t_real, dtype=np.float32)
        f_real = np.asarray(f_real, dtype=np.float32)
        v_real = np.asarray(v_real, dtype=np.float32)
        all_real = np.asarray(all_real, dtype=np.float32)

        t_fake = np.asarray(t_fake, dtype=np.float32)
        f_fake = np.asarray(f_fake, dtype=np.float32)
        v_fake = np.asarray(v_fake, dtype=np.float32)
        all_fake = np.asarray(all_fake, dtype=np.float32)

        print(
            "[可视化] 特征维度检查:",
            "time",
            t_real.shape,
            t_fake.shape,
            "freq",
            f_real.shape,
            f_fake.shape,
            "vmd",
            v_real.shape,
            v_fake.shape,
            "all",
            all_real.shape,
            all_fake.shape,
        )

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        tsne_and_plot_ax(
            t_real,
            t_fake,
            y_real_sub,
            y_fake,
            "TIME (11-D)",
            axes[0, 0],
            label_names,
            num_per_class_vis,
        )
        tsne_and_plot_ax(
            f_real,
            f_fake,
            y_real_sub,
            y_fake,
            "FREQ (12-D)",
            axes[0, 1],
            label_names,
            num_per_class_vis,
        )
        tsne_and_plot_ax(
            v_real,
            v_fake,
            y_real_sub,
            y_fake,
            "VMD (8-D)",
            axes[1, 0],
            label_names,
            num_per_class_vis,
        )
        tsne_and_plot_ax(
            all_real,
            all_fake,
            y_real_sub,
            y_fake,
            "ALL (31-D)",
            axes[1, 1],
            label_names,
            num_per_class_vis,
        )

        plt.tight_layout()
        vis_dir = os.path.join("results", "gan_plots")
        os.makedirs(vis_dir, exist_ok=True)
        tsne_path = os.path.join(vis_dir, "tsne_features.png")
        plt.savefig(tsne_path, dpi=300, bbox_inches="tight")
        print(f"[保存] t-SNE 可视化图: {tsne_path}")
        plt.close(fig)

        # 5.2 单类波形 + VMD 能量对比：默认画 Inner Race（若存在）
        target_class_name = "Inner Race"
        if target_class_name in label_names:
            class_idx = label_names.index(target_class_name)
        else:
            class_idx = 0
            target_class_name = label_names[0]

        print(f"[可视化] 生成类别 {target_class_name} 的时域与 VMD 能量对比图")
        mask_real_cls = y_real_sub == class_idx
        mask_fake_cls = y_fake == class_idx
        real_signals_cls = X_real_sub[mask_real_cls]
        fake_signals_cls = X_fake[mask_fake_cls]

        # 重新用 batch_extract_features 得到 31 维特征，避免手写索引
        real_feats_cls, scaler_vis = batch_extract_features(
            real_signals_cls,
            fs=config.fs,
            vmd_params=config.vmd_params,
            scaler=None,
            fit_scaler=True,
            n_jobs=config.n_jobs,
        )
        fake_feats_cls, _ = batch_extract_features(
            fake_signals_cls,
            fs=config.fs,
            vmd_params=config.vmd_params,
            scaler=scaler_vis,
            fit_scaler=False,
            n_jobs=config.n_jobs,
        )

        from features.time_freq_vmd import VMD_FEATURE_NAMES

        energy_names = VMD_FEATURE_NAMES[:4]
        energy_indices: List[int] = []
        for en in energy_names:
            idx = FULL_FEATURE_NAMES.index(en)
            energy_indices.append(idx)

        waveform_path = os.path.join(
            vis_dir, f"{target_class_name.replace(' ', '_')}_waveform.png"
        )
        vmd_path = os.path.join(
            vis_dir, f"{target_class_name.replace(' ', '_')}_vmd_energy.png"
        )

        plot_time_series_compare(
            real_signals_cls, fake_signals_cls, target_class_name, waveform_path
        )
        plot_vmd_energy_compare(
            real_feats_cls, fake_feats_cls, energy_indices, target_class_name, vmd_path
        )

        print("[可视化] 训练结束可视化完成。")
    except Exception as e:
        print(f"[警告] 训练结束可视化失败，不影响 checkpoint 保存: {e}")

    # ---------- 6. 保存生成器及先验到 checkpoint ----------
    ckpt_dir = os.path.join("models", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "generator_step5_ckpt.pt")

    # E_c 由训练器内部基于 FourBandEnergy 估计，这里直接从 trainer 中读取
    E_real_rel_np = trainer.E_c

    ckpt = {
        "generator_state_dict": trainer.G.state_dict(),
        "config": trainer.gen_config.__dict__,
        "label_names": label_names,
        "W_real": W_real_np,
        "E_real_rel": E_real_rel_np,
    }
    torch.save(ckpt, ckpt_path)
    print(f"[保存] 生成器及先验信息已保存到: {ckpt_path}")


__all__ = ["GANTrainConfig", "train_gan_with_physics"]
