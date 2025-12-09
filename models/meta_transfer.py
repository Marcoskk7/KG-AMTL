from __future__ import annotations

"""
Knowledge-Guided Meta-Transfer Learning (KG-AMTL) 训练模块。

对应 `SRP Guidance.pdf` 与 `reference/markdown/3.md` 中第 3 部分：
    - 3.1 Knowledge-Aware Initialization
    - 3.2 Feature Adaptation Layer
    - 3.3 Meta-Learning Update Mechanism

本实现目前聚焦于「带知识图谱先验的元学习训练」，整体流程为：
    1) 复用 `data.cwru_loader.load_cwru_signals` 加载 CWRU 信号；
    2) 使用 `features.pipeline.batch_extract_features` 提取 31 维物理特征；
    3) 使用 `models.gan_training.compute_feature_fault_weights` 或已有 KG 文件
       获取特征-故障相关矩阵 W_real；
    4) 构建包含「动态特征加权层」的分类网络；
    5) 按论文 3.3 / 6.3 节描述的 **二阶 MAML** 元学习循环，基于 N-way K-shot 任务
       训练一个具有快速适应能力的初始化参数 θ*；
    6) 将训练好的模型与 KG 先验一并保存到 checkpoint。

注意：
    - 当前实现为完整的二阶 MAML：在内循环更新 θ_i′ 时保留计算图，并在外循环
      对查询集元损失反向传播到初始参数 θ，从而显式包含二阶梯度项。
"""

import os
from collections import OrderedDict
from dataclasses import asdict, dataclass
from typing import Dict, List, Mapping, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.cwru_loader import load_cwru_signals
from features.pipeline import FULL_FEATURE_NAMES, batch_extract_features
from models.feature_adaptation import (
    DynamicFeatureWeighter,
    DynamicFeatureWeighterConfig,
)
from models.gan_training import compute_feature_fault_weights


@dataclass
class MetaTransferConfig:
    """
    KG 引导元迁移学习（KG-AMTL）训练配置。

    主要分为三部分：
        - 数据 / 特征提取相关参数；
        - 元任务 (N-way K-shot) 采样与内外循环超参数；
        - 日志与 checkpoint 保存路径。
    """

    # 数据
    root_dir: str = "data/CWRU"
    sample_length: int = 2400
    num_samples_per_file: int = 100
    channel: str = "DE"
    use_record: bool = True

    # 特征提取 / VMD 参数
    vmd_params: Dict | None = None
    fs: int = 12000
    n_jobs: int = -1

    # 元学习任务设置
    num_ways: int = 4  # N-way
    k_shot: int = 5  # 每类支持集样本数
    q_query: int = 15  # 每类查询集样本数
    inner_steps: int = 1  # 内循环梯度步数（MAML 内循环）
    inner_lr: float = 1e-2  # 内循环学习率（与最初版本保持一致）
    meta_lr: float = 1e-3  # 外循环（元更新）学习率
    num_epochs: int = 200
    tasks_per_epoch: int = 128

    # 是否启用 MMD 分布对齐
    use_mmd: bool = False
    mmd_lambda: float = 0.1
    mmd_gamma: float = 1.0

    # 目标域无标签数据根目录；若为 None，则退化为使用源域特征做自对齐
    target_root_dir: str | None = None

    # 设备与日志
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_every: int = 10
    log_dir: str | None = None

    # 模型保存路径
    ckpt_path: str = os.path.join("models", "checkpoints", "kg_meta_classifier.pt")


class KGMetaClassifier(nn.Module):
    """
    使用知识图谱相关矩阵 W_real 的元学习分类网络。

    结构：
        输入：31 维物理特征 x_feat (N×D)
        1) DynamicFeatureWeighter:  f_w = x_feat ⊙ σ(W[i])
        2) 两层 MLP 作为 backbone 提取高维表示 h
        3) 线性分类头输出最终 logits
    """

    def __init__(self, num_features: int, num_classes: int, W_real: np.ndarray):
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes

        # Step 6: 动态特征加权层（使用 KG 中的 W_real）
        dfw_cfg = DynamicFeatureWeighterConfig(
            num_classes=num_classes,
            feature_dim=num_features,
            apply_sigmoid_to_w=True,
        )
        self.weighter = DynamicFeatureWeighter(dfw_cfg)
        W_tensor = torch.tensor(W_real, dtype=torch.float32)
        self.weighter.register_knowledge_weights(W_tensor)

        # 简单的两层 MLP 作为特征提取 backbone
        hidden_dim = 64
        self.backbone = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(
        self, x: torch.Tensor, y: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : [B, D]
            物理特征向量。
        y : Optional[Tensor]
            故障类别索引 [B]；若提供，则用于知识引导的动态加权。

        Returns
        -------
        logits : [B, C]
        feats  : [B, H]  # backbone 输出特征，可用于 MMD 等正则。
        """
        if y is not None:
            x = self.weighter(x, y)  # [B, D]
        h = self.backbone(x)
        logits = self.classifier(h)
        return logits, h

    def forward_with_params(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None,
        params: Mapping[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        使用给定的参数字典进行前向传播，用于 MAML 内循环的功能式前向。

        Parameters
        ----------
        x : [B, D]
            输入特征（31 维物理特征）。
        y : Optional[Tensor]
            故障类别索引；若为 None，则跳过知识加权层（用于无标签目标域）。
        params : Dict[str, Tensor]
            由 `OrderedDict(model.named_parameters())` 得到并经过若干步
            梯度更新后的参数集合。
        """

        if y is not None:
            x = self.weighter(x, y)  # [B, D]

        # backbone 的两层全连接（对应 nn.Sequential 的 0 和 2 层）
        w1 = params["backbone.0.weight"]
        b1 = params["backbone.0.bias"]
        w2 = params["backbone.2.weight"]
        b2 = params["backbone.2.bias"]

        h = F.linear(x, w1, b1)
        h = F.relu(h, inplace=False)
        h = F.linear(h, w2, b2)
        h = F.relu(h, inplace=False)

        # 分类头
        w_cls = params["classifier.weight"]
        b_cls = params["classifier.bias"]
        logits = F.linear(h, w_cls, b_cls)
        return logits, h


def _build_or_load_W_real(
    features_np: np.ndarray,
    labels_np: np.ndarray,
    label_names: List[str],
) -> np.ndarray:
    """
    构建或加载特征-故障相关矩阵 W_real。

    优先从 `knowledge_graphs/kg_step2_w_v_sigma.npz` 读取，若不存在则
    调用 `compute_feature_fault_weights` 重新计算。
    """

    kg_dir = "knowledge_graphs"
    kg_path = os.path.join(kg_dir, "kg_step2_w_v_sigma.npz")
    if os.path.exists(kg_path):
        data = np.load(kg_path, allow_pickle=True)
        W = data["w"]  # (C, D)
        class_names = list(data["class_names"].tolist())
        if len(class_names) != len(label_names):
            raise RuntimeError(
                "kg_step2_w_v_sigma.npz 中的 class_names 与当前 label_names 数量不一致"
            )
        # 若类别顺序不同，则按 label_names 重排
        reorder_indices = [class_names.index(name) for name in label_names]
        W = W[reorder_indices]
        return W

    # 若 KG 文件不存在，则直接重算一遍（与 GAN 训练保持一致）
    _, _, W = compute_feature_fault_weights(
        features_np,
        labels_np,
        class_names=label_names,
        feature_names=FULL_FEATURE_NAMES,
    )
    return W


def _sample_task_indices(
    labels_np: np.ndarray,
    num_classes: int,
    num_ways: int,
    k_shot: int,
    q_query: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """
    从全数据中采样一个 N-way K-shot 任务的支持/查询索引。

    返回：
        support_idx, support_y, query_idx, query_y
    若样本不足以构成任务，返回 None。
    """

    rng = np.random.default_rng()
    class_indices: List[np.ndarray] = []
    valid_classes: List[int] = []
    for c in range(num_classes):
        idx = np.where(labels_np == c)[0]
        if idx.size >= k_shot + q_query:
            valid_classes.append(c)
            class_indices.append(idx)
    if len(valid_classes) < num_ways:
        return None

    chosen = rng.choice(valid_classes, size=num_ways, replace=False)
    support_indices: List[int] = []
    query_indices: List[int] = []
    support_labels: List[int] = []
    query_labels: List[int] = []

    for c in chosen:
        idx_all = np.where(labels_np == c)[0]
        perm = rng.permutation(idx_all)
        s_idx = perm[:k_shot]
        q_idx = perm[k_shot : k_shot + q_query]
        support_indices.extend(s_idx.tolist())
        query_indices.extend(q_idx.tolist())
        support_labels.extend([c] * k_shot)
        query_labels.extend([c] * q_query)

    return (
        np.asarray(support_indices, dtype=int),
        np.asarray(support_labels, dtype=int),
        np.asarray(query_indices, dtype=int),
        np.asarray(query_labels, dtype=int),
    )


def _compute_mmd(
    feats_p: torch.Tensor, feats_q: torch.Tensor, gamma: float = 1.0
) -> torch.Tensor:
    """
    简单的 RBF 核 MMD 计算，用于分布对齐正则（可选）。
    """
    if feats_p.size(0) == 0 or feats_q.size(0) == 0:
        return feats_p.new_zeros(())

    xx = torch.mm(feats_p, feats_p.t())
    yy = torch.mm(feats_q, feats_q.t())
    xy = torch.mm(feats_p, feats_q.t())

    rx = xx.diag().unsqueeze(0)
    ry = yy.diag().unsqueeze(0)

    Kxx = torch.exp(-gamma * (rx.t() + rx - 2 * xx))
    Kyy = torch.exp(-gamma * (ry.t() + ry - 2 * yy))
    Kxy = torch.exp(-gamma * (rx.t() + ry - 2 * xy))

    mmd = Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()
    return mmd


def train_meta_with_kg(config: MetaTransferConfig) -> None:
    """
    KG 引导元迁移学习主入口（论文要求的二阶 MAML 实现）。

    流程：
        1) 加载 CWRU 信号并提取 31 维特征；
        2) 通过 KG 计算 / 加载特征-故障权重矩阵 W_real；
        3) 构建 KGMetaClassifier；
        4) 基于 N-way K-shot 任务，按 MAML 进行内外循环更新（包含二阶梯度）；
        5) 保存模型与先验到 checkpoint。
    """

    if config.vmd_params is None:
        config.vmd_params = dict(alpha=1000, tau=0, K=4, DC=0, init=1, tol=1e-7)

    device = torch.device(config.device)
    print(f"[设备] 使用设备: {device}")

    # ---------- 1. 加载源域信号 ----------
    signals_np, labels_np, label_names = load_cwru_signals(
        root_dir=config.root_dir,
        sample_length=config.sample_length,
        num_samples_per_file=config.num_samples_per_file,
        channel=config.channel,
        use_record=config.use_record,
    )
    print(f"[数据] 信号形状: {signals_np.shape}, 标签形状: {labels_np.shape}")

    num_classes = len(label_names)

    # ---------- 2. 提取源域 31 维特征 ----------
    features_np, scaler = batch_extract_features(
        signals_np,
        fs=config.fs,
        vmd_params=config.vmd_params,
        scaler=None,
        fit_scaler=True,
        n_jobs=config.n_jobs,
    )
    print(f"[特征] 特征矩阵形状: {features_np.shape} (预期 31 维)")

    # ---------- 2.1 提取目标域无标签特征（用于 MMD 对齐） ----------
    target_features_np = features_np
    if config.use_mmd:
        if config.target_root_dir is None:
            print(
                "[警告] MetaTransferConfig.target_root_dir 未设置，"
                "MMD 将在源域内部自对齐（近似目标域）。"
            )
        else:
            print(f"[目标域] 从 {config.target_root_dir} 加载无标签信号用于 MMD 对齐")
            tgt_signals_np, _, _ = load_cwru_signals(
                root_dir=config.target_root_dir,
                sample_length=config.sample_length,
                num_samples_per_file=config.num_samples_per_file,
                channel=config.channel,
                use_record=config.use_record,
            )
            target_features_np, _ = batch_extract_features(
                tgt_signals_np,
                fs=config.fs,
                vmd_params=config.vmd_params,
                scaler=scaler,
                fit_scaler=False,
                n_jobs=config.n_jobs,
            )
            print(
                f"[目标域] 无标签特征矩阵形状: {target_features_np.shape} "
                "(用于 MMD 分布对齐)"
            )

    # ---------- 3. 加载 / 计算 W_real ----------
    W_real = _build_or_load_W_real(features_np, labels_np, label_names)
    print(f"[KG] W_real 形状: {W_real.shape}")

    # ---------- 4. 构建 KGMetaClassifier（元初始化参数 θ） ----------
    num_features = features_np.shape[1]
    model = KGMetaClassifier(
        num_features=num_features, num_classes=num_classes, W_real=W_real
    ).to(device)

    # 将特征缓存到 GPU 以减少数据搬运
    features = torch.tensor(features_np, dtype=torch.float32, device=device)
    target_features = torch.tensor(
        target_features_np, dtype=torch.float32, device=device
    )

    # 二阶 MAML 外循环：基于查询集元损失的梯度更新 θ
    meta_optimizer = torch.optim.Adam(model.parameters(), lr=config.meta_lr)

    for epoch in range(1, config.num_epochs + 1):
        model.train()
        meta_optimizer.zero_grad()
        task_count = 0
        acc_sum = 0.0
        acc_count = 0

        for _ in range(config.tasks_per_epoch):
            sampled = _sample_task_indices(
                labels_np,
                num_classes=num_classes,
                num_ways=config.num_ways,
                k_shot=config.k_shot,
                q_query=config.q_query,
            )
            if sampled is None:
                continue

            (
                s_idx,
                s_y_np,
                q_idx,
                q_y_np,
            ) = sampled

            s_x = features[s_idx]  # [B_s, D]
            s_y = torch.tensor(s_y_np, dtype=torch.long, device=device)
            q_x = features[q_idx]
            q_y = torch.tensor(q_y_np, dtype=torch.long, device=device)

            # ---------- 内循环：在支持集上执行若干步梯度更新（任务级适应） ----------
            # 使用功能式前向 + autograd.grad 实现 MAML，可对初始参数 θ 求二阶梯度
            fast_weights = OrderedDict(
                (name, param) for name, param in model.named_parameters()
            )

            for _step in range(config.inner_steps):
                logits_s, _ = model.forward_with_params(s_x, s_y, fast_weights)
                loss_task = F.cross_entropy(logits_s, s_y)

                grads = torch.autograd.grad(
                    loss_task,
                    list(fast_weights.values()),
                    create_graph=True,
                )

                fast_weights = OrderedDict(
                    (
                        name,
                        param - config.inner_lr * g,
                    )
                    for (name, param), g in zip(fast_weights.items(), grads)
                )

            # ---------- 外循环：在查询集上计算元损失（含可选 MMD），并对 θ 反向传播 ----------
            logits_q, feats_q = model.forward_with_params(q_x, q_y, fast_weights)
            loss_meta = F.cross_entropy(logits_q, q_y)

            # 可选：在查询集与目标域无标签特征之间加入 MMD 对齐项（与论文元目标一致）
            if config.use_mmd and target_features.size(0) > 0:
                t_batch_size = min(feats_q.size(0), target_features.size(0))
                idx_t = torch.randint(
                    0,
                    target_features.size(0),
                    (t_batch_size,),
                    device=device,
                )
                t_x = target_features[idx_t]
                # 目标域无标签样本只经过特征提取器 φ(·; θ_i′)
                _, feats_t = model.forward_with_params(t_x, None, fast_weights)
                loss_mmd = _compute_mmd(
                    feats_q[:t_batch_size],
                    feats_t,
                    gamma=config.mmd_gamma,
                )
                loss_meta = loss_meta + config.mmd_lambda * loss_mmd

            # 记录查询集精度（仅用于日志）
            with torch.no_grad():
                preds_q = logits_q.argmax(dim=1)
                acc = (preds_q == q_y).float().mean().item()
                acc_sum += acc
                acc_count += 1

            # 二阶 MAML：loss_meta 对 θ 的梯度中自然包含内循环更新的二阶项
            loss_meta.backward()
            task_count += 1

        if task_count == 0:
            print(
                "[警告] 本 epoch 未能采样到任何有效任务，请检查 k_shot/q_query 设置或数据量。"
            )
            continue

        # 对累计的元梯度做平均，并更新元参数 θ
        for p in model.parameters():
            if p.grad is not None:
                p.grad /= task_count
        meta_optimizer.step()

        if epoch % config.log_every == 0 or epoch == 1 or epoch == config.num_epochs:
            avg_acc = acc_sum / max(acc_count, 1)
            print(
                f"[Meta] epoch {epoch:03d} | tasks={task_count} | "
                f"avg query acc={avg_acc:.4f}"
            )

    # ---------- 5. 保存元学习后的初始化参数 θ* 与先验 ----------
    os.makedirs(os.path.dirname(config.ckpt_path), exist_ok=True)
    ckpt = {
        "model_state_dict": model.state_dict(),
        "config": asdict(config),
        "label_names": label_names,
        "W_real": W_real,
        "num_features": num_features,
        "num_classes": num_classes,
        # 保存在源域特征上拟合的 MinMaxScaler，便于目标域特征使用相同缩放规则
        "scaler": scaler,
    }
    torch.save(ckpt, config.ckpt_path)
    print(f"[保存] KG-AMTL 元学习模型已保存到: {config.ckpt_path}")


__all__ = ["MetaTransferConfig", "KGMetaClassifier", "train_meta_with_kg"]
