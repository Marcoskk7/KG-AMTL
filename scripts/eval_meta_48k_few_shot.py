from __future__ import annotations

"""
脚本：在 48k Drive End 目标域上对已训练好的 KG-AMTL 元学习模型进行 few-shot 评估。

整体流程：
    1) 从 checkpoint 加载在 12k 源域上训练好的 KGMetaClassifier (θ*)；
    2) 使用 data.cwru_loader.get_48k_drive_end_file_mapping 只加载 48k Drive End 故障数据；
    3) 提取 31 维物理特征（与源域一致的 pipeline）；
    4) 构造若干 N-way K-shot 任务，在每个任务上从 θ* 出发做内循环适应；
    5) 在查询集上评估分类准确率，统计平均 few-shot 性能。

注意：
    - 该脚本只做「目标域评估」，不会再更新元参数 θ*；
    - 默认不启用 MMD，仅关注 few-shot 适应精度；
    - 48k 数据仅包含故障类（Ball / Inner Race / Outer Race），不含 Normal Baseline。
"""

import argparse
import os
import sys
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# 确保项目根目录在 sys.path 中（允许在任意工作目录下直接运行该脚本）
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from models.knowledge_init import (
    compute_target_knowledge_vector,
    fuse_knowledge_prototypes,
)
from models.prototype_init import fuse_prototype_params


@dataclass
class EvalMeta48KConfig:
    # 数据与特征
    root_dir: str
    sample_length: int = 2400
    num_samples_per_file: int = 100
    channel: str = "DE"
    fs: int = 12000
    vmd_params: Dict | None = None
    n_jobs: int = -1

    # few-shot 任务设置（针对 48k 目标域）
    num_ways: int = 3  # 48k 目标域仅包含 3 个故障大类：IR/OR/B
    k_shot: int = 5
    q_query: int = 15
    inner_steps: int = 3
    inner_lr: float = 1e-3
    num_eval_tasks: int = 200

    # 知识感知初始化 / 原型加权（在每个 few-shot 任务上构造任务特定 W）
    use_knowledge_aware_init: bool = True
    ka_lambda: float = 1.0

    # 是否在评估阶段也使用原型级 KG 初始化（与训练阶段保持一致）
    use_prototype_init: bool = True

    # 模型与设备
    ckpt_path: str = os.path.join("models", "checkpoints", "kg_meta_classifier.pt")
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def _sample_task_indices(
    labels_np: np.ndarray,
    num_classes: int,
    num_ways: int,
    k_shot: int,
    q_query: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """
    从全数据中采样一个 N-way K-shot 任务的支持/查询索引。

    该实现与 models.meta_transfer._sample_task_indices 基本一致，
    在评估脚本中单独拷贝，避免引入不必要的依赖。
    """

    rng = np.random.default_rng()
    valid_classes: List[int] = []
    for c in range(num_classes):
        idx = np.where(labels_np == c)[0]
        if idx.size >= k_shot + q_query:
            valid_classes.append(c)
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


def _build_task_specific_W_for_eval(
    W_real: np.ndarray,
    support_features: np.ndarray,
    support_labels: np.ndarray,
    lambda_temp: float,
    device: torch.device,
) -> torch.Tensor:
    """
    针对 48k 目标域的单个 few-shot 任务，基于支持集构造任务特定的知识矩阵 W_task。

    实现思路与 models.meta_transfer._build_task_specific_W 一致：
        - 支持集每个类别 c：计算 w_c^{(t)}，再用 W_real 做原型加权得到 w_c^{(fused)}；
        - 将对应行替换为 w_c^{(fused)}，用于该任务内的动态特征加权。
    """

    W_task = W_real.copy()
    unique_labels = np.unique(support_labels)

    for c in unique_labels:
        c_int = int(c)
        mask = support_labels == c_int
        feats_c = support_features[mask]  # [N_c, D]
        if feats_c.shape[0] < 2:
            continue

        w_t = compute_target_knowledge_vector(feats_c)
        w_fused, _ = fuse_knowledge_prototypes(
            W_source=W_real,
            w_target=w_t,
            lambda_temp=lambda_temp,
        )
        if 0 <= c_int < W_task.shape[0]:
            W_task[c_int] = w_fused

    return torch.tensor(W_task, dtype=torch.float32, device=device)


def evaluate_meta_on_48k(cfg: EvalMeta48KConfig) -> None:
    # 延迟导入，确保脚本级入口与项目结构解耦
    from data.cwru_loader import (
        get_48k_drive_end_file_mapping,
        load_cwru_signals,
    )
    from features.pipeline import batch_extract_features
    from models.inception_time import InceptionTimeConfig
    from models.meta_transfer import KGMetaClassifier

    device = torch.device(cfg.device)
    print(f"[设备] 使用设备: {device}")

    # ---------- 1. 加载已训练好的 KG-AMTL 元模型 θ* ----------
    if not os.path.exists(cfg.ckpt_path):
        raise FileNotFoundError(
            f"未找到元学习模型 checkpoint: {cfg.ckpt_path}，"
            "请先运行 models.meta_transfer.train_meta_with_kg 完成训练。"
        )

    # PyTorch 2.6 起 torch.load 默认 weights_only=True，会阻止反序列化包含 numpy 对象的旧 checkpoint。
    # 本项目的 checkpoint 来自本地训练，可信，因此这里显式设置 weights_only=False 以保持兼容。
    ckpt = torch.load(cfg.ckpt_path, map_location=device, weights_only=False)
    num_features = ckpt["num_features"]
    num_classes = ckpt["num_classes"]
    W_real = ckpt["W_real"]
    scaler = ckpt.get("scaler", None)
    prototypes = ckpt.get("prototypes", None)
    backbone_info = ckpt.get("backbone", {})

    print(
        f"[Checkpoint] 加载 θ* 自 {cfg.ckpt_path} | "
        f"num_features={num_features}, num_classes={num_classes}"
    )

    backbone_cfg = InceptionTimeConfig(
        in_channels=1,
        num_blocks=backbone_info.get("inception_num_blocks", 3),
        out_channels=backbone_info.get("inception_out_channels", 32),
        bottleneck_channels=backbone_info.get("inception_bottleneck_channels", 32),
        kernel_sizes=backbone_info.get("inception_kernel_sizes", (41, 21, 11)),
        use_residual=backbone_info.get("inception_use_residual", True),
        dropout=backbone_info.get("inception_dropout", 0.1),
    )
    model = KGMetaClassifier(
        num_features=num_features,
        num_classes=num_classes,
        W_real=W_real,
        backbone_cfg=backbone_cfg,
        fusion_dropout=backbone_info.get("fusion_dropout", 0.1),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # ---------- 2. 加载 48k Drive End 目标域信号并提取特征 ----------
    if cfg.vmd_params is None:
        cfg.vmd_params = dict(alpha=1000, tau=0, K=4, DC=0, init=1, tol=1e-7)

    mapping_48k = get_48k_drive_end_file_mapping()
    signals_np, labels_np, label_names = load_cwru_signals(
        root_dir=cfg.root_dir,
        sample_length=cfg.sample_length,
        num_samples_per_file=cfg.num_samples_per_file,
        channel=cfg.channel,
        file_mapping=mapping_48k,
        use_record=True,
    )
    print(
        f"[数据-48k] 信号形状: {signals_np.shape}, 标签形状: {labels_np.shape}, "
        f"label_names={label_names}"
    )

    # 使用与源域相同的 MinMaxScaler 做特征缩放，保持分布一致
    features_np, _ = batch_extract_features(
        signals_np,
        fs=cfg.fs,
        vmd_params=cfg.vmd_params,
        scaler=scaler,
        fit_scaler=False if scaler is not None else True,
        n_jobs=cfg.n_jobs,
    )
    print(f"[特征-48k] 特征矩阵形状: {features_np.shape} (预期 31 维)")

    signals = torch.tensor(signals_np, dtype=torch.float32, device=device)
    features = torch.tensor(features_np, dtype=torch.float32, device=device)

    # ---------- 3. few-shot 评估：在 48k 目标域上构造多任务 ----------
    print(
        f"[评估配置] num_ways={cfg.num_ways}, k_shot={cfg.k_shot}, "
        f"q_query={cfg.q_query}, inner_steps={cfg.inner_steps}, "
        f"inner_lr={cfg.inner_lr}, num_tasks={cfg.num_eval_tasks}"
    )

    acc_list: List[float] = []

    for task_idx in range(cfg.num_eval_tasks):
        sampled = _sample_task_indices(
            labels_np,
            num_classes=num_classes,
            num_ways=cfg.num_ways,
            k_shot=cfg.k_shot,
            q_query=cfg.q_query,
        )
        if sampled is None:
            print(
                "[警告] 目标域数据不足以构成指定的 N-way K-shot 任务，"
                "请检查 num_ways / k_shot / q_query 设置。"
            )
            break

        s_idx, s_y_np, q_idx, q_y_np = sampled
        s_signal = signals[s_idx]
        s_feat = features[s_idx]
        s_y = torch.tensor(s_y_np, dtype=torch.long, device=device)
        q_signal = signals[q_idx]
        q_feat = features[q_idx]
        q_y = torch.tensor(q_y_np, dtype=torch.long, device=device)

        # 针对当前 few-shot 任务构造任务特定知识矩阵（可选）
        W_task_tensor: torch.Tensor | None = None
        if cfg.use_knowledge_aware_init:
            s_x_np = features_np[s_idx]
            W_task_tensor = _build_task_specific_W_for_eval(
                W_real=W_real,
                support_features=s_x_np,
                support_labels=s_y_np,
                lambda_temp=cfg.ka_lambda,
                device=device,
            )

        # 从 θ* 或原型融合初始化 fast weights（不改变全局模型参数）
        if prototypes is not None and cfg.use_prototype_init:
            # 使用支持集构造目标任务知识向量，并基于原型融合 θ0
            s_x_np = features_np[s_idx]
            w_target = compute_target_knowledge_vector(s_x_np)
            theta_0_state = fuse_prototype_params(
                prototypes=prototypes,
                W_source=W_real,
                w_target=w_target,
                lambda_temp=cfg.ka_lambda,
            )
            # 注意：state_dict 中的 tensor 默认 requires_grad=False，
            # 需要显式开启以支持内循环的 autograd.grad
            param_names = {name for name, _ in model.named_parameters()}
            fast_weights = OrderedDict(
                (name, param.to(device).requires_grad_(True))
                for name, param in theta_0_state.items()
                if name in param_names
            )
        else:
            fast_weights = OrderedDict(
                (name, param) for name, param in model.named_parameters()
            )

        # 内循环：在 48k 支持集上做任务级适应
        for _ in range(cfg.inner_steps):
            logits_s, _ = model.forward_with_params(
                s_signal, s_feat, s_y, fast_weights, W_override=W_task_tensor
            )
            loss_task = F.cross_entropy(logits_s, s_y)

            grads = torch.autograd.grad(
                loss_task,
                list(fast_weights.values()),
                create_graph=False,
            )

            fast_weights = OrderedDict(
                (
                    name,
                    param - cfg.inner_lr * g,
                )
                for (name, param), g in zip(fast_weights.items(), grads)
            )

        # 查询集评估：使用适应后的 fast_weights 与任务特定知识矩阵
        with torch.no_grad():
            logits_q, _ = model.forward_with_params(
                q_signal, q_feat, q_y, fast_weights, W_override=W_task_tensor
            )
            preds_q = logits_q.argmax(dim=1)
            acc = (preds_q == q_y).float().mean().item()
            acc_list.append(acc)

    if not acc_list:
        print("[结果] 未能评估任何有效任务，无法给出 few-shot 精度。")
        return

    acc_arr = np.asarray(acc_list, dtype=float)
    print(
        f"[结果] 在 48k Drive End 目标域上的 few-shot 评估：\n"
        f"    任务数      : {len(acc_arr)}\n"
        f"    平均精度    : {acc_arr.mean():.4f}\n"
        f"    标准差      : {acc_arr.std():.4f}\n"
        f"    最低/最高精度: {acc_arr.min():.4f} / {acc_arr.max():.4f}"
    )


def main() -> None:
    # 确保项目根目录在 sys.path 中
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    if project_root not in sys.path:
        sys.path.append(project_root)

    parser = argparse.ArgumentParser(
        description="Evaluate KG-AMTL meta-initialization on 48k Drive End (few-shot)."
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default=None,
        help="CWRU 数据根目录；若不指定，则默认使用 <project_root>/data/CWRU。",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="KG-AMTL 元学习模型 checkpoint 路径（默认 models/checkpoints/kg_meta_classifier.pt）。",
    )
    parser.add_argument(
        "--num_tasks",
        type=int,
        default=200,
        help="few-shot 评估的任务数量。",
    )
    args = parser.parse_args()

    if args.root_dir is None:
        root_dir = os.path.join(project_root, "data", "CWRU")
    else:
        root_dir = args.root_dir

    if args.ckpt is None:
        ckpt_path = os.path.join(
            project_root, "models", "checkpoints", "kg_meta_classifier.pt"
        )
    else:
        ckpt_path = args.ckpt

    print(f"[配置] 目标域 root_dir: {root_dir}")
    print(f"[配置] 使用元模型 checkpoint: {ckpt_path}")

    cfg = EvalMeta48KConfig(root_dir=root_dir, ckpt_path=ckpt_path)
    cfg.num_eval_tasks = args.num_tasks

    evaluate_meta_on_48k(cfg)


if __name__ == "__main__":
    main()
