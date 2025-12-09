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


def evaluate_meta_on_48k(cfg: EvalMeta48KConfig) -> None:
    # 延迟导入，确保脚本级入口与项目结构解耦
    from data.cwru_loader import (
        get_48k_drive_end_file_mapping,
        load_cwru_signals,
    )
    from features.pipeline import batch_extract_features
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

    print(
        f"[Checkpoint] 加载 θ* 自 {cfg.ckpt_path} | "
        f"num_features={num_features}, num_classes={num_classes}"
    )

    model = KGMetaClassifier(
        num_features=num_features,
        num_classes=num_classes,
        W_real=W_real,
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
        s_x = features[s_idx]
        s_y = torch.tensor(s_y_np, dtype=torch.long, device=device)
        q_x = features[q_idx]
        q_y = torch.tensor(q_y_np, dtype=torch.long, device=device)

        # 从 θ* 初始化 fast weights（不改变全局模型参数）
        fast_weights = OrderedDict(
            (name, param) for name, param in model.named_parameters()
        )

        # 内循环：在 48k 支持集上做任务级适应
        for _ in range(cfg.inner_steps):
            logits_s, _ = model.forward_with_params(s_x, s_y, fast_weights)
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

        # 查询集评估：使用适应后的 fast_weights
        with torch.no_grad():
            logits_q, _ = model.forward_with_params(q_x, q_y, fast_weights)
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
