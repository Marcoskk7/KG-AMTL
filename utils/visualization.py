from __future__ import annotations

import os
from typing import List

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from features.time_freq_vmd import VMD_FEATURE_NAMES


def _limit_samples_per_class(
    feats: np.ndarray,
    labels: np.ndarray,
    limit: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """每个类别最多保留 ``limit`` 条样本，用于加速可视化。"""

    if limit is None or limit <= 0:
        return feats, labels

    feats = np.asarray(feats)
    labels = np.asarray(labels)

    keep_indices: list[np.ndarray] = []
    for cls in np.unique(labels):
        cls_idx = np.where(labels == cls)[0]
        if cls_idx.size > limit:
            cls_idx = cls_idx[:limit]
        keep_indices.append(cls_idx)

    if not keep_indices:
        return feats[:0], labels[:0]

    gathered = np.concatenate(keep_indices)
    return feats[gathered], labels[gathered]


def setup_chinese_font() -> str | None:
    """设置 matplotlib 的中文字体，优先尝试常见的中文字体。"""

    chinese_fonts = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "KaiTi",
        "FangSong",
        "STSong",
        "Arial Unicode MS",
    ]

    available_fonts = [f.name for f in fm.fontManager.ttflist]

    for font_name in chinese_fonts:
        if font_name in available_fonts:
            plt.rcParams["font.sans-serif"] = [font_name]
            plt.rcParams["axes.unicode_minus"] = False
            print(f"[字体] 使用中文字体: {font_name}")
            return font_name

    print("[警告] 未找到中文字体，图表中的中文可能显示为方框")
    return None


def plot_time_series_compare(
    real_signals: np.ndarray,
    fake_signals: np.ndarray,
    class_name: str,
    save_path: str | None = None,
) -> None:
    """绘制几条真实 vs 生成的时域波形对比。"""

    num_to_plot = min(3, real_signals.shape[0], fake_signals.shape[0])
    if num_to_plot == 0:
        print("[警告] plot_time_series_compare: 输入样本数为 0，跳过绘图")
        return

    t = np.arange(real_signals.shape[1])

    fig, axes = plt.subplots(num_to_plot, 2, figsize=(10, 6), sharex=True)
    if num_to_plot == 1:
        axes = np.array([axes])

    for i in range(num_to_plot):
        axes[i, 0].plot(t, real_signals[i], color="tab:blue")
        axes[i, 0].set_title(f"真实样本 {i + 1}")

        axes[i, 1].plot(t, fake_signals[i], color="tab:orange")
        axes[i, 1].set_title(f"生成样本 {i + 1}")

    fig.suptitle(f"类别 {class_name} 的时域波形对比")
    for ax in axes[-1, :]:
        ax.set_xlabel("采样点")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        abs_path = os.path.abspath(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[保存] 时域对比图: {save_path}")
        print(f"      -> 绝对路径: {abs_path}")
    else:
        plt.show()
    plt.close(fig)


def compute_vmd_energy_relative(
    feats: np.ndarray, energy_indices: List[int]
) -> np.ndarray:
    """从 31 维特征中提取指定的 4 个 VMD 能量，并做相对归一化。"""

    if feats.size == 0:
        raise ValueError("compute_vmd_energy_relative: 输入特征为空")

    E = feats[:, energy_indices].mean(axis=0)
    E_rel = E / (np.linalg.norm(E) + 1e-8)
    return E_rel


def plot_vmd_energy_compare(
    real_feats: np.ndarray,
    fake_feats: np.ndarray,
    energy_indices: List[int],
    class_name: str,
    save_path: str | None = None,
) -> None:
    """绘制真实 vs 生成样本的 VMD 能量分布对比（条形图）。"""

    if real_feats.size == 0 or fake_feats.size == 0:
        print("[警告] plot_vmd_energy_compare: 输入特征为空，跳过绘图")
        return

    E_real_rel = compute_vmd_energy_relative(real_feats, energy_indices)
    E_fake_rel = compute_vmd_energy_relative(fake_feats, energy_indices)

    x = np.arange(4)
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - width / 2, E_real_rel, width, label="真实", color="tab:blue")
    ax.bar(x + width / 2, E_fake_rel, width, label="生成", color="tab:orange")

    ax.set_xticks(x)
    ax.set_xticklabels(VMD_FEATURE_NAMES[:4], rotation=30)
    ax.set_ylabel("相对能量")
    ax.set_title(f"类别 {class_name} 的 VMD 能量分布对比")
    ax.legend()

    plt.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        abs_path = os.path.abspath(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[保存] VMD 能量分布对比图: {save_path}")
        print(f"      -> 绝对路径: {abs_path}")
    else:
        plt.show()
    plt.close(fig)


def tsne_and_plot_ax(
    feat_real: np.ndarray,
    feat_fake: np.ndarray,
    y_real: np.ndarray,
    y_fake: np.ndarray,
    title: str,
    ax,
    class_names: List[str],
    num_per_class: int | None = None,
) -> None:
    """
    在 ax（子图）上绘制 t-SNE 结果。
    feat_real, feat_fake: (N_real, D), (N_fake, D)
    y_real, y_fake       : 类别标签
    """
    if feat_real.size == 0 or feat_fake.size == 0:
        print(f"[警告] tsne_and_plot_ax({title}): 输入特征为空，跳过绘图")
        return

    feat_real, y_real = _limit_samples_per_class(feat_real, y_real, num_per_class)
    feat_fake, y_fake = _limit_samples_per_class(feat_fake, y_fake, num_per_class)

    if feat_real.size == 0 or feat_fake.size == 0:
        print(f"[警告] tsne_and_plot_ax({title}): 裁剪后样本为空，跳过绘图")
        return

    if num_per_class is not None and num_per_class > 0:
        print(
            f"[采样] tsne_and_plot_ax({title}): 每类最多 {num_per_class} 条 real/fake 样本参与 t-SNE"
        )

    X_feat_all = np.vstack([feat_real, feat_fake])
    domain_labels = np.concatenate(
        [
            np.zeros(len(feat_real), dtype=int),
            np.ones(len(feat_fake), dtype=int),
        ]
    )
    class_labels = np.concatenate([y_real, y_fake])

    scaler = StandardScaler()
    X_feat_all_std = scaler.fit_transform(X_feat_all)

    print(f"[t-SNE] {title} 正在降维到 2 维用于可视化...")
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate=200.0,
        init="random",
        random_state=42,
        verbose=1,
    )
    X_emb = tsne.fit_transform(X_feat_all_std)
    print(f"[t-SNE] {title} 完成:", X_emb.shape)

    C = len(class_names)
    cmap = cm.get_cmap("tab10")
    colors = cmap(np.linspace(0, 1, C))

    for c in range(C):
        mask_real_c = (class_labels == c) & (domain_labels == 0)
        mask_fake_c = (class_labels == c) & (domain_labels == 1)

        ax.scatter(
            X_emb[mask_real_c, 0],
            X_emb[mask_real_c, 1],
            s=18,
            c=colors[c].reshape(1, -1),
            alpha=0.8,
            marker="o",
            edgecolors="none",
        )
        ax.scatter(
            X_emb[mask_fake_c, 0],
            X_emb[mask_fake_c, 1],
            s=36,
            facecolors="none",
            edgecolors=colors[c],
            alpha=0.9,
            marker="o",
            linewidths=1.0,
        )

    real_proxy = Line2D(
        [0],
        [0],
        marker="o",
        linestyle="none",
        markersize=6,
        markerfacecolor="k",
        markeredgecolor="k",
        label="Real (实心点)",
    )
    fake_proxy = Line2D(
        [0],
        [0],
        marker="o",
        linestyle="none",
        markersize=8,
        markerfacecolor="none",
        markeredgecolor="k",
        label="Fake (空心点)",
    )
    legend_domain = ax.legend(
        handles=[real_proxy, fake_proxy],
        loc="upper right",
        title="Domain",
        fontsize=8,
        title_fontsize=9,
    )
    ax.add_artist(legend_domain)

    class_handles = []
    for c in range(C):
        class_name = class_names[c] if c < len(class_names) else f"class {c}"
        h = Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=6,
            markerfacecolor=colors[c],
            markeredgecolor=colors[c],
            label=class_name,
        )
        class_handles.append(h)
    ax.legend(
        handles=class_handles,
        loc="lower right",
        title="Fault Class",
        fontsize=8,
        title_fontsize=9,
    )

    ax.set_title(title)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.grid(True, alpha=0.3)


__all__ = [
    "setup_chinese_font",
    "plot_time_series_compare",
    "compute_vmd_energy_relative",
    "plot_vmd_energy_compare",
    "tsne_and_plot_ax",
]
