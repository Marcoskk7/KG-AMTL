"""
辅助小脚本：使用已经训练好的生成器 G，针对某一故障类别生成样本，
并与真实样本在「时域波形 + VMD 能量分布」上做对比可视化。

使用方法（在项目根目录执行）：

    python -m tests.gan_sample_plot \\
        --ckpt models/checkpoints/generator_step5_ckpt.pt \\
        --class_name "Inner Race" \\
        --num_samples 5

该脚本依赖：
    - 你已运行过 `python -m models.gan_training`，生成了默认权重文件；
    - CWRU 数据集位于 data/CRWU；
    - requirements.txt 中的 vmdpy, matplotlib 等依赖已安装。
"""

from __future__ import annotations

import argparse
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import torch

from data.cwru_loader import CATEGORY_ORDER, load_cwru_signals
from features.pipeline import batch_extract_features
from features.time_freq_vmd import VMD_FEATURE_NAMES
from models.generator import AttributeConditionedGenerator, GeneratorConfig


def _setup_chinese_font():
    """设置 matplotlib 的中文字体，优先尝试常见的中文字体。"""
    # 常见中文字体列表（按优先级）
    chinese_fonts = [
        "Microsoft YaHei",  # Windows 常见
        "SimHei",  # Windows 黑体
        "SimSun",  # Windows 宋体
        "KaiTi",  # Windows 楷体
        "FangSong",  # Windows 仿宋
        "STSong",  # macOS
        "Arial Unicode MS",  # macOS
    ]

    # 获取系统所有可用字体
    available_fonts = [f.name for f in fm.fontManager.ttflist]

    # 查找第一个可用的中文字体
    for font_name in chinese_fonts:
        if font_name in available_fonts:
            plt.rcParams["font.sans-serif"] = [font_name]
            plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
            print(f"[字体] 使用中文字体: {font_name}")
            return font_name

    # 如果都没找到，使用默认字体（会显示警告，但不影响运行）
    print("[警告] 未找到中文字体，图表中的中文可能显示为方框")
    return None


def _find_vmd_energy_indices(full_feature_names: List[str]) -> List[int]:
    """与 models.gan_training 中逻辑保持一致：找到 4 个 VMD 能量特征的索引。"""

    from features.pipeline import FULL_FEATURE_NAMES

    energy_names = VMD_FEATURE_NAMES[:4]
    indices: List[int] = []
    for en in energy_names:
        idx = FULL_FEATURE_NAMES.index(en)
        indices.append(idx)
    return indices


def _select_real_samples_by_class(
    signals: np.ndarray, labels: np.ndarray, class_idx: int, max_num: int
) -> np.ndarray:
    """从真实信号中挑选指定类别的一部分样本。"""

    mask = labels == class_idx
    idxs = np.where(mask)[0]
    if len(idxs) == 0:
        raise RuntimeError(f"在真实数据中未找到类别索引 {class_idx} 的样本")

    idxs = idxs[:max_num]
    return signals[idxs]


def plot_time_series_compare(
    real_signals: np.ndarray,
    fake_signals: np.ndarray,
    class_name: str,
    save_path: str | None = None,
) -> None:
    """绘制几条真实 vs 生成的时域波形对比。"""

    num_to_plot = min(3, real_signals.shape[0], fake_signals.shape[0])
    t = np.arange(real_signals.shape[1])

    fig, axes = plt.subplots(num_to_plot, 2, figsize=(10, 6), sharex=True)
    if num_to_plot == 1:
        axes = np.array([axes])

    for i in range(num_to_plot):
        axes[i, 0].plot(t, real_signals[i], color="tab:blue")
        axes[i, 0].set_title(f"真实样本 {i+1}")

        axes[i, 1].plot(t, fake_signals[i], color="tab:orange")
        axes[i, 1].set_title(f"生成样本 {i+1}")

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


def plot_vmd_energy_compare(
    real_feats: np.ndarray,
    fake_feats: np.ndarray,
    energy_indices: List[int],
    class_name: str,
    save_path: str | None = None,
) -> None:
    """绘制真实 vs 生成样本的 VMD 能量分布对比（条形图）。"""

    E_real = real_feats[:, energy_indices].mean(axis=0)
    E_fake = fake_feats[:, energy_indices].mean(axis=0)

    # 归一化为相对能量分布
    E_real_rel = E_real / (np.linalg.norm(E_real) + 1e-8)
    E_fake_rel = E_fake / (np.linalg.norm(E_fake) + 1e-8)

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


def main() -> None:
    parser = argparse.ArgumentParser(description="可视化真实 vs 生成样本（时域 + VMD 能量）")
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/checkpoints/generator_step5_ckpt.pt",
        help="训练好的生成器 checkpoint 路径（由 models.gan_training 保存）",
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default="data/CRWU",
        help="CWRU 数据集根目录",
    )
    parser.add_argument(
        "--class_name",
        type=str,
        default="Inner Race",
        help=f"要可视化的故障类别名称，可选: {CATEGORY_ORDER}",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=50,
        help="用于统计 / 生成的样本数量",
    )
    parser.add_argument(
        "--channel",
        type=str,
        default="DE",
        help="CWRU 通道: DE / FE",
    )
    parser.add_argument(
        "--no_save",
        action="store_true",
        help="若设置，则不保存图片，仅弹出窗口显示",
    )
    args = parser.parse_args()

    # 设置中文字体（必须在创建任何图表之前）
    _setup_chinese_font()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[设备] 使用设备: {device}")

    # ---------- 1) 加载 checkpoint ----------
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"未找到 checkpoint 文件: {args.ckpt}")

    # PyTorch 2.6 开始，torch.load 默认 weights_only=True，会限制反序列化的对象类型，
    # 而我们在 ckpt 中额外保存了 numpy 数组（W_real, E_real_rel），因此需要显式关闭
    # weights_only 限制。由于该 ckpt 来自你本地训练，可认为是“可信来源”。
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)

    gen_cfg_dict = ckpt["config"]
    label_names = ckpt.get("label_names", CATEGORY_ORDER)
    W_real_np = ckpt["W_real"]
    E_real_rel_np = ckpt["E_real_rel"]

    print(f"[CKPT] 已加载: {args.ckpt}")
    print(f"[CKPT] 类别名称: {label_names}")

    # 对齐类别名称到索引
    if args.class_name not in label_names:
        raise ValueError(f"class_name={args.class_name} 不在 ckpt.label_names 中: {label_names}")
    class_idx = label_names.index(args.class_name)
    print(f"[类别] 选择类别: {args.class_name} (索引={class_idx})")

    # ---------- 2) 构建生成器 G ----------
    gen_cfg = GeneratorConfig(**gen_cfg_dict)
    G = AttributeConditionedGenerator(gen_cfg).to(device)
    G.load_state_dict(ckpt["generator_state_dict"])
    G.eval()

    W_real = torch.tensor(W_real_np, dtype=torch.float32, device=device)
    E_real_rel = torch.tensor(E_real_rel_np, dtype=torch.float32, device=device)

    # ---------- 3) 加载真实信号并提取特征 ----------
    signals_np, labels_np, label_names_data = load_cwru_signals(
        root_dir=args.root_dir,
        sample_length=gen_cfg.signal_length,
        num_samples_per_file=args.num_samples,
        channel=args.channel,
    )
    print(f"[数据] 信号形状: {signals_np.shape}, 标签形状: {labels_np.shape}")
    print(f"[数据] CATEGORY_ORDER: {label_names_data}")

    real_signals_class = _select_real_samples_by_class(
        signals_np, labels_np, class_idx, max_num=args.num_samples
    )
    print(f"[真实] 选取 {real_signals_class.shape[0]} 条 {args.class_name} 样本进行对比")

    # 提取真实样本的 31 维特征（包含 VMD）
    real_feats_np, scaler = batch_extract_features(
        real_signals_class,
        fs=12000,
        vmd_params=None,
        scaler=None,
        fit_scaler=True,
        n_jobs=1,
    )

    # ---------- 4) 使用 G 生成同类样本 ----------
    bsz = real_signals_class.shape[0]
    z = torch.randn(bsz, gen_cfg.noise_dim, device=device)
    y = torch.full((bsz,), class_idx, dtype=torch.long, device=device)

    w_batch = W_real[y]  # [B, 31]
    e_batch = E_real_rel[y]  # [B, 4]，这里用相对能量向量近似绝对 e_c

    with torch.no_grad():
        x_fake = G(z, y, w_batch, e_batch)  # [B, L]
    fake_signals_np = x_fake.detach().cpu().numpy()

    # 提取生成样本的 31 维特征，使用相同 scaler 归一化
    fake_feats_np, _ = batch_extract_features(
        fake_signals_np,
        fs=12000,
        vmd_params=None,
        scaler=scaler,
        fit_scaler=False,
        n_jobs=1,
    )

    energy_indices = _find_vmd_energy_indices(full_feature_names=None)

    # ---------- 5) 作图 ----------
    save_dir = os.path.join("tests", "gan_plots")
    if args.no_save:
        save_ts, save_vmd = None, None
    else:
        save_ts = os.path.join(save_dir, f"{args.class_name.replace(' ', '_')}_waveform.png")
        save_vmd = os.path.join(save_dir, f"{args.class_name.replace(' ', '_')}_vmd_energy.png")

    plot_time_series_compare(real_signals_class, fake_signals_np, args.class_name, save_ts)
    plot_vmd_energy_compare(real_feats_np, fake_feats_np, energy_indices, args.class_name, save_vmd)

    print("[完成] 可视化结束。")


if __name__ == "__main__":
    main()


