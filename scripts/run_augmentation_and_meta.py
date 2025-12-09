from __future__ import annotations

"""
脚本：一键运行 2.2 + 2.3 的核心实验流程。

包含两部分：
    - 2.2 Physics-Constrained Generative Augmentation
        => 调用 models.gan_training.train_gan_with_physics
    - 2.3 Knowledge-Guided Meta-Transfer Learning
        => 调用 models.meta_transfer.train_meta_with_kg

说明：
    - 2.1 Knowledge Graph Construction 已由 scripts/build_cwru_kg.py 实现；
      本脚本更多是一个「整合入口」，方便从命令行快速跑通生成增强 + 元迁移学习。
"""

import argparse
import os
import sys


def main() -> None:
    # 确保项目根目录在 sys.path 中
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    if project_root not in sys.path:
        sys.path.append(project_root)

    # 延迟导入，避免静态检查器对「非顶部导入」报警
    from models.gan_training import GANTrainConfig, train_gan_with_physics
    from models.meta_transfer import MetaTransferConfig, train_meta_with_kg

    parser = argparse.ArgumentParser(
        description=(
            "Run physics-constrained GAN training (2.2) and "
            "KG-guided meta-transfer learning (2.3)."
        )
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default=None,
        help="CWRU 数据根目录；若不指定，则默认使用 <project_root>/data/CWRU。",
    )
    parser.add_argument(
        "--skip_gan",
        action="store_true",
        help="若指定，则跳过 2.2 GAN 物理约束训练，仅运行 2.3 元迁移学习。",
    )
    parser.add_argument(
        "--skip_meta",
        action="store_true",
        help="若指定，则跳过 2.3 KG-AMTL 训练，仅运行 2.2 GAN。",
    )
    args = parser.parse_args()

    if args.root_dir is None:
        root_dir = os.path.join(project_root, "data", "CWRU")
    else:
        root_dir = args.root_dir

    print(f"[配置] 使用数据根目录: {root_dir}")

    # ---------- 2.2: 物理约束生成增强（GAN 训练） ----------
    if not args.skip_gan:
        print("\n=== 运行 2.2: Physics-Constrained Generative Augmentation (GAN) ===")
        gan_cfg = GANTrainConfig()
        # 覆盖默认 root_dir，使其与当前项目路径一致
        gan_cfg.root_dir = root_dir
        train_gan_with_physics(gan_cfg)
    else:
        print("\n[跳过] 2.2 GAN 训练（根据 --skip_gan 参数）")

    # ---------- 2.3: KG 引导元迁移学习 ----------
    if not args.skip_meta:
        print("\n=== 运行 2.3: Knowledge-Guided Meta-Transfer Learning ===")
        meta_cfg = MetaTransferConfig()
        meta_cfg.root_dir = root_dir
        train_meta_with_kg(meta_cfg)
    else:
        print("\n[跳过] 2.3 元迁移学习训练（根据 --skip_meta 参数）")


if __name__ == "__main__":
    main()


