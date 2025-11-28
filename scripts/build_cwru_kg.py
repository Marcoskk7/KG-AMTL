import argparse
import os
import sys
from datetime import datetime
from typing import Any, Dict

import yaml


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(config_path: str) -> None:
    # 确保项目根目录在 sys.path 中，方便以 `python scripts/build_cwru_kg.py` 方式运行
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    if project_root not in sys.path:
        sys.path.append(project_root)

    # 延迟导入项目内部模块，避免 linter 对“非顶部导入”的警告
    from data.cwru_loader import CATEGORY_ORDER, load_cwru_signals
    from features.pipeline import FULL_FEATURE_NAMES, batch_extract_features
    from kg.fault_evolution import (
        build_fault_evolution_graph,
        build_fault_transition_matrix,
        save_fault_evolution_graph,
    )
    from kg.graph import build_knowledge_graph, save_knowledge_graph
    from kg.visualization import (
        draw_bipartite_graph,
        plot_topk_features,
        plot_weight_heatmap,
    )
    from kg.weighting import compute_correlation_matrix

    cfg = load_config(config_path)

    data_cfg = cfg["data"]
    feat_cfg = cfg["features"]
    kg_cfg = cfg["kg"]
    vis_cfg = cfg.get("visualization", {})
    fault_cfg = cfg.get("fault_evolution", {})

    root_dir = data_cfg["root_dir"]
    channel = data_cfg.get("channel", "DE")
    sample_length = int(data_cfg.get("sample_length", 2400))
    num_samples_per_file = int(data_cfg.get("num_samples_per_file", 50))

    print("=== Step 1: 加载 CWRU 信号并随机裁剪片段 ===")
    signals, labels, label_names = load_cwru_signals(
        root_dir=root_dir,
        sample_length=sample_length,
        num_samples_per_file=num_samples_per_file,
        channel=channel,
    )
    print(f"信号矩阵形状: {signals.shape}")
    print(f"故障类别: {label_names}")

    print("\n=== Step 2: 特征提取与归一化 ===")
    vmd_params = feat_cfg.get("vmd", {})
    fs = int(feat_cfg.get("fs", 12000))
    n_jobs = int(feat_cfg.get("n_jobs", 1))
    features, scaler = batch_extract_features(
        signals,
        fs=fs,
        vmd_params=vmd_params,
        scaler=None,
        fit_scaler=True,
        n_jobs=n_jobs,
    )
    print(f"特征矩阵形状: {features.shape} (预期 31 维)")

    print("\n=== Step 3: 计算特征-故障相关矩阵 W ===")
    fcm_cfg = kg_cfg.get("fcm", {})
    W, u = compute_correlation_matrix(
        features,
        num_classes=len(label_names),
        m=float(fcm_cfg.get("m", 2.0)),
        max_iter=int(fcm_cfg.get("max_iter", 100)),
        tol=float(fcm_cfg.get("tol", 1.0e-6)),
        random_state=fcm_cfg.get("random_state", None),
    )
    print(f"W 形状: {W.shape}")
    print(f"每行权重和 (应接近 1): {W.sum(axis=1)}")

    print("\n=== Step 4: 构建并保存特征-故障知识图谱 ===")
    edge_threshold = float(kg_cfg.get("edge_threshold", 0.005))
    base_save_root = kg_cfg.get("save_dir", "kg_outputs")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(base_save_root, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    print(f"[输出目录] 本次运行结果将保存到: {save_dir}")

    KG = build_knowledge_graph(
        feature_names=FULL_FEATURE_NAMES,
        fault_names=label_names,
        W=W,
        edge_threshold=edge_threshold,
    )
    save_paths = save_knowledge_graph(
        KG,
        save_dir=save_dir,
        save_name_prefix="cwru_bearing_kg",
    )
    print("知识图谱已保存：")
    for k, v in save_paths.items():
        print(f"  {k}: {v}")

    if vis_cfg.get("enable_bipartite_plot", False):
        cfg_png = vis_cfg.get("bipartite_png", "kg_visualization.png")
        png_name = os.path.basename(cfg_png)
        png_path = os.path.join(save_dir, png_name)
        print("\n=== Step 5: 知识图谱可视化 ===")
        draw_bipartite_graph(
            KG,
            feature_names=FULL_FEATURE_NAMES,
            fault_names=label_names,
            save_path=png_path,
            show=False,
        )

    if vis_cfg.get("enable_heatmap", False):
        cfg_heatmap = vis_cfg.get("heatmap_png", "W_heatmap.png")
        heatmap_name = os.path.basename(cfg_heatmap)
        heatmap_path = os.path.join(save_dir, heatmap_name)
        print("\n=== Step 6: 绘制权重矩阵热力图 ===")
        plot_weight_heatmap(
            W,
            fault_names=label_names,
            feature_names=FULL_FEATURE_NAMES,
            save_path=heatmap_path,
            show=False,
        )

    if vis_cfg.get("enable_topk", False):
        topk_k = int(vis_cfg.get("topk_k", 10))
        topk_dir_cfg = vis_cfg.get("topk_dir", "topk_features")
        topk_dir_name = os.path.basename(os.path.normpath(topk_dir_cfg))
        topk_dir = os.path.join(save_dir, topk_dir_name)
        print("\n=== Step 7: 绘制每类故障的 Top-K 特征条形图 ===")
        plot_topk_features(
            W,
            fault_names=label_names,
            feature_names=FULL_FEATURE_NAMES,
            k=topk_k,
            save_dir=topk_dir,
            show=False,
        )

    if fault_cfg.get("enable", False):
        print("\n=== Step 8: 构建示例故障演化图 ===")
        fault_states = fault_cfg.get("fault_states", CATEGORY_ORDER)
        preset = fault_cfg.get("preset", "doc5_example")
        fe_edge_th = float(fault_cfg.get("edge_threshold", 0.01))
        fe_save_dir_cfg = fault_cfg.get("save_dir", save_dir)
        if os.path.isabs(fe_save_dir_cfg):
            fe_save_dir = fe_save_dir_cfg
        else:
            fe_dir_name = os.path.basename(os.path.normpath(fe_save_dir_cfg))
            fe_save_dir = os.path.join(save_dir, fe_dir_name)

        P = build_fault_transition_matrix(fault_states, preset=preset)
        G_fault = build_fault_evolution_graph(
            fault_states, P, edge_threshold=fe_edge_th
        )
        fault_paths = save_fault_evolution_graph(
            G_fault,
            save_dir=fe_save_dir,
            name_prefix="cwru_fault_evolution",
        )
        print("故障演化图已保存：")
        for k, v in fault_paths.items():
            print(f"  {k}: {v}")

    print("\n=== 完成：CWRU 知识图谱构建流水线 ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build CWRU knowledge graph from raw signals."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/kg_cwru.yaml",
        help="Path to YAML config file.",
    )
    args = parser.parse_args()
    main(args.config)
