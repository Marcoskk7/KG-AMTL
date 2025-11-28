from typing import List, Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns


def draw_bipartite_graph(
    G: nx.Graph,
    feature_names: List[str],
    fault_names: List[str],
    save_path: Optional[str] = None,
    show: bool = False,
) -> None:
    """
    对特征-故障双部图进行可视化。

    实现思路与 reference/KG_2_Final.py 中的 draw_bipartite_graph 保持一致：
    - 特征节点和故障节点分行排列；
    - 边宽和颜色由权重决定；
    - 可选择保存为 PNG 文件。
    """
    feature_nodes = [n for n in G.nodes if G.nodes[n].get("type") == "feature"]
    fault_nodes = [n for n in G.nodes if G.nodes[n].get("type") == "fault"]

    # 双部布局
    pos = nx.bipartite_layout(G, feature_nodes, align="horizontal", scale=4.0)
    for node in fault_nodes:
        pos[node] = (pos[node][0], pos[node][1] + 0.5)

    edge_weights = [G[u][v].get("weight", 0.0) for u, v in G.edges()]
    if not edge_weights:
        print("[可视化] 图中无边，不绘制。")
        return

    weight_95 = np.percentile(edge_weights, 95)
    if weight_95 <= 0:
        weight_95 = max(edge_weights)
    if weight_95 <= 0:
        weight_95 = 1.0

    edge_weights_scaled = np.clip(edge_weights, 0, weight_95) / weight_95
    edge_widths = 1.0 + edge_weights_scaled * 9.0
    edge_colors = plt.cm.gray_r(edge_weights_scaled * 0.8 + 0.2)

    # 显式创建 Figure 和 Axes，后续 colorbar 使用同一 Axes，避免 Matplotlib 报错
    fig, ax = plt.subplots(figsize=(32, 20))

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=feature_nodes,
        node_color="#87CEEB",
        node_size=800,
        alpha=0.9,
        edgecolors="#4682B4",
        linewidths=1.5,
        ax=ax,
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=fault_nodes,
        node_color="#FFB6C1",
        node_size=1500,
        alpha=0.9,
        edgecolors="#DC143C",
        linewidths=2.0,
        ax=ax,
    )

    nx.draw_networkx_edges(
        G,
        pos,
        width=edge_widths,
        edge_color=edge_colors,
        alpha=0.8,
        arrows=False,
        ax=ax,
    )

    nx.draw_networkx_labels(
        G,
        pos,
        labels={f: f for f in feature_nodes},
        font_size=9,
        font_weight="normal",
        font_family="Arial",
        ax=ax,
    )
    nx.draw_networkx_labels(
        G,
        pos,
        labels={f: f for f in fault_nodes},
        font_size=12,
        font_weight="bold",
        font_family="Arial",
        ax=ax,
    )

    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.gray_r, norm=plt.Normalize(vmin=0, vmax=weight_95)
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02, aspect=20)
    cbar.set_label("Feature-Fault Correlation Weight", fontsize=14, fontweight="bold")

    ax.set_title(
        "Bearing Fault Knowledge Graph (Feature-Fault Correlation)", fontsize=20, pad=20
    )
    ax.set_axis_off()
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[可视化图保存] {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_weight_heatmap(
    W: np.ndarray,
    fault_names: List[str],
    feature_names: List[str],
    save_path: Optional[str] = None,
    show: bool = False,
) -> None:
    """
    使用 seaborn 绘制特征-故障权重矩阵 W 的热力图。
    """
    import pandas as pd

    df = pd.DataFrame(W, index=fault_names, columns=feature_names)
    plt.figure(figsize=(14, 6))
    sns.heatmap(df, cmap="viridis", cbar=True)
    plt.xlabel("Features")
    plt.ylabel("Fault classes")
    plt.title("Feature-Fault Correlation Matrix (W)")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[热力图保存] {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_topk_features(
    W: np.ndarray,
    fault_names: List[str],
    feature_names: List[str],
    k: int = 10,
    save_dir: Optional[str] = None,
    show: bool = False,
) -> None:
    """
    对每个故障类别，绘制 Top-k 相关特征的条形图。
    """
    import os

    num_classes, num_feats = W.shape
    k = max(1, min(k, num_feats))

    for i in range(num_classes):
        weights = W[i]
        idx = np.argsort(weights)[::-1][:k]
        feats = [feature_names[j] for j in idx]
        vals = weights[idx]

        plt.figure(figsize=(10, 4))
        sns.barplot(x=feats, y=vals, palette="Blues_d")
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Weight")
        plt.title(f"Top-{k} Features for Fault: {fault_names[i]}")
        plt.tight_layout()

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            fname = f"top{k}_features_{fault_names[i].replace(' ', '_')}.png"
            path = os.path.join(save_dir, fname)
            plt.savefig(path, dpi=300, bbox_inches="tight")
            print(f"[Top-{k} 特征图保存] {path}")

        if show:
            plt.show()
        else:
            plt.close()


__all__ = ["draw_bipartite_graph", "plot_weight_heatmap", "plot_topk_features"]
