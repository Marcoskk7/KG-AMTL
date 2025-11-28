from typing import Dict, List

import csv
import datetime
import os

import networkx as nx


def build_knowledge_graph(
    feature_names: List[str],
    fault_names: List[str],
    W: "np.ndarray",
    edge_threshold: float = 0.005,
) -> nx.Graph:
    """
    构建特征-故障双部知识图谱。

    其中：
    - 特征节点 bipartite=0, type="feature";
    - 故障节点 bipartite=1, type="fault"。
    边权重来自相关矩阵 W。
    """
    import numpy as np  # 延迟导入，避免循环依赖

    if W.shape != (len(fault_names), len(feature_names)):
        raise ValueError(
            f"W 形状 {W.shape} 与故障/特征数量不匹配：len(fault_names)={len(fault_names)}, len(feature_names)={len(feature_names)}"
        )

    G = nx.Graph()

    for f in feature_names:
        G.add_node(f, bipartite=0, type="feature")

    for fault in fault_names:
        G.add_node(fault, bipartite=1, type="fault")

    for fault_idx, fault in enumerate(fault_names):
        for feat_idx, feat in enumerate(feature_names):
            weight = float(W[fault_idx, feat_idx])
            if weight > edge_threshold:
                G.add_edge(feat, fault, weight=weight, correlation=round(weight, 6))

    return G


def save_knowledge_graph(
    G: nx.Graph,
    save_dir: str,
    save_name_prefix: str = "cwru_bearing_kg",
    save_graphml: bool = True,
    save_gexf: bool = True,
    save_pickle: bool = True,
    save_triples: bool = True,
) -> Dict[str, str]:
    """
    将知识图谱保存为多种格式（GraphML / GEXF / pickle / triples CSV/TSV）。

    返回一个字典：格式名 -> 路径。
    """
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    save_paths: Dict[str, str] = {}

    if save_graphml:
        path = os.path.join(save_dir, f"{save_name_prefix}_{timestamp}.graphml")
        nx.write_graphml(G, path)
        save_paths["graphml"] = path

    if save_gexf:
        path = os.path.join(save_dir, f"{save_name_prefix}_{timestamp}.gexf")
        nx.write_gexf(G, path)
        save_paths["gexf"] = path

    if save_pickle:
        import pickle

        path = os.path.join(save_dir, f"{save_name_prefix}_{timestamp}.pkl")
        with open(path, "wb") as f:
            pickle.dump(G, f)
        save_paths["pickle"] = path

    if save_triples:
        triples: List[tuple] = []
        for u, v, data in G.edges(data=True):
            weight = data.get("weight", 0.0)
            if G.nodes[u].get("type") == "feature" and G.nodes[v].get("type") == "fault":
                triples.append((u, v, weight))
            elif G.nodes[v].get("type") == "feature" and G.nodes[u].get("type") == "fault":
                triples.append((v, u, weight))

        csv_path = os.path.join(save_dir, f"{save_name_prefix}_{timestamp}_triples.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["feature", "fault", "weight"])
            writer.writerows(triples)
        save_paths["triples_csv"] = csv_path

        tsv_path = os.path.join(save_dir, f"{save_name_prefix}_{timestamp}_triples.tsv")
        with open(tsv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="t")
            writer.writerow(["feature", "fault", "weight"])
            writer.writerows(triples)
        save_paths["triples_tsv"] = tsv_path

    return save_paths


__all__ = ["build_knowledge_graph", "save_knowledge_graph"]


