from typing import Dict, List, Optional

import networkx as nx
import numpy as np


def build_fault_transition_matrix(
    fault_states: List[str],
    preset: str = "doc5_example",
) -> np.ndarray:
    """
    构造一个示例性的故障状态转移概率矩阵。

    根据你的选择，这里采用“手工设定示例矩阵”方案，后续可以很方便地替换为
    从外部 CSV / 实验序列统计得到的矩阵。

    Parameters
    ----------
    fault_states : List[str]
        故障状态名称列表，例如 ["Normal", "IR", "OR", "B"]。
    preset : str
        预设矩阵名称，目前仅支持 "doc5_example"。

    Returns
    -------
    P : np.ndarray
        形状为 (S, S) 的转移概率矩阵，S 为状态数。
        P[i, j] = P(fault_i -> fault_j)。
    """
    n = len(fault_states)
    P = np.zeros((n, n), dtype=float)

    if preset != "doc5_example":
        raise ValueError(f"未知的 preset: {preset}")

    # 一个简单的示例：
    # - 正常状态大部分自环，小概率转移到任意故障；
    # - 故障状态内部可以有弱转移（例如 IR -> OR），并允许自环。
    try:
        idx_normal = fault_states.index("Normal")
    except ValueError:
        idx_normal = None

    for i in range(n):
        if idx_normal is not None and i == idx_normal:
            # Normal -> self 为主，少量到其他
            P[i, i] = 0.85
            remain = 0.15
            if n > 1:
                share = remain / (n - 1)
                for j in range(n):
                    if j != i:
                        P[i, j] = share
        else:
            # 故障状态：自环较大，同时允许向其他故障 / Normal 演化
            P[i, i] = 0.6
            remain = 0.4
            if n > 1:
                share = remain / (n - 1)
                for j in range(n):
                    if j != i:
                        P[i, j] = share

    # 数值稳定性：行归一化
    row_sums = P.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    P = P / row_sums

    return P


def build_fault_evolution_graph(
    fault_states: List[str],
    P: np.ndarray,
    edge_threshold: float = 0.01,
) -> nx.DiGraph:
    """
    根据故障状态列表与转移概率矩阵构建有向故障演化图。
    """
    n = len(fault_states)
    if P.shape != (n, n):
        raise ValueError(f"P 形状 {P.shape} 与 fault_states 长度 {n} 不匹配")

    G = nx.DiGraph()
    for state in fault_states:
        G.add_node(state, type="fault_state")

    for i, src in enumerate(fault_states):
        for j, dst in enumerate(fault_states):
            prob = float(P[i, j])
            if prob >= edge_threshold:
                G.add_edge(src, dst, weight=prob, prob=prob)

    return G


def save_fault_evolution_graph(
    G: nx.DiGraph,
    save_dir: str,
    name_prefix: str = "cwru_fault_evolution",
) -> Dict[str, str]:
    """
    故障演化图的保存接口，与 kg.graph.save_knowledge_graph 保持风格一致。
    """
    import os
    import datetime

    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    paths: Dict[str, str] = {}

    graphml_path = os.path.join(save_dir, f"{name_prefix}_{timestamp}.graphml")
    nx.write_graphml(G, graphml_path)
    paths["graphml"] = graphml_path

    gexf_path = os.path.join(save_dir, f"{name_prefix}_{timestamp}.gexf")
    nx.write_gexf(G, gexf_path)
    paths["gexf"] = gexf_path

    return paths


__all__ = [
    "build_fault_transition_matrix",
    "build_fault_evolution_graph",
    "save_fault_evolution_graph",
]


