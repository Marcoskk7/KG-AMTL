我现在要实现一个知识图谱，方便我后续对已有 MAML 模型进行改良，请你对ai 实现的代码进行评审：

# 我的实现思路和对应公式：

## 1. 知识图谱（Knowledge Graph）构建与表示

知识图谱定义为 $KG = (\mathcal{V}, \mathcal{E}, \mathcal{W})$，其中包含物理特征节点、故障类型节点及其相互关系。

### 1.1 节点定义
*   **特征节点 $\mathcal{V}_f$**：包含时域、频域和时频域（VMD模态能量）特征。
*   **故障节点 $\mathcal{V}_d$**：代表不同的轴承状态（如 IR, OR 等）。

### 1.2 特征-故障关联权重计算 (Correlation Matrix)
对于每个特征 $k$ 与故障类别 $i$，计算其物理相关性权重 $w_{ik}$：

1.  **特征中心 $v_{ik}$ 计算**：
    $$v_{ik} = \frac{\sum_{j=1}^{N} u_{ij} x_{jk}}{\sum_{j=1}^{N} u_{ij}} \tag{17}$$
    *其中 $u_{ij}$ 是隶属度，$x_{jk}$ 是样本特征。*

2.  **特征方差 $\sigma_{ik}$ 计算**：
    $$\sigma_{ik} = \sum_{j=1}^{N} u_{ij} (x_{jk} - v_{ik})^2 \tag{16}$$

3.  **归一化关联权重 $w_{ik}$**：
    $$w_{ik} = \frac{(\sigma_{ik})^{-1/2}}{\sum_{k'=1}^{D} (\sigma_{ik'})^{-1/2}} \tag{18}$$
    *该权重反映了特征对特定故障的区分能力。*

### 1.3 故障演化图（Fault Evolution Graph）
构建有向图表示故障状态转移：
$$\mathcal{E} = \{e_{ij} = P(\text{fault}_i \to \text{fault}_j)\}$$


# ai 对公式的实现：
import networkx as nx


def compute_feature_fault_weights(
    X, y, class_names, feature_names, membership_soft=None
):
    """
    X: (N, D) 特征矩阵（你上一步生成的 31 维）
    y: (N,) 硬标签 [0..C-1]
    membership_soft: (N, C) 可选，软成员度（若不提供，按 one-hot）
    返回:
      v: (C, D) 类中心 v_{ik}
      sigma: (C, D) 类内离散度 sigma_{ik} = sum_j u_{ij}(x_{jk}-v_{ik})^2
      w: (C, D) 相关性权重  w_{ik} = (sigma_{ik})^{-1/2} / sum_{k'}(...)
    """
    N, D = X.shape
    C = len(class_names)

    # 成员度 U (N, C)
    if membership_soft is not None:
        U = membership_soft.astype(float)
        U = np.clip(U, 0.0, 1.0)
    else:
        U = np.zeros((N, C), dtype=float)
        U[np.arange(N), y] = 1.0

    # v_{ik}
    denom = U.sum(axis=0, keepdims=True).T  # (C,1)
    denom[denom == 0] = 1e-12
    v = (U.T @ X) / denom  # (C,D)

    # sigma_{ik}
    sigma = np.zeros((C, D), dtype=float)
    for i in range(C):
        diff = X - v[i]  # (N,D)
        sigma[i] = (U[:, i][:, None] * (diff**2)).sum(axis=0)
    sigma = np.maximum(sigma, 1e-12)

    # w_{ik} ∝ (sigma_{ik})^{-1/2}，并在 k 维上归一
    inv_sqrt = 1.0 / np.sqrt(sigma)
    w = inv_sqrt / inv_sqrt.sum(axis=1, keepdims=True)

    # 便于检查：每类权重之和应为 1
    assert np.allclose(w.sum(axis=1), 1.0, atol=1e-6)

    # 保存到磁盘（可视化/复用）
    os.makedirs(KG_SAVE_DIR, exist_ok=True)
    np.savez(
        os.path.join(KG_SAVE_DIR, "kg_step2_w_v_sigma.npz"),
        w=w,
        v=v,
        sigma=sigma,
        feature_names=np.array(feature_names),
        class_names=np.array(class_names),
    )
    return v, sigma, w

def build_fault_transition_matrix(y, num_classes, group_ids=None, smoothing=1e-3):
    """
    P[i,j] = P(fault_i -> fault_j)
    - 若给出 group_ids（同一设备/序列的片段编号），按组内索引顺序统计转移；
    - 否则用"自环为主+拉普拉斯平滑"的先验。
    """
    C = num_classes
    P = np.zeros((C, C), dtype=float)

    if group_ids is not None:
        from collections import defaultdict

        groups = defaultdict(list)
        for idx, g in enumerate(group_ids):
            groups[g].append(idx)
        for g, idxs in groups.items():
            idxs = sorted(idxs)
            for a, b in zip(idxs[:-1], idxs[1:]):
                P[y[a], y[b]] += 1.0
        P = P + smoothing
        P = P / P.sum(axis=1, keepdims=True)
    else:
        # 无时间顺序信息：构造保守的平滑马尔可夫矩阵
        P[:] = smoothing
        np.fill_diagonal(P, 1.0)
        P = P / P.sum(axis=1, keepdims=True)

    np.save(os.path.join(KG_SAVE_DIR, "kg_step3_P_transition.npy"), P)
    return P

def assemble_kg_graph(w, P, class_names, feature_names, graph_name="bearing_KG"):
    """
    返回一个 networkx.DiGraph：
      节点：
        F_<k>  : feature 节点
        C_<i>  : fault 节点
      边：
        (F_k -> C_i)  权重 = w_{ik}
        (C_i -> C_j)  概率 = P_{ij}
    """
    G = nx.DiGraph(name=graph_name)

    # 节点
    for k, fn in enumerate(feature_names):
        G.add_node(f"F_{k}", kind="feature", name=fn)
    for i, cn in enumerate(class_names):
        G.add_node(f"C_{i}", kind="fault", name=cn)

    # feature -> fault
    C, D = w.shape
    for i in range(C):
        for k in range(D):
            weight = float(w[i, k])
            # 可阈值截断使图更清爽，例如 1e-6
            if weight > 1e-8:
                G.add_edge(f"F_{k}", f"C_{i}", etype="feature_fault", weight=weight)

    # fault -> fault
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            prob = float(P[i, j])
            if prob > 1e-8:
                G.add_edge(f"C_{i}", f"C_{j}", etype="fault_transition", prob=prob)

    # 保存 GraphML 和 JSON（任选其一查看/绘图）
    nx.write_graphml(G, os.path.join(KG_SAVE_DIR, "bearing_KG.graphml"))
    data = nx.node_link_data(G, edges="links")
    with open(os.path.join(KG_SAVE_DIR, "bearing_KG.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return G

# 1) 读取或生成样本（你已经有缓存的话，直接加载）
X_signals, y, class_names = load_cwru_data_fixed(file_mapping, DATA_DIR, sample_length=2400, num_samples_per_class=100)

# 2) 特征X
X = signals_to_features(X_signals, fs=12000)
# 保存原始信号序列 (N, 2400)
np.save(SIGNAL_FILE, X_signals)  # SIGNAL_FILE = "cwru_signals_4class.npy"

# 保存特征和标签等信息
np.savez(
    FEATURE_FILE,  # FEATURE_FILE = "cwru_features_labels_4class_optimized.npz"
    X=X,  # 特征矩阵 (N, 31)
    y=y,  # 标签 (N,)
    class_names=np.array(class_names),
    feature_names=np.array(FULL_FEATURE_NAMES),
)
print("保存完成：", SIGNAL_FILE, "和", FEATURE_FILE)