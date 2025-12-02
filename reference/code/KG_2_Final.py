import numpy as np
import scipy.io as sio
import os
import random
import datetime
import csv
from typing import Dict
from scipy.fft import fft
from scipy.stats import kurtosis, skew
from vmdpy import VMD
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pickle

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 方案一：数据文件夹与脚本在同一目录下
DATA_DIR = os.path.join(CURRENT_DIR, "CWRU")
FEATURE_FILE = os.path.join(CURRENT_DIR, "cwru_features_labels_4class_optimized.npz")# 缓存的特征文件
SIGNAL_FILE = os.path.join(CURRENT_DIR, "cwru_signals_4class.npy")# 新增：保存原始信号序列的缓存文件
KG_SAVE_DIR = os.path.join(CURRENT_DIR, "knowledge_graphs")# 知识图谱保存目录

os.makedirs(KG_SAVE_DIR, exist_ok=True)  # 确保保存目录存在

# 故障类型到文件的映射
file_mapping = {
    "Normal_0": "97.mat", "Normal_1": "98.mat", "Normal_2": "99.mat", "Normal_3": "100.mat",
    "OR_007_0": "130.mat", "OR_007_1": "131.mat", "OR_007_2": "132.mat", "OR_007_3": "133.mat",
    "OR_021_0": "234.mat", "OR_021_1": "235.mat", "OR_021_2": "236.mat", "OR_021_3": "237.mat",
    "IR_007_0": "105.mat", "IR_007_1": "106.mat", "IR_007_2": "107.mat", "IR_007_3": "108.mat",
    "IR_021_0": "209.mat", "IR_021_1": "210.mat", "IR_021_2": "211.mat", "IR_021_3": "212.mat",
    "B_007_0": "118.mat", "B_007_1": "119.mat", "B_007_2": "120.mat", "B_007_3": "121.mat",
    "B_021_0": "222.mat", "B_021_1": "223.mat", "B_021_2": "224.mat", "B_021_3": "225.mat",
}

# 类别映射（仅英文）
category_mapping = {
    "Normal": "Normal",  # 正常
    "IR": "Inner Race",  # 内圈故障
    "OR": "Outer Race",  # 外圈故障
    "B": "Ball"  # 滚珠故障
}
unique_categories = list(category_mapping.values())

# 特征名称定义（对应31个特征，含具体含义，便于后续分析）
TIME_FEATURE_NAMES = [
    "Mean", "Std", "Sq-Mean-of-Sqrt", "RMS", "Peak",
    "Skewness", "Kurtosis", "Waveform-Factor", "Peak-Factor",
    "Impulse-Factor", "Crest-Factor"
]
FREQ_FEATURE_NAMES = [
    "Freq-Mean", "Freq-Var", "Freq-Skewness", "Freq-Kurtosis",
    "Freq-Center", "Freq-RMS", "Freq-Std", "Freq-4th-RMS",
    "Freq-Shape-Factor", "Freq-Skew-Factor", "Freq-Skewness-2", "Freq-Kurtosis-2"
]
VMD_FEATURE_NAMES = [
    "VMD-Energy-1", "VMD-Energy-2", "VMD-Energy-3", "VMD-Energy-4",
    "VMD-SVD-1", "VMD-SVD-2", "VMD-SVD-3", "VMD-SVD-4"
]
# 合并所有特征名称（共31个）
FULL_FEATURE_NAMES = TIME_FEATURE_NAMES + FREQ_FEATURE_NAMES + VMD_FEATURE_NAMES


# ======== 1. 数据加载函数 =========
def load_cwru_data_fixed(file_mapping: Dict, data_dir: str, sample_length=2400, num_samples_per_class=50):
    X, y = [], []

    for label_name, file_name in file_mapping.items():
        base_type = label_name.split('_')[0]
        category = category_mapping[base_type]
        category_idx = unique_categories.index(category)

        file_path = os.path.join(data_dir, file_name)
        if not os.path.exists(file_path):
            print(f"[文件缺失] {file_path}")
            continue

        data = sio.loadmat(file_path)
        signal_key = [k for k in data.keys() if "_DE_time" in k]
        if not signal_key:
            print(f"[未找到DE_time信号] {file_path}")
            continue

        signal = data[signal_key[0]].flatten()
        if len(signal) < sample_length:
            print(f"[信号过短] {file_path}, 长度: {len(signal)}")
            continue

        for _ in range(num_samples_per_class):
            start = random.randint(0, len(signal) - sample_length)
            X.append(signal[start:start + sample_length])
            y.append(category_idx)

        print(f"[完成] {label_name} -> {category} 从 {file_name} 采样 {num_samples_per_class} 个片段")

    return np.array(X), np.array(y), unique_categories


# ======== 2. 特征提取函数 (更新以匹配论文公式) =========
def extract_time_domain_features(signal):
    """提取时域特征（11个特征，严格按论文TABLE I公式计算）"""
    N = len(signal)
    mean_val = np.mean(signal)  # p1
    std_val = np.std(signal, ddof=1)  # p2

    # 计算中间量
    abs_signal = np.abs(signal)
    sqrt_abs = np.sqrt(abs_signal)
    mean_abs = np.mean(abs_signal)  # 用于多个特征计算
    mean_sqrt_abs = np.mean(sqrt_abs)  # 方根幅值

    # 严格按照论文公式计算
    p1 = mean_val
    p2 = std_val
    p3 = np.square(np.mean(sqrt_abs))  # 平方幅值的均值
    p4 = np.sqrt(np.mean(signal ** 2))  # RMS
    p5 = np.max(abs_signal)  # 峰值
    p6 = skew(signal)  # 偏度
    p7 = kurtosis(signal)  # 峰度
    p8 = p5 / p4  # 峰值因子
    p9 = p5 / np.square(mean_sqrt_abs)  # 裕度因子
    p10 = p4 / mean_abs  # 波形因子
    p11 = p5 / mean_abs  # 脉冲因子


    return [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11]


def extract_frequency_domain_features(signal, fs=12000):
    """提取频域特征（12个特征，严格按论文TABLE II公式计算）"""
    n = len(signal)
    # 计算幅度谱（与论文中的s(l)一致）
    fft_vals = fft(signal)
    amplitude_spectrum = np.abs(fft_vals[:n // 2]) / n  # 幅度谱
    freqs = np.fft.fftfreq(n, 1 / fs)[:n // 2]
    L = len(amplitude_spectrum)
    s = amplitude_spectrum  # 使用s表示频谱，与论文一致

    # 严格按照论文公式计算
    p12 = np.sum(s) / L  # 频谱均值
    p13 = np.sum((s - p12) ** 2) / (L - 1)  # 频谱方差（N-1归一化）
    p14 = np.sum((s - p12) ** 3) / (L * (np.sqrt(p13)) ** 3)  # 频谱偏度
    p15 = np.sum((s - p12) ** 4) / (L * p13 ** 2)  # 频谱峰度
    p16 = np.sum(freqs * s) / np.sum(s)  # 频率重心
    p17 = np.sqrt(np.sum(freqs ** 2 * s) / np.sum(s))  # 均方频率
    p18 = np.sqrt(np.sum((freqs - p16) ** 2 * s) / L)  # 频率标准差
    p19 = np.sqrt(np.sum(freqs ** 4 * s) / np.sum(freqs ** 2 * s))  # 频率均方根
    numerator = np.sum(freqs ** 2 * s)
    denominator = np.sum(s) * np.sum(freqs ** 4 * s)
    p20 = np.sqrt(numerator / denominator)  # 频率形状因子
    p21 = p18 / p16  # 频率偏度因子
    p22 = np.sum((freqs - p16) ** 3 * s) / (L * p18 ** 3)  # 频率偏度因子2
    p23 = np.sum((freqs - p16) ** 4 * s) / (L * p18 ** 4)  # 频率峰度2

    return [p12, p13, p14, p15, p16, p17, p18, p19, p20, p21, p22, p23]


def extract_vmd_features(signal, alpha=1000, tau=0, K=4, DC=0, init=1, tol=1e-7):
    """提取VMD时频域特征（4个能量特征 + 4个奇异值特征）"""
    # VMD分解（需要确保有VMD实现）
    u, _, _ = VMD(signal, alpha, tau, K, DC, init, tol)

    # 确保分解得到4个模态
    if u.shape[0] != 4:
        raise ValueError(f"VMD分解应得到4个模态，但实际得到{u.shape[0]}个")

    # 1. 能量特征（4个）- 严格按论文公式计算
    energy_features = [np.sum(np.square(mode)) for mode in u]

    # 2. 奇异值特征（4个）
    mode_matrix = np.array(u)  # 4×N的模态矩阵
    U, S, Vh = np.linalg.svd(mode_matrix, full_matrices=False)
    svd_features = S.tolist()

    # 如果奇异值少于4个，用0填充
    while len(svd_features) < 4:
        svd_features.append(0.0)

    return energy_features + svd_features


def extract_all_features(signal):
    """提取全部31个特征（时域11 + 频域12 + 时频域8）"""
    time_features = extract_time_domain_features(signal)
    freq_features = extract_frequency_domain_features(signal)
    vmd_features = extract_vmd_features(signal)

    # 验证特征总数是否为31个
    total_features = time_features + freq_features + vmd_features
    if len(total_features) != 31:
        raise ValueError(f"特征总数应为31个，但实际为{len(total_features)}个")

    return total_features


# ======== 3. 特征缓存和标准化 =========
def get_features_and_labels(file_mapping, data_dir, sample_length=2400, num_samples_per_class=50):
    # 检查特征缓存和信号缓存是否存在
    features_available = os.path.exists(FEATURE_FILE)
    signals_available = os.path.exists(SIGNAL_FILE)

    # 如果两者都已缓存，直接加载
    if features_available and signals_available:
        print(f"[加载缓存] 从 {FEATURE_FILE} 读取特征和标签...")
        data = np.load(FEATURE_FILE, allow_pickle=True)
        print(f"[加载缓存] 从 {SIGNAL_FILE} 读取信号序列...")
        signals = np.load(SIGNAL_FILE)
        return data["features"], data["labels"], data["label_names"].tolist(), signals

    print("[未找到缓存] 开始数据处理...")
    X, y, label_names = load_cwru_data_fixed(file_mapping, data_dir, sample_length, num_samples_per_class)

    # 保存原始信号序列
    np.save(SIGNAL_FILE, X)
    print(f"[保存缓存] 信号序列已保存到 {SIGNAL_FILE}")

    features = []
    for idx, sig in enumerate(X):
        try:
            feats = extract_all_features(sig)
            features.append(feats)
            if idx < 5 or idx % 200 == 0:
                print(f"样本 {idx + 1}/{len(X)} 特征预览: {np.round(feats[:5], 4)}...")
        except Exception as e:
            print(f"提取样本 {idx + 1} 的特征时出错: {str(e)}")

    features = np.array(features)
    # 标准化特征
    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)
    print("特征已标准化.")

    np.savez(FEATURE_FILE, features=features, labels=y, label_names=np.array(label_names))
    print(f"[保存缓存] 特征和标签已保存到 {FEATURE_FILE}")

    return features, y, label_names, X  # 返回原始信号序列


# ======== 4. 模糊C均值聚类获取隶属度矩阵 =========
def fuzzy_c_means(X, n_clusters, m=2, max_iter=100, tol=1e-6):
    """实现模糊C均值聚类算法，获取隶属度矩阵"""
    n_samples = X.shape[0]

    # 初始化隶属度矩阵
    u = np.random.rand(n_clusters, n_samples)
    u = u / np.sum(u, axis=0, keepdims=True)

    for _ in range(max_iter):
        # 保存当前隶属度矩阵用于收敛检查
        u_prev = u.copy()

        # 计算聚类中心
        v = np.dot(u ** m, X) / np.sum(u ** m, axis=1, keepdims=True)

        # 更新隶属度矩阵
        for i in range(n_clusters):
            for j in range(n_samples):
                # 计算距离
                dist = np.linalg.norm(X[j] - v[i])
                # 避免除零错误
                if dist < 1e-10:
                    u[i, j] = 1.0
                else:
                    # 计算与所有聚类中心的距离比率
                    ratio_sum = 0.0
                    for k in range(n_clusters):
                        dist_k = np.linalg.norm(X[j] - v[k])
                        ratio_sum += (dist / dist_k) ** (2 / (m - 1))
                    u[i, j] = 1.0 / ratio_sum

        # 检查收敛
        if np.linalg.norm(u - u_prev) < tol:
            break

    return u, v


# ======== 5. 权重计算（严格按照论文公式） =========
def compute_correlation_matrix(features, labels, num_classes):
    """根据论文公式(2)(3)(4)计算特征-故障相关矩阵"""
    D = features.shape[1]  # 特征数量，应为31
    C = num_classes  # 故障类别数量
    W = np.zeros((C, D))  # 相关矩阵

    print(f"计算相关矩阵: 故障类别数={C}, 特征数={D}")

    # 应用模糊C均值聚类获取隶属度矩阵
    u, _ = fuzzy_c_means(features, num_classes)

    for i in range(C):
        # 公式(4): 计算第i类故障的第k维数据中心v_ik
        u_ij = u[i]  # 第i类的隶属度
        sum_u = np.sum(u_ij)

        if sum_u < 1e-10:
            continue

        # 计算每个特征维度的中心
        v_ik = np.zeros(D)
        for k in range(D):
            v_ik[k] = np.sum(u_ij * features[:, k]) / sum_u

        # 公式(3): 计算σ_ik
        sigma_ik = np.zeros(D)
        for k in range(D):
            sigma_ik[k] = np.sum(u_ij * (features[:, k] - v_ik[k]) ** 2)

        # 添加小的常数避免除零错误
        sigma_ik = np.maximum(sigma_ik, 1e-10)

        # 公式(2): 计算w_ik
        w_i = sigma_ik ** (-0.5)
        w_i = w_i / np.sum(w_i)  # 归一化，确保同一故障的所有特征相关性之和为1

        W[i] = w_i

    return W


# ======== 6. 知识图谱构建、保存与可视化（优化） =========
def build_knowledge_graph(feature_names, fault_names, W, edge_threshold=0.005):
    """构建特征-故障双部知识图谱"""
    G = nx.Graph()
    # 添加特征节点（属性bipartite=0标记为特征类）
    for f in feature_names:
        G.add_node(f, bipartite=0, type="feature")
    # 添加故障节点（属性bipartite=1标记为故障类）
    for fault in fault_names:
        G.add_node(fault, bipartite=1, type="fault")

    # 添加边（特征-故障关联，权重为相关矩阵值）
    for fault_idx, fault in enumerate(fault_names):
        for feat_idx, feat in enumerate(feature_names):
            weight = W[fault_idx, feat_idx]
            if weight > edge_threshold:
                G.add_edge(feat, fault, weight=weight, correlation=round(weight, 6))
    return G


def save_knowledge_graph(G, save_name_prefix="cwru_bearing_kg"):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_paths = {
        "graphml": os.path.join(KG_SAVE_DIR, f"{save_name_prefix}_{timestamp}.graphml"),
        "gexf": os.path.join(KG_SAVE_DIR, f"{save_name_prefix}_{timestamp}.gexf"),
        "pickle": os.path.join(KG_SAVE_DIR, f"{save_name_prefix}_{timestamp}.pkl"),
        # 新增三元组格式
        "triples_csv": os.path.join(KG_SAVE_DIR, f"{save_name_prefix}_{timestamp}_triples.csv"),
        "triples_tsv": os.path.join(KG_SAVE_DIR, f"{save_name_prefix}_{timestamp}_triples.tsv")
    }

    # 保存为 GraphML（通用格式）
    nx.write_graphml(G, save_paths["graphml"])

    # 保存为 GEXF（Gephi 可视化）
    nx.write_gexf(G, save_paths["gexf"])

    # 使用标准 pickle 保存（兼容新版 NetworkX）
    with open(save_paths["pickle"], "wb") as f:
        pickle.dump(G, f)

    # 新增：保存为三元组格式（便于直接读取）
    triples = []
    for u, v, data in G.edges(data=True):
        weight = data.get('weight', 0)
        # 确定方向：特征 -> 故障
        if G.nodes[u]['type'] == 'feature' and G.nodes[v]['type'] == 'fault':
            triples.append((u, v, weight))
        elif G.nodes[v]['type'] == 'feature' and G.nodes[u]['type'] == 'fault':
            triples.append((v, u, weight))

    # 保存为CSV
    with open(save_paths["triples_csv"], 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['feature', 'fault', 'weight'])  # 表头
        writer.writerows(triples)

    # 保存为TSV（制表符分隔）
    with open(save_paths["triples_tsv"], 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['feature', 'fault', 'weight'])
        writer.writerows(triples)

    print(f"\n[知识图谱保存完成]")
    for fmt, path in save_paths.items():
        print(f"- {fmt.upper()}: {path}")

    return save_paths


def draw_bipartite_graph(G, feature_names, fault_names):
    """优化边权重可视化：通过宽度+颜色双重区分权重差异"""
    # 分离两类节点
    feature_nodes = [n for n, d in G.nodes(data=True) if d["bipartite"] == 0]
    fault_nodes = [n for n in G.nodes if n not in feature_nodes]

    # 双部图布局（扩大间距，避免节点重叠）
    pos = nx.bipartite_layout(G, feature_nodes, align="horizontal", scale=4.0)
    # 微调故障节点位置（向上偏移，增强视觉分层）
    for node in fault_nodes:
        pos[node] = (pos[node][0], pos[node][1] + 0.5)

    # 提取边权重并优化缩放（解决原权重差异不明显问题）
    edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
    if not edge_weights:
        print("[警告] 知识图谱无有效边（可能阈值过高）")
        return

    # 用95分位数替代最大值，避免极端权重干扰缩放
    weight_95 = np.percentile(edge_weights, 95)
    # 截断极端权重（归一化范围更合理）
    edge_weights_scaled = np.clip(edge_weights, 0, weight_95) / weight_95

    # 边宽度：1~10的范围（线性缩放，差异更明显）
    edge_widths = 1.0 + edge_weights_scaled * 9.0
    # 边颜色：从浅灰到深黑（权重越高颜色越深）
    edge_colors = plt.cm.gray_r(edge_weights_scaled * 0.8 + 0.2)  # 避免全黑

    # 创建图形（更大尺寸，适配31个特征节点）
    plt.figure(figsize=(32, 20))

    # 绘制特征节点（浅蓝色，带边框增强区分）
    nx.draw_networkx_nodes(
        G, pos, nodelist=feature_nodes,
        node_color="#87CEEB", node_size=800, alpha=0.9,
        edgecolors="#4682B4", linewidths=1.5
    )

    # 绘制故障节点（浅红色，更大尺寸突出）
    nx.draw_networkx_nodes(
        G, pos, nodelist=fault_nodes,
        node_color="#FFB6C1", node_size=1500, alpha=0.9,
        edgecolors="#DC143C", linewidths=2.0
    )

    # 绘制边（宽度+颜色双重区分权重）
    nx.draw_networkx_edges(
        G, pos, width=edge_widths, edge_color=edge_colors,
        alpha=0.8, arrows=False
    )

    # 绘制标签（特征标签斜体，故障标签加粗）
    feature_labels = {f: f for f in feature_nodes}
    fault_labels = {f: f for f in fault_nodes}

    nx.draw_networkx_labels(
        G, pos, labels=feature_labels, font_size=9, font_weight="normal",
        font_family="Arial"
    )
    nx.draw_networkx_labels(
        G, pos, labels=fault_labels, font_size=12, font_weight="bold",
        font_family="Arial"
    )

    # 添加颜色条（说明边颜色与权重的对应关系）
    sm = plt.cm.ScalarMappable(cmap=plt.cm.gray_r, norm=plt.Normalize(vmin=0, vmax=weight_95))
    sm.set_array([])
    cbar = plt.colorbar(sm, shrink=0.6, pad=0.02, aspect=20)
    cbar.set_label("Feature-Fault Correlation Weight", fontsize=14, fontweight="bold")

    # 标题与布局
    plt.title("Bearing Fault Knowledge Graph (Feature-Fault Correlation)", fontsize=20, pad=20)
    plt.axis("off")
    plt.tight_layout()
    # 保存可视化图（高分辨率）
    vis_save_path = os.path.join(KG_SAVE_DIR,
                                 f"kg_visualization_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(vis_save_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"[可视化图保存] {vis_save_path}")


# ======== 主流程 =========
if __name__ == "__main__":
    # 1. 获取特征、标签和原始信号
    features, labels, label_names, signals = get_features_and_labels(file_mapping, DATA_DIR)

    print(f"\n数据统计:")
    print(f"故障类别: {label_names}")
    print(f"样本数量: {len(labels)}")
    print(f"每类样本数: {dict(zip(label_names, np.bincount(labels)))}")
    print(f"特征数量: {features.shape[1]} (预期31个: {len(FULL_FEATURE_NAMES)})")
    print(f"信号序列形状: {signals.shape} (样本数×序列长度)")

    # 2. 计算特征-故障相关矩阵
    W = compute_correlation_matrix(features, labels, len(label_names))
    print(f"\n相关矩阵统计:")
    print(f"矩阵形状: {W.shape} (故障数×特征数)")
    row_sums = np.sum(W, axis=1)
    print(f"每行权重和 (应接近1): {np.round(row_sums, 4)}")

    # 3. 构建知识图谱
    KG = build_knowledge_graph(
        feature_names=FULL_FEATURE_NAMES,
        fault_names=label_names,
        W=W,
        edge_threshold=0.005  # 可根据权重分布调整（如0.003/0.007）
    )
    print(f"\n知识图谱统计:")
    print(
        f"节点数: {len(KG.nodes)} (特征{len([n for n, d in KG.nodes(data=True) if d['type'] == 'feature'])}个 + 故障{len([n for n, d in KG.nodes(data=True) if d['type'] == 'fault'])}个)")
    print(f"边数: {len(KG.edges)} (相关权重>0.005的关联)")

    # 4. 保存知识图谱（支持后续对抗网络/数据更新）
    save_paths = save_knowledge_graph(KG, save_name_prefix="cwru_bearing_4class_kg")

    # 5. 打印三元组信息
    print(f"\n三元组格式预览:")
    triples_csv_path = save_paths["triples_csv"]
    with open(triples_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)  # 跳过表头
        for i, row in enumerate(reader):
            if i < 5:  # 只显示前5行
                print(f"  {row[0]} → {row[1]}, 权重: {float(row[2]):.6f}")
            elif i == 5:
                print("  ... (更多三元组请查看CSV/TSV文件)")
                break

    print(f"\n三元组文件可直接用以下方式读取:")
    print(f"  pandas: pd.read_csv('{triples_csv_path}')")
    print(f"  numpy: np.loadtxt('{triples_csv_path}', delimiter=',', skiprows=1, dtype=str)")

    # 6. 可视化知识图谱（边粗细+颜色双重区分权重）
    draw_bipartite_graph(KG, feature_names=FULL_FEATURE_NAMES, fault_names=label_names)

    # 示例：展示如何使用保存的信号数据
    print("\n[信号数据使用示例] 绘制前5个样本的信号波形...")
    plt.figure(figsize=(15, 10))
    for i in range(min(5, len(signals))):
        plt.subplot(5, 1, i + 1)
        plt.plot(signals[i])
        plt.title(f"样本 {i + 1}, 故障类型: {label_names[labels[i]]}")
        plt.ylabel("振幅")
    plt.tight_layout()
    signal_vis_path = os.path.join(KG_SAVE_DIR,
                                   f"signal_visualization_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(signal_vis_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"[信号可视化保存] {signal_vis_path}")



