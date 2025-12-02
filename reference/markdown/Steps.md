# 2 Detailed Implementation Steps

## 2.1 Knowledge Graph Construction

**Step 1: Physical Feature Extraction**
Extract multi-domain features from vibration signals:

$$
\begin{aligned}
\mathcal{P}_{time} &= \{p_1, \cdots, p_{11}\} \quad (\text{Time-domain}) \\
\mathcal{P}_{freq} &= \{p_{12}, \cdots, p_{23}\} \quad (\text{Frequency-domain}) \\
\mathcal{P}_{t-f} &= \{E_m, \lambda_k\} \quad (\text{Time-frequency})
\end{aligned}
$$

where $E_m = \int_{-\infty}^{+\infty} |C_m|^2 dt$ is VMD modal energy (Doc5 Eq.1).
*Reference: Doc5 Tables I-II*

**Step 2: Feature-Fault Correlation**
Compute physical correlation weights:

$$
w_{ik} = \frac{(\sigma_{ik})^{-1/2}}{\sum_{k'=1}^{D}(\sigma_{ik'})^{-1/2}}, \quad \sigma_{ik} = \sum_{j=1}^{N} u_{ij}(x_{jk} - v_{ik})^2 \tag{1}
$$

where $v_{ik}$ is feature center (Doc5 Eq.4).
*Reference: Doc5 Section III.B*

**Step 3: Fault Evolution Graph**
Construct directed graph $G = (\mathcal{V}, \mathcal{E})$:

$$
\begin{aligned}
\mathcal{V} &= \{\text{IR}_1, \text{OR}_2, \cdots\} \\
\mathcal{E} &= \{e_{ij} = P(\text{fault}_i \to \text{fault}_j)\}
\end{aligned}
$$

*Reference: Doc5 Fig.4 fault types*

Figure 2: Fault evolution graph showing state transition probabilities