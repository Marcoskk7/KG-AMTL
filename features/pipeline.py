from typing import Tuple, Optional, List

import numpy as np
from joblib import Parallel, delayed
from sklearn.preprocessing import MinMaxScaler

from .time_domain import extract_time_domain_features, TIME_FEATURE_NAMES
from .freq_domain import extract_frequency_domain_features, FREQ_FEATURE_NAMES
from .time_freq_vmd import extract_vmd_features, VMD_FEATURE_NAMES


FULL_FEATURE_NAMES: List[str] = TIME_FEATURE_NAMES + FREQ_FEATURE_NAMES + VMD_FEATURE_NAMES


def extract_all_features(
    signal: np.ndarray,
    fs: int = 12000,
    vmd_params: Optional[dict] = None,
) -> list[float]:
    """
    提取单条 1D 信号的全部 31 维特征：
    - 11 个时域特征
    - 12 个频域特征
    - 8 个 VMD 时频特征
    """
    if vmd_params is None:
        vmd_params = {}

    time_features = extract_time_domain_features(signal)
    freq_features = extract_frequency_domain_features(signal, fs=fs)
    vmd_features = extract_vmd_features(signal, **vmd_params)

    features = time_features + freq_features + vmd_features
    if len(features) != 31:
        raise ValueError(f"特征总数应为 31 个，但实际为 {len(features)}")
    return features


def _extract_one_signal(
    signal: np.ndarray,
    fs: int,
    vmd_params: Optional[dict],
) -> list[float]:
    """便于 joblib 并行调用的单样本特征提取包装。"""
    return extract_all_features(signal, fs=fs, vmd_params=vmd_params)


def batch_extract_features(
    signals: np.ndarray,
    fs: int = 12000,
    vmd_params: Optional[dict] = None,
    scaler: Optional[MinMaxScaler] = None,
    fit_scaler: bool = True,
    n_jobs: int = -1,
) -> Tuple[np.ndarray, MinMaxScaler]:
    """
    对一批信号（N×L）执行特征提取并做 MinMax 归一化。

    Parameters
    ----------
    signals : np.ndarray
        形状为 (N, L) 的信号矩阵。
    fs : int, default 12000
        采样频率。
    vmd_params : Optional[dict]
        传递给 extract_vmd_features 的参数。
    scaler : Optional[MinMaxScaler]
        若提供，则使用该 scaler 做变换；否则在内部新建。
    fit_scaler : bool, default True
        若为 True，则在当前特征上拟合 scaler。
    n_jobs : int, default 1
        joblib 并行工作进程数（1 表示串行，-1 表示使用所有 CPU 核）。

    Returns
    -------
    features : np.ndarray
        形状为 (N, 31) 的特征矩阵。
    scaler : MinMaxScaler
        用于该批数据的 MinMaxScaler 实例。
    """
    if vmd_params is None:
        vmd_params = {}

    if n_jobs is None or n_jobs == 1:
        # 串行路径，避免小数据集时的并行开销
        all_features = [
            extract_all_features(sig, fs=fs, vmd_params=vmd_params) for sig in signals
        ]
    else:
        # 并行提取特征
        all_features = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_extract_one_signal)(sig, fs, vmd_params) for sig in signals
        )

    features = np.asarray(all_features, dtype=float)

    if scaler is None:
        scaler = MinMaxScaler()

    if fit_scaler:
        features_scaled = scaler.fit_transform(features)
    else:
        features_scaled = scaler.transform(features)

    return features_scaled, scaler


__all__ = [
    "FULL_FEATURE_NAMES",
    "extract_all_features",
    "batch_extract_features",
]


