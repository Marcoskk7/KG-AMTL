import numpy as np
from typing import Iterable, List, Tuple


def random_crop(signal: np.ndarray, sample_length: int, rng: np.random.RandomState | None = None) -> np.ndarray:
    """
    从 1D 振动信号中随机裁剪固定长度片段。

    Parameters
    ----------
    signal : np.ndarray
        原始 1D 信号（时间序列）。
    sample_length : int
        需要裁剪的片段长度（采样点数）。
    rng : np.random.RandomState | None
        可选的随机数生成器，便于复现实验；若为 None，则使用全局 np.random。

    Returns
    -------
    np.ndarray
        长度为 sample_length 的信号片段。
    """
    if signal.ndim != 1:
        raise ValueError(f"signal 必须是一维数组，但实际为 {signal.ndim} 维")

    if len(signal) < sample_length:
        raise ValueError(f"signal 长度 {len(signal)} 小于 sample_length {sample_length}")

    if rng is None:
        rng = np.random

    start = rng.randint(0, len(signal) - sample_length + 1)
    return signal[start:start + sample_length]


def segment_signal(
    signal: np.ndarray,
    frame_size: int,
    overlap: float = 0.5,
) -> np.ndarray:
    """
    将 1D 振动信号按固定窗口 + 重叠率切分为多个片段。

    Parameters
    ----------
    signal : np.ndarray
        原始 1D 信号（时间序列）。
    frame_size : int
        每个窗口的长度（采样点数）。
    overlap : float, default 0.5
        相邻窗口之间的重叠比例，取值范围 [0, 1)。

    Returns
    -------
    np.ndarray
        形状为 (num_frames, frame_size) 的二维数组，每行是一个时间片段。
    """
    if signal.ndim != 1:
        raise ValueError(f"signal 必须是一维数组，但实际为 {signal.ndim} 维")

    if not (0 <= overlap < 1):
        raise ValueError(f"overlap 必须在 [0, 1) 之间，但实际为 {overlap}")

    if frame_size <= 0:
        raise ValueError(f"frame_size 必须为正整数，但实际为 {frame_size}")

    if len(signal) < frame_size:
        # 不足一个窗口时，直接返回空数组，更方便上游统一处理
        return np.empty((0, frame_size), dtype=signal.dtype)

    step = int(frame_size * (1 - overlap))
    if step <= 0:
        raise ValueError(
            f"给定 frame_size={frame_size}, overlap={overlap} 导致步长 step={step} <= 0，"
            f"请减小 overlap 或增大 frame_size"
        )

    segments: List[np.ndarray] = []
    start = 0
    while start + frame_size <= len(signal):
        segments.append(signal[start:start + frame_size])
        start += step

    if not segments:
        return np.empty((0, frame_size), dtype=signal.dtype)

    return np.vstack(segments)


def batch_random_crop(
    signals: Iterable[np.ndarray],
    sample_length: int,
    num_crops_per_signal: int,
    rng: np.random.RandomState | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    对一批 1D 信号执行随机裁剪，生成统一长度的片段集合。

    Parameters
    ----------
    signals : Iterable[np.ndarray]
        多个 1D 信号序列。
    sample_length : int
        每个片段的长度。
    num_crops_per_signal : int
        每条原始信号生成的随机片段数。
    rng : np.random.RandomState | None
        可选的随机数生成器。

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        - crops: 形状为 (N, sample_length) 的数组，N 为总片段数；
        - index: 形状为 (N,) 的数组，指明每个片段来源于输入 signals 中的第几个信号。
    """
    if rng is None:
        rng = np.random

    crops: List[np.ndarray] = []
    indices: List[int] = []

    for idx, sig in enumerate(signals):
        for _ in range(num_crops_per_signal):
            try:
                crop = random_crop(sig, sample_length, rng=rng)
            except ValueError:
                # 若单条信号长度不足以裁剪，跳过
                continue
            crops.append(crop)
            indices.append(idx)

    if not crops:
        return np.empty((0, sample_length), dtype=float), np.empty((0,), dtype=int)

    return np.vstack(crops), np.array(indices, dtype=int)


