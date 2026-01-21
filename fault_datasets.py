"""
PyTorch dataset classes for CWRU dataset.
"""

import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset
from torchvision import transforms
from preprocess_cwru import load_CWRU_dataset
from utils import extract_dict_data
from PIL import Image
import os

class CWRU_FFT(Dataset):

    def __init__(self, 
                 domain,
                 dir_path,
                 fft=True):
        super(CWRU_FFT, self).__init__()
        self.root = dir_path

        if domain not in [0, 1, 2, 3]:
            raise ValueError('Argument "domain" must be 0, 1, 2 or 3.')
        self.domain = domain
        self.dataset = load_CWRU_dataset(domain, dir_path, raw=True, fft=fft)
        self.data, self.labels = extract_dict_data(self.dataset)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        sample = self.data[index]
        label = self.labels[index]

        sample = torch.from_numpy(sample).float()
        label = torch.tensor(label)
        return sample, label


class CWRU_PhysFeat(Dataset):
    """
    CWRU 31维物理特征数据集（用于 KG-MLP）。

    - 数据来源：`preprocess_cwru.load_CWRU_dataset(..., raw=True, fft=False)` 得到原始时域片段 (1, L)
    - 特征提取：`knowledge_graph.signals_to_features` 输出 (N,31)
      且特征顺序严格固定为 `knowledge_graph.FULL_FEATURE_NAMES`。

    为避免每次运行重复执行 VMD 等耗时步骤，默认启用磁盘缓存：
      cache_dir/CWRU_physfeat_domain{d}_ts{time_steps}_ol{overlap_ratio}.npz
    """

    def __init__(
        self,
        domain: int,
        dir_path: str,
        time_steps: int = 1024,
        overlap_ratio: float = 0.5,
        normalization: bool = False,
        random_seed: int = 42,
        fs: int = 12000,
        scale_features: bool = True,
        cache_dir: str = "./data/physfeat_cache",
        use_cache: bool = True,
    ):
        super().__init__()
        if domain not in [0, 1, 2, 3]:
            raise ValueError('Argument "domain" must be 0, 1, 2 or 3.')
        self.domain = int(domain)
        self.dir_path = dir_path
        self.time_steps = int(time_steps)
        self.overlap_ratio = float(overlap_ratio)
        self.normalization = bool(normalization)
        self.random_seed = int(random_seed)
        self.fs = int(fs)
        self.scale_features = bool(scale_features)

        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(
            cache_dir,
            f"CWRU_physfeat_domain{self.domain}_ts{self.time_steps}_ol{self.overlap_ratio:.2f}_norm{int(self.normalization)}.npz",
        )

        if use_cache and os.path.exists(cache_path):
            cached = np.load(cache_path, allow_pickle=True)
            self.features = cached["X"].astype(np.float32)
            self.labels = cached["y"].astype(np.int64)
            self.feature_names = cached["feature_names"].astype(str).tolist()
        else:
            # 1) 加载原始时域片段（不做 FFT），shape 约为 (N,1,L)
            dataset = load_CWRU_dataset(
                domain=self.domain,
                dir_path=self.dir_path,
                time_steps=self.time_steps,
                overlap_ratio=self.overlap_ratio,
                normalization=self.normalization,
                random_seed=self.random_seed,
                raw=True,
                fft=False,
            )
            signals, y = extract_dict_data(dataset)  # signals: (N,1,L) or (N,L)

            # 2) 31维特征提取（顺序由 FULL_FEATURE_NAMES 固定）
            from knowledge_graph import signals_to_features, FULL_FEATURE_NAMES

            X = signals_to_features(signals, fs=self.fs)  # (N,31)
            if X.shape[1] != 31:
                raise ValueError(f"Expected 31-D features, got {X.shape[1]}")

            # 3) 可选 MinMax 归一化（与 build_kg_cwru 的默认设置一致：scale_features=True）
            if self.scale_features:
                from sklearn.preprocessing import MinMaxScaler

                scaler = MinMaxScaler()
                X = scaler.fit_transform(X)

            self.features = X.astype(np.float32)
            self.labels = np.asarray(y, dtype=np.int64)
            self.feature_names = list(FULL_FEATURE_NAMES)

            if use_cache:
                np.savez(
                    cache_path,
                    X=self.features,
                    y=self.labels,
                    feature_names=np.asarray(self.feature_names),
                    domain=self.domain,
                    time_steps=self.time_steps,
                    overlap_ratio=self.overlap_ratio,
                    normalization=self.normalization,
                    fs=self.fs,
                    scale_features=self.scale_features,
                    random_seed=self.random_seed,
                )

    def __len__(self):
        return int(self.features.shape[0])

    def __getitem__(self, index):
        x = torch.from_numpy(self.features[index]).float()  # (31,)
        y = torch.tensor(int(self.labels[index]), dtype=torch.int64)
        return x, y

if __name__ == '__main__':
    data = CWRU_FFT(1, './data')
    data.__getitem__(0)


