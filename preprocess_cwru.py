import numpy as np
import logging
import os

from scipy.io import loadmat
from sklearn.utils import shuffle
from utils import (
    setup_logger, 
    normalize,
)


# Name dictionary of different fault types in each working condition
dataname_dict= {
    0:[97, 105, 118, 130, 169, 185, 197, 209, 222, 234],  # load 0 HP, motor speed 1797 RPM
    1:[98, 106, 119, 131, 170, 186, 198, 210, 223, 235],  # load 1 HP, motor speed 1772 RPM
    2:[99, 107, 120, 132, 171, 187, 199, 211, 224, 236],  # load 2 HP, motor speed 1750 RPM
    3:[100, 108, 121, 133, 172, 188, 200, 212, 225, 237]  # load 3 HP, motor speed 1730 RPM
    }  
# Partial part of the axis name
axis = "_DE_time"
# Labels of different fault types
labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def load_CWRU_dataset(
        domain, 
        dir_path, 
        time_steps=1024,
        overlap_ratio=0.5,
        normalization=False,
        random_seed=42,
        raw=False,
        fft=True
):
    logging.info("Domain: {}, normalization: {}, time_steps: {}, overlap_ratio: {}."
                 .format(domain, normalization, time_steps, overlap_ratio))

    # dataset {class label : data list of this class}
    # e.g., {0: [data1, data2, ...], 1: [data1, data2, ...], ...}
    dataset = {label: [] for label in labels}

    for label in labels:
        fault_type = dataname_dict[domain][label]
        if fault_type < 100:
            realaxis = "X0" + str(fault_type) + axis
        else:
            realaxis = "X" + str(fault_type) + axis
        data_path = dir_path + "/CWRU_12k/Drive_end_" + str(domain) + "/" + str(fault_type) + ".mat"
        mat_data = loadmat(data_path)[realaxis].reshape(-1)
        if normalization:
            mat_data = normalize(mat_data)
        
        # Total number of samples is calculated automatically. No need to set it manually.
        stride = int(time_steps * (1 - overlap_ratio))
        sample_number = (len(mat_data) - time_steps) // stride + 1
        logging.info("Loading Data: fault type: {}, total num: {}, sample num: {}"
                     .format(label, mat_data.shape[0], sample_number))
        # sample_number = 20 # for testing

        for i in range(sample_number):
            start:int = i * stride
            end:int = start + time_steps
            sub_data:np.ndarray = mat_data[start : end]
            if raw:
                sub_data = sample_preprocessing(sub_data, fft)
            dataset[label].append(sub_data)
        # Shuffle the data
        dataset[label] = shuffle(dataset[label], random_state=random_seed)
        logging.info("Data is shuffled using random seed: {}\n"
                     .format(random_seed))
    return dataset


def sample_preprocessing(sub_data, fft):
    """
    对单个样本进行预处理
    
    Args:
        sub_data: 原始时序数据样本
        fft: 是否进行FFT变换，默认True
    Returns:
        预处理后的数据，形状为(1, length)
    """
    # 如果需要进行FFT变换
    if fft:
        # 执行快速傅里叶变换
        sub_data = np.fft.fft(sub_data)
        # 计算幅度谱并归一化（除以数据长度）
        sub_data = np.abs(sub_data) / len(sub_data)
        # 只保留正频率部分（FFT结果是对称的，只需一半）
        sub_data = sub_data[:int(sub_data.shape[0] / 2)].reshape(-1,)           
    # 增加一个维度，使形状变为(1, length)，便于后续处理
    sub_data = sub_data[np.newaxis, :]

    # 返回预处理后的数据
    return sub_data


if __name__ == '__main__':
    # 该脚本当前仅保留 1D 数据切片 +（可选）FFT 的预处理能力。
    # WT/STFT 的 2D 时频图像生成已从仓库中移除。
    pass