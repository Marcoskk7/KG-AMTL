# 导入日志模块，用于记录程序运行信息
import logging
# 导入操作系统接口模块，用于文件路径操作
import os
# 导入PyTorch深度学习框架
import torch
# 导入PyWavelets小波变换库
import pywt
# 重复导入logging（冗余，可删除）
import logging
# 导入h5py库，用于读取HDF5格式的MATLAB文件（v7.3版本）
import h5py

# 导入NumPy数值计算库
import numpy as np
# 导入matplotlib绘图库
import matplotlib.pyplot as plt

# 从scipy导入短时傅里叶变换函数
from scipy.signal import stft
# 从PIL导入图像处理模块
from PIL import Image



def setup_logger(log_path, experiment_title):
    """
    设置日志记录器，同时输出到控制台和文件
    
    Args:
        log_path: 日志文件保存路径
        experiment_title: 实验标题，用于命名日志文件
    """
    # 获取根日志记录器
    logger = logging.getLogger()
    # 设置日志级别为INFO
    logger.setLevel(logging.INFO)
    # 创建日志格式器，包含时间戳和消息内容
    formatter = logging.Formatter("[%(asctime)s] %(message)s",
                                   datefmt="%Y-%m-%d %H:%M:%S")  # 日期格式：年-月-日 时:分:秒
    # 创建控制台处理器，用于在终端输出日志
    ch = logging.StreamHandler()
    # 设置控制台处理器日志级别
    ch.setLevel(logging.INFO)
    # 为控制台处理器设置格式器
    ch.setFormatter(formatter)
    # 创建文件处理器，用于将日志写入文件
    fh = logging.FileHandler(os.path.join(log_path, 
                                          experiment_title + '.log'))  # 日志文件路径
    # 设置文件处理器日志级别
    fh.setLevel(logging.INFO)
    # 为文件处理器设置格式器
    fh.setFormatter(formatter)
    # 将控制台处理器添加到日志记录器
    logger.addHandler(ch)
    # 将文件处理器添加到日志记录器
    logger.addHandler(fh)


def accuracy(predictions, targets):
    """
    计算分类准确率
    
    Args:
        predictions: 模型预测结果，形状为(batch_size, num_classes)
        targets: 真实标签，形状为(batch_size,)
    Returns:
        准确率（0到1之间的浮点数）
    """
    # 获取预测类别：在最后一个维度上取最大值索引，并调整形状与targets一致
    predictions = predictions.argmax(dim=1).view(targets.shape)
    # 计算预测正确的样本数，转换为浮点数后除以总样本数
    return (predictions == targets).sum().float() / targets.size(0)


def fast_adapt(batch, learner, loss, adaptation_steps, shots, ways, device):
    """
    MAML快速适应函数：在支持集上适应模型，在查询集上评估
    
    Args:
        batch: 包含数据和标签的批次，数据已按类别和样本组织
        learner: MAML克隆的模型，用于当前任务
        loss: 损失函数
        adaptation_steps: 内循环适应步数
        shots: 每类的支持样本数
        ways: 类别数
        device: 运行设备（CPU或GPU）
    Returns:
        valid_error: 查询集上的误差
        valid_accuracy: 查询集上的准确率
    """
    # 从批次中提取数据和标签
    data, labels = batch
    # 将数据和标签移动到指定设备
    data, labels = data.to(device), labels.to(device)

    # 将数据分为适应集（支持集）和评估集（查询集）
    # 初始化适应集索引数组（全为False）
    adaptation_indices = np.zeros(data.size(0), dtype=bool)
    # 每隔一个样本选择为适应集：0, 2, 4, 6, ...（共shots*ways个）
    adaptation_indices[np.arange(shots*ways) * 2] = True
    # 评估集索引是适应集索引的补集
    evaluation_indices = torch.from_numpy(~adaptation_indices)
    # 将适应集索引转换为PyTorch张量
    adaptation_indices = torch.from_numpy(adaptation_indices)
    # 根据索引提取适应集数据和标签
    adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
    # 根据索引提取评估集数据和标签
    evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]

    # 是否支持“support-set 真值标签门控”（用于 KG-MLP，避免 query-set 标签泄漏）
    use_label_gate = bool(getattr(learner, "supports_label_gate", False))

    # 在适应集上适应模型（内循环）
    for step in range(adaptation_steps):
        # 计算适应集上的损失
        if use_label_gate:
            # 仅在 support set 允许使用真值标签构造 Gate=Sigmoid(W_y)
            train_logits = learner(adaptation_data, adaptation_labels)
        else:
            train_logits = learner(adaptation_data)
        train_error = loss(train_logits, adaptation_labels)
        # 使用MAML的adapt方法更新模型参数（一步梯度下降）
        learner.adapt(train_error)

    # 在评估集上评估适应后的模型
    # 使用适应后的模型进行预测
    predictions = learner(evaluation_data)
    # 计算评估集上的损失
    valid_error = loss(predictions, evaluation_labels)
    # 计算评估集上的准确率
    valid_accuracy = accuracy(predictions, evaluation_labels)
    # 返回评估误差和准确率
    return valid_error, valid_accuracy


def pairwise_distances_logits(a, b):
    """
    计算两组向量之间的成对距离，返回负的平方距离作为logits
    
    Args:
        a: 第一组向量，形状为(n, feature_dim)
        b: 第二组向量，形状为(m, feature_dim)
    Returns:
        logits: 成对距离的负值，形状为(n, m)
    """
    # 获取第一组向量的数量
    n = a.shape[0]
    # 获取第二组向量的数量
    m = b.shape[0]
    # 计算成对欧氏距离的负值：先扩展维度，计算差值的平方，求和，取负值
    logits = -((a.unsqueeze(1).expand(n, m, -1) -  # 扩展a到(n, m, feature_dim)
                b.unsqueeze(0).expand(n, m, -1))**2).sum(dim=2)  # 扩展b到(n, m, feature_dim)，计算平方差并求和
    # 返回logits
    return logits

def print_logs(iteration, meta_train_error, meta_train_accuracy, meta_test_error, meta_test_accuracy):
    """
    打印训练日志信息
    
    Args:
        iteration: 当前迭代次数
        meta_train_error: 元训练误差
        meta_train_accuracy: 元训练准确率
        meta_test_error: 元测试误差
        meta_test_accuracy: 元测试准确率
    """
    # 记录迭代次数
    logging.info('Iteration {}:'.format(iteration))
    # 记录元训练结果标题
    logging.info('Meta Train Results:')
    # 记录元训练误差
    logging.info('Meta Train Error: {}.'.format(meta_train_error))
    # 记录元训练准确率
    logging.info('Meta Train Accuracy: {}.'.format(meta_train_accuracy))
    # 记录元测试结果标题
    logging.info('Meta Test Results:')
    # 记录元测试误差
    logging.info('Meta Test Error: {}.'.format(meta_test_error))
    # 记录元测试准确率，末尾换行
    logging.info('Meta Test Accuracy: {}.\n'.format(meta_test_accuracy))

def normalize(data):
    """
    对数据进行最小-最大归一化，将数据缩放到[0, 1]区间
    
    Args:
        data: 输入数据数组
    Returns:
        归一化后的数据
    """
    # 最小-最大归一化公式：(x - min) / (max - min)
    return (data-min(data)) / (max(data)-min(data))


def loadmat_v73(data_path, realaxis, channel):
    with h5py.File(data_path, 'r') as f:
        mat_data = f[f[realaxis]['Y']['Data'][channel][0]]
        return mat_data[:].reshape(-1)
    

def extract_dict_data(dataset):
    x = np.concatenate([dataset[key] for key in dataset.keys()])
    y = []
    for i, key in enumerate(dataset.keys()):
        number = len(dataset[key])
        y.append(np.tile(i, number))
    y = np.concatenate(y)
    return x, y



if __name__ == '__main__':
    pass