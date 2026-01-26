# 导入故障诊断数据集类：CWRU数据集（FFT/31维物理特征）
from fault_datasets import CWRU_FFT, CWRU_PhysFeat
# 导入模型：CNN1D / KG_MLP
from models import CNN1D, KG_MLP
# 导入工具函数：打印日志和快速适应函数
from utils import (
    print_logs,  # 打印训练日志
    fast_adapt,  # MAML快速适应函数
)

# 导入日志模块，用于记录训练过程
import logging
# 导入PyTorch深度学习框架
import torch
# 导入随机数生成模块
import random
# 导入NumPy数值计算库
import numpy as np
# 导入learn2learn元学习库
import learn2learn as l2l
# 导入matplotlib绘图库
import matplotlib.pyplot as plt
import os

# 从PyTorch导入神经网络模块
from torch import nn
# 从learn2learn导入数据变换函数，用于构建few-shot学习任务
from learn2learn.data.transforms import (
    FusedNWaysKShots,  # 融合N-way K-shot采样
    LoadData,  # 加载数据
    RemapLabels,  # 重新映射标签
    ConsecutiveLabels,  # 连续标签
)

def train(args, experiment_title):
    """
    在指定数据集上训练MAML模型

    Args:
        args: 解析后的命令行参数
        experiment_title: 实验标题，用于日志和文件命名
    """
    # 记录实验标题到日志
    logging.info('Experiment: {}'.format(experiment_title))
    # 设置随机种子，确保实验可复现
    random.seed(args.seed)  # Python随机数生成器种子
    np.random.seed(args.seed)  # NumPy随机数生成器种子
    torch.manual_seed(args.seed)  # PyTorch随机数生成器种子
    # 设置训练设备，如果可用则使用GPU
    if args.cuda and torch.cuda.is_available():  # 检查是否启用CUDA且GPU可用
        torch.cuda.manual_seed(args.seed)  # 设置CUDA随机数生成器种子
        device_count = torch.cuda.device_count()  # 获取可用GPU数量
        device = torch.device('cuda')  # 设置设备为CUDA（GPU）
        logging.info('Training MAML with {} GPU(s).'.format(device_count))  # 记录使用的GPU数量
    else:
        device = torch.device('cpu')  # 设置设备为CPU
        logging.info('Training MAML with CPU.')  # 记录使用CPU训练
    
    # 创建训练和测试任务数据集
    train_tasks, test_tasks = create_datasets(args)
    # 创建模型、MAML算法、优化器和损失函数
    model, maml, opt, loss = create_model(args, device)

    # 预加载各 domain 的 KG（W,P），训练时按域切换
    kg_cache = None
    use_kg_effective = bool(args.use_kg) or args.preprocess == "PHYS"
    if use_kg_effective:
        kg_cache = _load_kg_cache(args)

    # 开始训练模型
    train_model(args, model, maml, opt, loss, train_tasks, test_tasks, device, experiment_title, kg_cache=kg_cache)


def _load_kg_cache(args):
    """
    从 args.kg_dir 中读取各 domain 的 `kg_domain{d}_W_P.npz` 并缓存。
    训练/测试时会根据当前任务所在域切换到对应 KG。
    """
    domains = list(set(list(args.train_domains) + [int(args.test_domain)]))

    # 兜底：若某些 domain 没有单独的 KG 文件，则回退到第一个存在的 kg_domain{d}_W_P.npz
    fallback_path = None
    for cand in [0, 1, 2, 3, int(args.test_domain)]:
        p = os.path.join(args.kg_dir, f"kg_domain{int(cand)}_W_P.npz")
        if os.path.exists(p):
            fallback_path = p
            break
    if fallback_path is None:
        raise FileNotFoundError(f"No KG file found under {args.kg_dir} (expected kg_domain{{d}}_W_P.npz).")

    cache = {}
    for d in domains:
        kg_path = os.path.join(args.kg_dir, f"kg_domain{int(d)}_W_P.npz")
        if not os.path.exists(kg_path):
            logging.warning(f"KG file not found for domain {int(d)}. Fallback to: {fallback_path}")
            kg_path = fallback_path
        W, P, feature_names = KG_MLP._load_kg_npz(kg_path)
        cache[int(d)] = {"W": W, "P": P, "feature_names": feature_names}
    return cache


def create_datasets(args):
    """
    创建训练、验证和测试数据集
    
    Args:
        args: 解析后的命令行参数
    Returns:
        train_tasks: 训练任务集合
        test_tasks: 测试任务集合
    """
    # 记录训练域信息
    logging.info('Training domains: {}.'.format(args.train_domains))
    # 记录测试域信息
    logging.info('Testing domain: {}.'.format(args.test_domain))
    # 初始化训练数据集列表
    train_datasets = []
    # 初始化训练数据变换列表
    train_transforms = []
    # 初始化训练任务列表
    train_tasks = []

    # 遍历每个训练域，创建对应的数据集
    for i in range(len(args.train_domains)):
        # 如果预处理方式是FFT（快速傅里叶变换）
        if args.preprocess == 'FFT':
            # 使用CWRU_FFT类
            train_datasets.append(CWRU_FFT(args.train_domains[i], 
                                           args.data_dir_path))
        elif args.preprocess == "PHYS":
            # 31维物理特征（严格顺序由 knowledge_graph.FULL_FEATURE_NAMES 固定）
            train_datasets.append(
                CWRU_PhysFeat(
                    args.train_domains[i],
                    args.data_dir_path,
                    time_steps=args.time_steps,
                    overlap_ratio=args.overlap_ratio,
                    normalization=args.normalization,
                    random_seed=args.seed,
                    fs=args.fs,
                    scale_features=args.scale_features,
                )
            )
        else:
            raise ValueError('Unsupported preprocess. Expected FFT or PHYS.')
        # 将数据集包装为元学习数据集
        train_datasets[i] = l2l.data.MetaDataset(train_datasets[i])
        # 定义数据变换管道：N-way K-shot采样、加载数据、重映射标签、连续标签
        train_transforms.append([
            FusedNWaysKShots(train_datasets[i], n=args.ways, k=2*args.shots),  # N-way K-shot采样，k=2*shots因为需要支持集和查询集
            LoadData(train_datasets[i]),  # 加载数据到内存
            RemapLabels(train_datasets[i]),  # 重新映射标签
            ConsecutiveLabels(train_datasets[i]),  # 使标签连续（0,1,2,...）
        ])
        # 创建任务集合，包含指定数量的任务
        train_tasks.append(l2l.data.Taskset(
            train_datasets[i],  # 元学习数据集
            task_transforms=train_transforms[i],  # 任务变换管道
            num_tasks=args.train_task_num,  # 任务数量
        ))
    # 创建测试数据集
    if args.preprocess == 'FFT':
        # 如果预处理方式是FFT
        test_dataset = CWRU_FFT(args.test_domain, 
                                args.data_dir_path)
    elif args.preprocess == "PHYS":
        test_dataset = CWRU_PhysFeat(
            args.test_domain,
            args.data_dir_path,
            time_steps=args.time_steps,
            overlap_ratio=args.overlap_ratio,
            normalization=args.normalization,
            random_seed=args.seed,
            fs=args.fs,
            scale_features=args.scale_features,
        )
    else:
        raise ValueError('Unsupported preprocess. Expected FFT or PHYS.')
    # 将测试数据集包装为元学习数据集
    test_dataset = l2l.data.MetaDataset(test_dataset)
    # 定义测试数据变换管道
    test_transforms = [
        FusedNWaysKShots(test_dataset, n=args.ways, k=2*args.shots),  # N-way K-shot采样
        LoadData(test_dataset),  # 加载数据
        RemapLabels(test_dataset),  # 重映射标签
        ConsecutiveLabels(test_dataset),  # 连续标签
    ]
    # 创建测试任务集合
    test_tasks = l2l.data.Taskset(
        test_dataset,  # 测试数据集
        task_transforms=test_transforms,  # 测试数据变换
        num_tasks=args.test_task_num,  # 测试任务数量
    )

    # 返回训练任务和测试任务
    return train_tasks, test_tasks


def create_model(args, device):
    """
    创建MAML模型、MAML算法、优化器和损失函数
    
    Args:
        args: 解析后的命令行参数
        device: 运行模型的设备（CPU或GPU）
    Returns:
        model: MAML基础模型
        maml: MAML算法包装器
        opt: 优化器
        loss: 损失函数
    """
    # 默认输出类别数为10（CWRU数据集有10类故障）
    output_size = 10
    # 如果预处理方式是FFT，使用一维CNN模型
    if args.preprocess == 'FFT':
        if args.use_kg:
            model = CNN1D(output_size=output_size, use_kg=True)
            kg_path = os.path.join(args.kg_dir, f"kg_domain{int(args.test_domain)}_W_P.npz")
            if not os.path.exists(kg_path):
                for cand in [0, 1, 2, 3]:
                    p = os.path.join(args.kg_dir, f"kg_domain{int(cand)}_W_P.npz")
                    if os.path.exists(p):
                        kg_path = p
                        break
            W, P, feature_names = CNN1D._load_kg_npz(kg_path)
            model.set_kg(W=W, P=P, feature_names=feature_names)
        else:
            model = CNN1D(output_size=output_size)  # 创建一维卷积神经网络
    elif args.preprocess == "PHYS":
        # KG-MLP：先加载一个 KG 文件作为初始化，训练时会按域动态切换到对应 KG
        kg_path = os.path.join(args.kg_dir, f"kg_domain{int(args.test_domain)}_W_P.npz")
        if not os.path.exists(kg_path):
            # 回退到第一个存在的 KG 文件
            for cand in [0, 1, 2, 3]:
                p = os.path.join(args.kg_dir, f"kg_domain{int(cand)}_W_P.npz")
                if os.path.exists(p):
                    kg_path = p
                    break
        model = KG_MLP.from_kg_file(output_size=output_size, kg_npz_path=kg_path)
    else:
        raise ValueError('Unsupported preprocess. Expected FFT or PHYS.')
    # 将模型移动到指定设备（CPU或GPU）
    model.to(device)
    # 创建MAML算法包装器，设置内循环学习率和是否使用一阶近似
    maml = l2l.algorithms.MAML(model, lr=args.fast_lr, first_order=args.first_order)
    # 创建Adam优化器，用于外循环优化，学习率为元学习率
    opt = torch.optim.Adam(model.parameters(), args.meta_lr)
    # 创建交叉熵损失函数，用于分类任务
    loss = nn.CrossEntropyLoss(reduction='mean')

    # 返回模型、MAML算法、优化器和损失函数
    return model, maml, opt, loss


def train_model(args, model, maml, opt, loss, 
                train_tasks, test_tasks, 
                device, 
                experiment_title,
                kg_cache=None):
    """
    训练MAML模型
    
    Args:
        args: 解析后的命令行参数
        model: 基础模型
        maml: MAML算法包装器
        opt: 优化器
        loss: 损失函数
        train_tasks: 训练任务集合
        test_tasks: 测试任务集合
        device: 运行设备
        experiment_title: 实验标题
    """
    # 初始化训练准确率列表
    train_acc_list = []
    # 初始化训练误差列表
    train_err_list = []
    # 初始化测试准确率列表
    test_acc_list = []
    # 初始化测试误差列表
    test_err_list = []

    # 注释掉的代码：将训练域字符串分割并转换为整数列表
    # train_domains = args.train_domains.split(',')
    # train_domains = [int(i) for i in train_domains]

    # 开始训练迭代
    for iteration in range(1, args.iters+1):
        # # KG alpha 退火：前期保持强KG，后期再缓慢减弱到下限（避免中期就“关掉KG”）
        # if hasattr(model, "kg_alpha_scale"):
        #     if args.iters > 1:
        #         progress = (iteration - 1) / float(args.iters - 1)  # [0,1]
        #     else:
        #         progress = 0.0
        #     warmup_ratio = 0.6
        #     min_scale = 0.2
        #     if progress <= warmup_ratio:
        #         alpha_scale = 1.0
        #     else:
        #         t = (progress - warmup_ratio) / max(1e-8, (1.0 - warmup_ratio))  # [0,1]
        #         alpha_scale = min_scale + 0.5 * (1.0 - min_scale) * (1.0 + np.cos(np.pi * t))
        #     model.kg_alpha_scale.fill_(alpha_scale)
        #     if args.log and hasattr(model, "kg_alpha"):
        #         alpha_value = (torch.sigmoid(model.kg_alpha) * model.kg_alpha_scale).item()
        #         logging.info('KG alpha(scale): {:.6f}, alpha: {:.6f}'.format(alpha_scale, alpha_value))
        # 清零梯度，准备新一轮反向传播
        opt.zero_grad()
        # 初始化元训练误差累加器
        meta_train_err_sum = 0.0
        # 初始化元训练准确率累加器
        meta_train_acc_sum = 0.0
        # 初始化元测试误差累加器
        meta_test_err_sum = 0.0
        # 初始化元测试准确率累加器
        meta_test_acc_sum = 0.0

        # 随机选择一个训练域索引
        train_index = random.randint(0, len(args.train_domains)-1)

        # 遍历元批次中的每个任务
        for task in range(args.meta_batch_size):
            # 计算元训练损失
            # 克隆MAML模型，为当前任务创建独立的模型副本
            learner = maml.clone()
            # 若启用KG：按训练域切换 KG（W,P）
            if kg_cache is not None and hasattr(learner, "set_kg") and (args.use_kg or args.preprocess == "PHYS"):
                d = int(args.train_domains[train_index])
                learner.set_kg(**kg_cache[d])
            # 从选定的训练域中采样一个批次
            batch = train_tasks[train_index].sample()
            # 快速适应：在支持集上训练，在查询集上评估
            evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                               learner,
                                                               loss,
                                                               args.adapt_steps,  # 内循环适应步数
                                                               args.shots,  # 每类支持样本数
                                                               args.ways,  # 类别数
                                                               device)
            # 反向传播，计算梯度
            evaluation_error.backward()
            # 累加训练误差
            meta_train_err_sum += evaluation_error.item()
            # 累加训练准确率
            meta_train_acc_sum += evaluation_accuracy.item()

            # 计算元测试损失（用于监控，不用于更新梯度）
            # 克隆MAML模型用于测试
            learner = maml.clone()
            if kg_cache is not None and hasattr(learner, "set_kg") and (args.use_kg or args.preprocess == "PHYS"):
                d = int(args.test_domain)
                learner.set_kg(**kg_cache[d])
            # 从测试任务中采样一个批次
            batch = test_tasks.sample()
            # 快速适应并评估
            evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                               learner,
                                                               loss,
                                                               args.adapt_steps,
                                                               args.shots,
                                                               args.ways,
                                                               device)
            # 累加测试误差（注意：这里不进行反向传播）
            meta_test_err_sum += evaluation_error.item()
            # 累加测试准确率
            meta_test_acc_sum += evaluation_accuracy.item()

        # 计算平均元训练准确率
        meta_train_acc = meta_train_acc_sum / args.meta_batch_size
        # 计算平均元训练误差
        meta_train_err = meta_train_err_sum / args.meta_batch_size
        # 计算平均元测试误差
        meta_test_err = meta_test_err_sum / args.meta_batch_size
        # 计算平均元测试准确率
        meta_test_acc = meta_test_acc_sum / args.meta_batch_size

        # 记录训练准确率
        train_acc_list.append(meta_train_acc)
        # 记录测试准确率
        test_acc_list.append(meta_test_acc)
        # 记录训练误差
        train_err_list.append(meta_train_err)
        # 记录测试误差
        test_err_list.append(meta_test_err)

        # 绘制学习曲线
        if args.plot and iteration % args.plot_step == 0:
            plot_metrics(args, 
                         iteration, 
                         train_acc_list, test_acc_list, 
                         train_err_list, test_err_list, 
                         experiment_title)

        # 保存模型检查点
        if args.checkpoint and iteration % args.checkpoint_step == 0:
            torch.save(model.state_dict(),  # 保存模型状态字典
                       args.checkpoint_path + '/' +
                       experiment_title + 
                       '_{}.pt'.format(iteration))
        # 记录训练指标
        if args.log:
            print_logs(iteration, meta_train_err, meta_train_acc, meta_test_err, meta_test_acc)

        # 平均累积的梯度并优化
        # 由于多个任务累积了梯度，需要除以批次大小进行平均
        for p in model.parameters():
            p.grad.data.mul_(1.0 / args.meta_batch_size)  # 梯度平均化
        # 执行优化步骤，更新模型参数
        opt.step()


def plot_metrics(args, 
                 iteration, 
                 train_acc, test_acc, 
                 train_loss, test_loss, 
                 experiment_title):
    """
    绘制训练和测试的准确率和损失曲线
    
    Args:
        args: 解析后的命令行参数
        iteration: 当前迭代次数
        train_acc: 训练准确率列表
        test_acc: 测试准确率列表
        train_loss: 训练损失列表
        test_loss: 测试损失列表
        experiment_title: 实验标题
    """
    # 检查是否到达绘制步数
    if (iteration % args.plot_step == 0):
        # 创建图形，设置大小为12x4英寸
        plt.figure(figsize=(12, 4))
        # 创建第一个子图（1行2列的第1个）
        plt.subplot(121)
        # 绘制训练准确率曲线，使用圆点标记
        plt.plot(train_acc, '-o', label="train acc")
        # 绘制测试准确率曲线，使用圆点标记
        plt.plot(test_acc, '-o', label="test acc")
        # 设置x轴标签
        plt.xlabel('Iteration')
        # 设置y轴标签
        plt.ylabel('Accuracy')
        # 设置子图标题
        plt.title("Accuracy Curve by Iteration")
        # 显示图例
        plt.legend()
        # 创建第二个子图（1行2列的第2个）
        plt.subplot(122)
        # 绘制训练损失曲线，使用圆点标记
        plt.plot(train_loss, '-o', label="train loss")
        # 绘制测试损失曲线，使用圆点标记
        plt.plot(test_loss, '-o', label="test loss")
        # 设置x轴标签
        plt.xlabel('Iteration')
        # 设置y轴标签
        plt.ylabel('Loss')
        # 设置子图标题
        plt.title("Loss Curve by Iteration")
        # 显示图例
        plt.legend()
        # 注释掉的代码：设置总标题
        # plt.suptitle("CWRU Bearing Fault Diagnosis {}way-{}shot".format(args.ways, args.shots))
        # 保存图形到文件
        plt.savefig(args.plot_path + '/' + experiment_title + '_{}.png'.format(iteration))
        # 显示图形
        plt.show()


# 主程序入口
if __name__ == '__main__':
    train()  # 调用训练函数（需要传入参数，这里可能不完整）