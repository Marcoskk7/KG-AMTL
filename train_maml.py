from utils import setup_logger

import argparse
import os
import maml

SUPPORTED_DATASETS = ['CWRU']  # 预留扩展（如PU），当前仅支持CWRU

def parse_args():
    """
    解析MAML模型的命令行参数
    
    Returns:
        args: 解析后的参数对象
    """
    # 创建参数解析器，设置程序描述
    parser = argparse.ArgumentParser(description='Implementation of \
                                     Model-Agnostic Meta Learning on \
                                     Fault Diagnosis Datasets')
    # 训练参数
    # N-way：每个任务中的类别数
    parser.add_argument('--ways', type=int, default=10,
                        help='Number of classes per task, default=10')
    # K-shot：每个类别的支持样本数
    parser.add_argument('--shots', type=int, default=5,
                        help='Number of support examples per class, default=1')
    
    # 元学习参数
    # 外循环学习率（元学习率）
    parser.add_argument('--meta_lr', type=float, default=0.001,
                        help='Outer loop learning rate, default=0.001')
    # 内循环学习率（快速适应学习率）
    parser.add_argument('--fast_lr', type=float, default=0.1,
                        help='Inner loop learning rate, default=0.1')
    # 内循环适应步数
    parser.add_argument('--adapt_steps', type=int, default=5,
                        help='Number of inner loop steps for adaptation, default=5')
    # 元批次大小：每个批次包含的任务数
    parser.add_argument('--meta_batch_size', type=int, default=32,
                        help='Number of outer loop iterations, \
                              i.e. no. of meta-tasks for each batch, \
                              default=32')
    # 外循环迭代次数
    parser.add_argument('--iters', type=int, default=300,
                        help='Number of outer-loop iterations, default=300')
    # 是否使用一阶近似（简化MAML计算）
    parser.add_argument('--first_order', type=bool, default=True,
                        help='Use the first-order approximation, default=True')
    
    # CUDA和随机种子参数
    # 是否使用CUDA（GPU）
    parser.add_argument('--cuda', type=bool, default=True,
                        help='Use CUDA if available, default=True')
    # 随机种子，用于实验可复现
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed, default=42')
    
    # 数据集参数
    # 数据目录路径
    parser.add_argument('--data_dir_path', type=str, default='./data',
                        help='Path to the data directory, default=./data')
    # 数据集名称
    parser.add_argument('--dataset', type=str, default='CWRU',
                        help='Which dataset to use, \
                            default=CWRU, \
                            options=[CWRU]')
    # 预处理方法
    parser.add_argument('--preprocess', type=str, default='FFT',
                        help='Which preprocessing technique to use, \
                            default=FFT, \
                            options=[FFT, PHYS]')
    # 训练域（多个域用逗号分隔）
    parser.add_argument('--train_domains', type=str, default='0,1,2',
                        help='Training domain, integer(s) separated by commas, default=0,1,2')
    # 测试域（单个整数）
    parser.add_argument('--test_domain', type=int, default=3,
                        help='Test domain, single integer, default=3')
    # 每个训练域的任务数量
    parser.add_argument('--train_task_num', type=int, default=200,
                        help='Number of samples per domain for training, default=200')
    # 测试任务数量
    parser.add_argument('--test_task_num', type=int, default=100,
                        help='Number of samples per domain for testing, default=100')

    # ====== PHYS（31维物理特征 + KG-MLP）分支相关参数 ======
    # 仅在 --preprocess PHYS 时使用，其它预处理方式会被忽略
    parser.add_argument('--kg_dir', type=str, default='./data/kg',
                        help='Directory containing KG files (kg_domain{d}_W_P.npz), used when preprocess=PHYS or FFT+use_kg')
    parser.add_argument('--use_kg', type=bool, default=False,
                        help='Enable KG gate for FFT; PHYS follows original KG behavior, default=False')
    parser.add_argument('--time_steps', type=int, default=1024,
                        help='Window length for raw time-series slicing, used when preprocess=PHYS, default=1024')
    parser.add_argument('--overlap_ratio', type=float, default=0.5,
                        help='Overlap ratio for slicing, used when preprocess=PHYS, default=0.5')
    parser.add_argument('--normalization', type=bool, default=False,
                        help='Whether to normalize raw signal before feature extraction, used when preprocess=PHYS, default=False')
    parser.add_argument('--fs', type=int, default=12000,
                        help='Sampling frequency, used when preprocess=PHYS, default=12000')
    parser.add_argument('--scale_features', type=bool, default=True,
                        help='Whether to MinMax scale extracted 31-D features, used when preprocess=PHYS, default=True')
    
    # 曲线绘制参数
    # 是否绘制学习曲线
    parser.add_argument('--plot', type=bool, default=True,
                        help='Plot the learning curve, default=True')
    # 学习曲线保存目录
    parser.add_argument('--plot_path', type=str, default='./images',
                        help='Directory to save the learning curve, default=./images')
    # 绘制学习曲线的步长间隔
    parser.add_argument('--plot_step', type=int, default=50,
                        help='Step for plotting the learning curve, default=50')
    
    # 日志参数
    # 是否记录日志
    parser.add_argument('--log', type=bool, default=True,
                        help='Log the training process, default=True')
    # 日志保存目录
    parser.add_argument('--log_path', type=str, default='./logs',
                        help='Directory to save the logs, default=./logs')
    
    # 模型检查点参数
    # 是否保存模型检查点
    parser.add_argument('--checkpoint', type=bool, default=True,
                        help='Save the model checkpoints, default=True')
    # 检查点保存目录
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints',
                        help='Directory to save the model checkpoints, default=./checkpoints')
    # 保存检查点的步长间隔
    parser.add_argument('--checkpoint_step', type=int, default=50,
                        help='Step for saving the model checkpoints, default=50')
    
    # 解析并返回参数
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.dataset not in SUPPORTED_DATASETS:
        raise ValueError(f'Dataset must be one of {SUPPORTED_DATASETS}.')
    if args.preprocess not in ['FFT', 'PHYS']:
        raise ValueError('Preprocessing technique must be one of FFT or PHYS.')
    
    args.train_domains = args.train_domains.split(',')
    train_domains_str = ''
    for i in range(len(args.train_domains)):
        train_domains_str += str(args.train_domains[i])
    args.train_domains = [int(i) for i in args.train_domains]

    # Experiment title in the format:
    # MAML_"dataset name"_"number of ways" + "number of shots"_"source domains"_"target domain".log
    experiment_title = 'MAML_{}_{}_{}w{}s_source{}_target{}'.format(args.dataset, 
                                                args.preprocess,
                                                args.ways,
                                                args.shots,
                                                train_domains_str,
                                                args.test_domain)
    if args.plot:
        if not os.path.exists(args.plot_path):
            os.makedirs(args.plot_path)
    
    if args.checkpoint:
        if not os.path.exists(args.checkpoint_path):
            os.makedirs(args.checkpoint_path)
    
    if args.log:
        if not os.path.exists(args.log_path):
            os.makedirs(args.log_path)
        setup_logger(args.log_path, experiment_title)

    maml.train(args, experiment_title)