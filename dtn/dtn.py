"""
True_DTN: 理论下界 - 无预训练直接训练
每个episode独立初始化+训练，模拟最坏情况

理论定位：评估在极端数据稀缺情况下，无先验知识的学习能力
设计目标：为其他方法提供性能比较的基准线
"""
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Dict, Any
import time
import numpy as np

from methods.base_trainer import BaseTrainer, EpisodeMetrics
from models.networks import CNN1dEncoder, LinearClassifier, init_weights
from data.pu_loader import FinetuneTask, get_finetune_loader


class DTNTrainer(BaseTrainer):
    """DTN: 无预训练的理论下界"""

    def __init__(self, config: Any):
        super().__init__('DTN', config)
        self.feature_dim = config.model.feature_dim
        self.learning_rate = config.training.learning_rate

        # 训练轮数配置
        self.train_episode = getattr(
            config.training, 'dtn_train_episode',
            config.training.finetune_episode
        )

        self.test_episode = config.training.test_episode
        self.batch_size_test = config.training.batch_size_test

        # 关键：跟踪当前是第几次 run
        self._current_run_id = 0

    def train(self, metatrain_data: list) -> Tuple[None, float]:
        """预训练阶段 - 完全跳过

    关键设计：不利用任何源域知识
    - 返回None表示无预训练模型
    - 训练时间为0
    - 模拟真实场景中无法获得相关数据的情况
    """
        self._current_run_id += 1
        self.logger.warning("DTN: No pre-training (theoretical lower bound)")
        return None, 0.0

    def test(self, model: None, metatest_data: list) -> Dict[str, Any]:
        """
        测试阶段：每个shot配置测试100个episode

        关键修改：
        - 基于 _current_run_id 生成独立的 run_seed
        - 每次 run 使用不同的任务采样种子

        Args:
            model: 占位符（实际为None）
            metatest_data: 测试类别数据

        Returns:
            results: 各shot配置的测试结果
        """
        # 关键：基于 run_id 生成独立的种子
        run_seed = self.config.training.random_seed + (self._current_run_id - 1) * 100000

        self.logger.info(f"Using run_seed={run_seed} for task sampling")

        results = {}

        for shot in self.config.training.shot_configs:
            self.logger.info(f"Testing {shot}-shot (training from scratch)...")

            # 传递 run_seed 到测试函数
            shot_acc = self._test_single_shot(metatest_data, shot, run_seed)
            results[f'{shot}shot'] = shot_acc

            self.logger.info(
                f"{shot}-shot: Mean={shot_acc['mean']:.4f} ± {shot_acc['std']:.4f}"
            )

        return results

    def _test_single_shot(self, metatest_data: list, shot: int,
                          run_seed: int) -> Dict[str, Any]:
        """测试单个shot配置 - 实际包含训练过程

    关键特性：每个episode完全独立
    - 不同的随机初始化
    - 不同的任务采样
    - 不同的训练过程
    """
        metrics = EpisodeMetrics()

        for episode in range(self.test_episode):
            # 关键：每个 episode 的种子由 run_seed 和 episode_id 共同决定
            episode_seed = run_seed + episode * 1000

            # 设置随机种子（影响任务采样）
            torch.manual_seed(episode_seed)
            np.random.seed(episode_seed)

            # 创建任务（不同 run 的相同 episode_id 会得到不同的任务）
            task = FinetuneTask(
                metatest_data,
                support_num=shot,
                seed=episode_seed
            )

            # 数据加载器
            support_loader = get_finetune_loader(
                task,
                batch_size=len(task.support_files),
                split='support',
                shuffle=True,
                data_type=self.config.data.data_type
            )

            query_loader = get_finetune_loader(
                task,
                batch_size=self.batch_size_test,
                split='query',
                shuffle=False,
                data_type=self.config.data.data_type
            )

            # 🎯 核心差异点：每个episode从头训练
            accuracy = self._train_and_evaluate(
                support_loader, query_loader,
                len(metatest_data), episode_seed
            )

            metrics.update(accuracy)

            # 定期输出进度
            if (episode + 1) % 20 == 0:
                self.logger.info(
                    f"Test Episode {episode + 1}/{self.test_episode} - "
                    f"Acc: {accuracy:.4f}"
                )

        return metrics.compute()

    def _train_and_evaluate(self, support_loader, query_loader,
                            num_classes: int, episode_seed: int) -> float:
        """
        单个episode的训练+评估流程

        关键修改：
        - 使用 episode_seed 初始化模型权重
        - 确保不同 run 的相同 episode_id 产生不同的模型初始化

        Args:
            support_loader: 支持集数据
            query_loader: 查询集数据
            num_classes: 类别数
            episode_seed: episode 随机种子

        Returns:
            accuracy: 查询集准确率
        """
        # 1. 设置随机种子（影响模型初始化）
        torch.manual_seed(episode_seed)
        np.random.seed(episode_seed)

        # 2. 随机初始化模型
        feature_encoder = CNN1dEncoder(
            feature_dim=self.feature_dim,
            flatten=True
        ).to(self.device)

        classifier = LinearClassifier(
            input_dim=self.feature_dim * 25,
            num_classes=num_classes
        ).to(self.device)

        init_weights(feature_encoder)
        init_weights(classifier)

        # 3. 优化器 - 同时优化编码器和分类器
        optimizer = optim.Adam(
            list(feature_encoder.parameters()) + list(classifier.parameters()),
            lr=self.learning_rate
        )
        criterion = nn.CrossEntropyLoss()

        # 4. 训练循环 - 在少量支持集样本上训练
        feature_encoder.train()
        classifier.train()

        for epoch in range(self.train_episode):
            for batch_x, batch_y in support_loader:
                batch_x, batch_y = self._to_device(batch_x, batch_y)

                # 前向传播
                features = feature_encoder(batch_x)
                logits = classifier(features)
                loss = criterion(logits, batch_y)

                # 反向传播 - 更新所有参数
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # 5. 评估 - 在查询集上测试
        feature_encoder.eval()
        classifier.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for batch_x, batch_y in query_loader:
                batch_x, batch_y = self._to_device(batch_x, batch_y)

                features = feature_encoder(batch_x)
                logits = classifier(features)
                pred = torch.argmax(logits, dim=1)

                correct += (pred == batch_y).sum().item()
                total += batch_y.size(0)

        return correct / total

