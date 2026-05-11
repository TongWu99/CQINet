"""
学习率调度器模块
================
提供带有预热的余弦退火学习率调度功能。
"""

import math
from torch.optim.lr_scheduler import _LRScheduler

__all__ = ['WarmUpCosineAnnealingLR', 'FakeLR', 'LinearDecayLR']


class WarmUpCosineAnnealingLR(_LRScheduler):
    """带有预热的余弦退火学习率调度器
    
    学习率变化曲线：
    - 预热阶段：线性增长
    - 余弦阶段：从 lr 下降到 eta_min
    """

    def __init__(
        self, 
        optimizer, 
        T_max: int, 
        T_warmup: int, 
        eta_min: float = 1e-6,
        lr_max: float = None,  # 可选的初始学习率（覆盖base_lrs）
        last_epoch: int = -1
    ):
        """
        Args:
            optimizer: 优化器
            T_max: 总训练步数（余弦退火阶段的长度）
            T_warmup: 预热步数
            eta_min: 最小学习率
            lr_max: 初始学习率（如果为None，使用优化器的初始lr）
            last_epoch: 当前轮数
        """
        self.T_max = T_max
        self.T_warmup = T_warmup
        self.eta_min = eta_min
        self.lr_max = lr_max
        super(WarmUpCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.T_warmup:
            # 预热阶段：线性增长
            warmup_factor = self.last_epoch / max(1, self.T_warmup)
            if self.lr_max is not None:
                return [self.lr_max * warmup_factor] * len(self.optimizer.param_groups)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # 余弦退火阶段
            progress = (self.last_epoch - self.T_warmup) / (self.T_max - self.T_warmup)
            progress = min(progress, 1.0)
            
            cos_decay = 0.5 * (1 + math.cos(math.pi * progress))
            
            if self.lr_max is not None:
                lr_max = self.lr_max
            else:
                lr_max = self.base_lrs[0]
            
            return [self.eta_min + (lr_max - self.eta_min) * cos_decay 
                    for _ in self.base_lrs]


class LinearDecayLR(_LRScheduler):
    """线性衰减学习率调度器（带预热）
    
    学习率变化曲线：
    - 预热阶段：线性增长
    - 线性阶段：从 lr 下降到 eta_min
    """
    
    def __init__(
        self, 
        optimizer, 
        T_max: int, 
        T_warmup: int, 
        eta_min: float = 1e-6,
        lr_max: float = None,
        last_epoch: int = -1
    ):
        """
        Args:
            optimizer: 优化器
            T_max: 总训练步数
            T_warmup: 预热步数
            eta_min: 最小学习率
            lr_max: 初始学习率（如果为None，使用优化器的初始lr）
            last_epoch: 当前轮数
        """
        self.T_max = T_max
        self.T_warmup = T_warmup
        self.eta_min = eta_min
        self.lr_max = lr_max
        super(LinearDecayLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.T_warmup:
            # 预热阶段：线性增长
            warmup_factor = self.last_epoch / max(1, self.T_warmup)
            if self.lr_max is not None:
                return [self.lr_max * warmup_factor] * len(self.optimizer.param_groups)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # 线性衰减阶段
            progress = (self.last_epoch - self.T_warmup) / (self.T_max - self.T_warmup)
            progress = min(progress, 1.0)
            
            if self.lr_max is not None:
                lr_max = self.lr_max
            else:
                lr_max = self.base_lrs[0]
            
            # 线性衰减：lr = eta_min + (lr_max - eta_min) * (1 - progress)
            return [self.eta_min + (lr_max - self.eta_min) * (1 - progress) 
                    for _ in self.base_lrs]


class FakeLR(_LRScheduler):
    """恒定学习率调度器（用于不使用学习率调度的场景）"""
    
    def __init__(self, optimizer):
        super(FakeLR, self).__init__(optimizer=optimizer)

    def get_lr(self):
        return self.base_lrs
