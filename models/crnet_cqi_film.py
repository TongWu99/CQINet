r""" CRNet with CQI embedding module using FiLM (Feature-wise Linear Modulation)

将 CQI 向量通过全连接层映射成权重(gamma)和偏置(beta)，作用在 decoder 输出的 CSI 上：
    output = gamma * x + beta

这种方式的优点：
1. 初始化为恒等映射 (gamma=1, beta=0)，避免微调初期性能大幅下降
2. 参数更少，训练更稳定
3. 理论上可以学习更丰富的 CQI 到 CSI 的映射关系
"""

import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_
import math

from utils import logger

__all__ = ["crnet_cqi_film"]


class ConvBN(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, groups=1):
        if not isinstance(kernel_size, int):
            padding = [(i - 1) // 2 for i in kernel_size]
        else:
            padding = (kernel_size - 1) // 2
        super(ConvBN, self).__init__(OrderedDict([
            ('conv', nn.Conv2d(in_planes, out_planes, kernel_size, stride,
                               padding=padding, groups=groups, bias=False)),
            ('bn', nn.BatchNorm2d(out_planes))
        ]))


class CRBlock(nn.Module):
    def __init__(self):
        super(CRBlock, self).__init__()
        self.path1 = nn.Sequential(OrderedDict([
            ('conv3x3', ConvBN(2, 7, 3)),
            ('relu1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv1x9', ConvBN(7, 7, [1, 9])),
            ('relu2', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv9x1', ConvBN(7, 7, [9, 1])),
        ]))
        self.path2 = nn.Sequential(OrderedDict([
            ('conv1x5', ConvBN(2, 7, [1, 5])),
            ('relu', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv5x1', ConvBN(7, 7, [5, 1])),
        ]))
        self.conv1x1 = ConvBN(7 * 2, 2, 1)
        self.identity = nn.Identity()
        self.relu = nn.LeakyReLU(negative_slope=0.3, inplace=True)

    def forward(self, x):
        identity = self.identity(x)

        out1 = self.path1(x)
        out2 = self.path2(x)
        out = torch.cat((out1, out2), dim=1)
        out = self.relu(out)
        out = self.conv1x1(out)

        out = self.relu(out + identity)
        return out


# ==================== FiLM CQI 嵌入模块 ====================

class CQIFiLMModulator(nn.Module):
    """FiLM (Feature-wise Linear Modulation) 调制器
    
    将 CQI 向量映射为 gamma (缩放) 和 beta (偏置)
    
    公式：output = gamma * x + beta
    
    初始化策略：
        - gamma 初始化为 1（恒等映射的缩放）
        - beta 初始化为 0（恒等映射的偏置）
    """

    def __init__(self, num_subbands=13, num_channels=2, hidden_dim=32):
        super().__init__()
        self.num_subbands = num_subbands
        self.num_channels = num_channels
        
        # MLP
        self.fc1 = nn.Linear(num_subbands, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_channels * 2)
        
        # 初始化
        self._init_weights()

    def _init_weights(self):
        """初始化为恒等映射：gamma=0, beta=0（残差形式）"""
        # 全部用常数初始化，确保确定性
        for m in [self.fc1, self.fc2]:
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
        
        # 残差形式：gamma=0, beta=0 → 输出 = out（恒等映射）
        # 由于权重全为0，fc1输出一定为0，GELU(0)=0，fc2输出一定为0

    def forward(self, cqi: torch.Tensor) -> tuple:
        """前向传播
        
        Args:
            cqi: (B, num_subbands) 或 (B, 1) - CQI 向量
            
        Returns:
            gamma: (B, num_channels, 1, 1) - 缩放因子
            beta: (B, num_channels, 1, 1) - 偏置
        """
        B = cqi.shape[0]
        
        # 处理 wideband 模式：扩展到 num_subbands 维
        if cqi.dim() == 2 and cqi.shape[1] == 1:
            cqi_expanded = cqi.expand(-1, self.num_subbands)
        else:
            cqi_expanded = cqi
        
        # MLP 前向
        out = self.fc2(F.gelu(self.fc1(cqi_expanded.float())))
        
        # 分离 gamma 和 beta
        gamma = out[:, :self.num_channels].view(B, self.num_channels, 1, 1)
        beta = out[:, self.num_channels:].view(B, self.num_channels, 1, 1)
        
        return gamma, beta


class CRNet_CQI_FiLM(nn.Module):
    """CRNet with FiLM CQI embedding module
    
    结构：
        输入 CSI → 编码器 → 潜在空间 → 解码器 → FiLM调制 → 输出 CSI
        
    FiLM 调制：
        output = gamma * decoder_output + beta
        其中 gamma, beta 由 CQI 向量通过 MLP 生成
    """

    def __init__(self, reduction=4, film_hidden_dim=32):
        super(CRNet_CQI_FiLM, self).__init__()
        # 32×52×2 = 3328
        total_size, in_channel, w, h = 3328, 2, 52, 32
        logger.info(f'reduction={reduction}')

        # ==================== 编码器部分（与原始 CRNet 完全一致） ====================
        self.encoder1 = nn.Sequential(OrderedDict([
            ("conv3x3_bn", ConvBN(in_channel, 2, 3)),
            ("relu1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("conv1x9_bn", ConvBN(2, 2, [1, 9])),
            ("relu2", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("conv9x1_bn", ConvBN(2, 2, [9, 1])),
        ]))
        self.encoder2 = ConvBN(in_channel, 2, 3)
        self.encoder_conv = nn.Sequential(OrderedDict([
            ("relu1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("conv1x1_bn", ConvBN(4, 2, 1)),
            ("relu2", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ]))
        self.encoder_fc = nn.Linear(total_size, total_size // reduction)

        # ==================== 压缩层 ====================
        self.decoder_fc = nn.Linear(total_size // reduction, total_size)

        # ==================== FiLM CQI 嵌入模块 ====================
        # 初始化为 gamma=1, beta=0，实现恒等映射
        self.cqi_modulator = CQIFiLMModulator(
            num_subbands=13, 
            num_channels=in_channel,
            hidden_dim=film_hidden_dim
        )

        # ==================== 解码器部分（与原始 CRNet 完全一致） ====================
        decoder = OrderedDict([
            ("conv5x5_bn", ConvBN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("CRBlock1", CRBlock()),
            ("CRBlock2", CRBlock())
        ])
        self.decoder_feature = nn.Sequential(decoder)

        # 基础模型参数初始化（跳过 cqi_modulator）
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                # 跳过 cqi_modulator 的 MLP
                if 'CQIFiLMModulator' in str(type(m).__name__):
                    continue
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, cqi=None):
        """前向传播
        
        Args:
            x: (B, 2, 32, 52) - 输入 CSI 矩阵
            cqi: (B, CQI_dim) - CQI 向量，可选
            
        Returns:
            output: (B, 2, 32, 52) - 重建的 CSI 矩阵
        """
        n, c, h, w = x.detach().size()

        # 编码器（与原始 CRNet 一致）
        encode1 = self.encoder1(x)
        encode2 = self.encoder2(x)
        out = torch.cat((encode1, encode2), dim=1)
        out = self.encoder_conv(out)
        out = self.encoder_fc(out.view(n, -1))

        # 压缩到潜在空间
        out = self.decoder_fc(out).view(n, c, h, w)

        # FiLM 调制（如果提供了 CQI）
        if cqi is not None:
            # CQI → gamma (缩放) + beta (偏置)
            gamma, beta = self.cqi_modulator(cqi)  # each: (B, 2, 1, 1)
            
            # 残差调制：output = out + (gamma * out + beta)
            # 初始化为 gamma=0, beta=0 时，out = out（恒等映射）
            # 通过调整 gamma/beta 来逐步添加调整
            film_out = gamma * out + beta
            out = out + film_out

        # 解码器（与原始 CRNet 一致）
        out = self.decoder_feature(out)

        return out


def crnet_cqi_film(reduction=4, film_hidden_dim=32):
    r""" Create a CRNet with FiLM CQI embedding module.

    :param reduction: the reciprocal of compression ratio
    :param film_hidden_dim: hidden dimension for FiLM MLP
    :return: an instance of CRNet_CQI_FiLM
    """

    model = CRNet_CQI_FiLM(reduction=reduction, film_hidden_dim=film_hidden_dim)
    return model
