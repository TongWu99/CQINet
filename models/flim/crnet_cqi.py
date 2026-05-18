r""" CRNet with simple CQI embedding module for fine-tuning
"""

import torch
import torch.nn as nn
from collections import OrderedDict

from utils import logger

__all__ = ["crnet_cqi"]


# ==================== CQI 配置 ====================
# CQI 是离散索引 (0-15)，共 16 个等级
NUM_CQI = 16


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


# ==================== CQI 嵌入模块 ====================

class CQIAffineModulator(nn.Module):
    """CQI 仿射调制器

    简单的线性嵌入：H' = α * Hde + β

    每个 CQI 索引对应一组可学习的标量参数 αᵢ 和 βᵢ。

    Args:
        num_cqi: CQI 索引范围 (默认16，对应 0-15)
        initial_alpha: α 初始值 (默认1.0)
        initial_beta: β 初始值 (默认0.0)
    """

    def __init__(self, num_cqi=16, initial_alpha=1.0, initial_beta=0.0):
        super().__init__()
        self.num_cqi = num_cqi

        # 16 个 α 参数 + 16 个 β 参数（均为标量）
        self.alpha = nn.Parameter(torch.ones(num_cqi) * initial_alpha)
        self.beta = nn.Parameter(torch.ones(num_cqi) * initial_beta)

    def forward(self, hde: torch.Tensor, cqi: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hde: (B, 2, H, W) - backbone 输出的重建特征
            cqi: (B,) - CQI 离散索引 (0-15)
        Returns:
            h_prime: (B, 2, H, W) - 调制后的 CSI 输出
        """
        B = hde.shape[0]

        # 将 CQI 索引转换为整数
        cqi_idx = cqi.long() if cqi.dtype.is_floating_point else cqi

        # 收集对应索引的 α 和 β
        alpha = self.alpha[cqi_idx]  # (B,)
        beta = self.beta[cqi_idx]     # (B,)

        # 扩展到与 hde 相同的形状
        alpha = alpha.view(B, 1, 1, 1)  # (B, 1, 1, 1)
        beta = beta.view(B, 1, 1, 1)    # (B, 1, 1, 1)

        # 仿射调制：H' = α * Hde + β
        h_prime = alpha * hde + beta

        return h_prime


class CRNet_CQI(nn.Module):
    """CRNet with simple CQI embedding module

    Backbone: 原始 CRNet 编码器-解码器结构
    Embedding: H' = α * Hde + β
    """

    def __init__(self, reduction=4):
        super(CRNet_CQI, self).__init__()
        # 32×52×2 = 3328
        total_size, in_channel, w, h = 3328, 2, 52, 32
        logger.info(f'reduction={reduction}')

        # ==================== Backbone: 编码器 ====================
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

        # ==================== Backbone: 解码器 ====================
        self.decoder_fc = nn.Linear(total_size // reduction, total_size)

        decoder = OrderedDict([
            ("conv5x5_bn", ConvBN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("CRBlock1", CRBlock()),
            ("CRBlock2", CRBlock())
        ])
        self.decoder_feature = nn.Sequential(decoder)

        # ==================== CQI 嵌入模块 ====================
        # 简单的仿射调制：H' = α * Hde + β
        # α 和 β 各 16 个可学习参数（对应 CQI 0-15）
        self.cqi_modulator = CQIAffineModulator(
            num_cqi=NUM_CQI,
            initial_alpha=1.0,
            initial_beta=0.0
        )

        # 参数初始化
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, cqi=None):
        """前向传播

        Args:
            x: (B, 2, 32, 52) - 输入 CSI 矩阵
            cqi: (B,) - CQI 离散索引 (0-15)，可选
        """
        n, c, h, w = x.detach().size()

        # Backbone: 编码器
        encode1 = self.encoder1(x)
        encode2 = self.encoder2(x)
        out = torch.cat((encode1, encode2), dim=1)
        out = self.encoder_conv(out)
        out = self.encoder_fc(out.view(n, -1))

        # Backbone: 解码器
        hde = self.decoder_fc(out).view(n, c, h, w)
        hde = self.decoder_feature(hde)

        # CQI 嵌入：H' = α * Hde + β
        if cqi is not None:
            h_prime = self.cqi_modulator(hde, cqi)
            return h_prime

        return hde


def crnet_cqi(reduction=4,sr_hidden_channels=16):
    r""" Create a CRNet with simple CQI embedding module.

    :param reduction: the reciprocal of compression ratio
    :return: an instance of CRNet_CQI
    """

    model = CRNet_CQI(reduction=reduction)
    return model
