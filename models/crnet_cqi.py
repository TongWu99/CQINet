r""" CRNet with CQI embedding module for fine-tuning
"""

import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_
import math

from utils import logger

__all__ = ["crnet_cqi"]


# ==================== Patch 形状配置（对比实验用） ====================
# CSI 形状: (32, 52) = 32行(天线) × 52列(频率)
# 总 patch 数必须等于 52
# 配置格式: CSI_ROWS // patch_rows = num_patch_rows, CSI_COLS // patch_cols = num_patch_cols
# 要求: patch_rows × patch_cols = 52

CSI_SHAPE = (32, 52)  # CSI 原始形状 (行, 列)
PATCH_ROWS = 8        # ← 每次运行时修改这里！
PATCH_COLS = 4        # ← 每次运行时修改这里！
# 验证: 确保 PATCH_ROWS × PATCH_COLS = 52
assert PATCH_ROWS * PATCH_COLS == CSI_SHAPE[0] * CSI_SHAPE[1] // (CSI_SHAPE[0] // PATCH_ROWS * CSI_SHAPE[1] // PATCH_COLS), \
    f"Patch数量不匹配: {PATCH_ROWS}x{PATCH_COLS} != 52"
# ==================== 配置结束 ====================


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


# ==================== CQI 嵌入模块（与 TransNet_H 相同） ====================

class CQIAlphaModulator(nn.Module):
    """CQI 频率权重 + 偏置调制器

    将13维CQI向量转换为：
        1. 52 维频率权重向量 weight[k] ∈ (0, 1)
        2. 52 维频率偏置向量 bias[k] ∈ (-∞, +∞)

    调制公式：output = weight * SR(x) + (1 - weight) * x + bias
    - weight ≈ 0, bias ≈ 0 → 输出 CSI（零改动）
    - weight ≈ 1, bias ≈ 0 → 输出 SR(x)

    参数量：13 → 32 → 52*2 = 104

    Args:
        num_subbands: CQI 向量维度 (13)
        freq_bins: 频率维度 (52)
        initial_weight: 权重初始化值 (默认0.0，恒等变换)
        initial_bias: 偏置初始化值 (默认0.0，恒等变换)
    """

    def __init__(self, num_subbands=13, freq_bins=52, initial_weight=0.0, initial_bias=0.0):
        super().__init__()
        self.num_subbands = num_subbands
        self.freq_bins = freq_bins
        self.initial_weight = initial_weight
        self.initial_bias = initial_bias

        # CQI → 频率权重 + 偏置
        hidden_dim = 32
        self.fc = nn.Sequential(
            nn.Linear(num_subbands, hidden_dim),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Linear(hidden_dim, freq_bins * 2),  # 输出 weight 和 bias
        )
        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                xavier_uniform_(m.weight, gain=0.1)
                constant_(m.bias, 0.0)
        # 恒等变换初始化
        # weight: sigmoid 输出
        #   initial_weight=0 → bias=-10 → sigmoid(-10) ≈ 0
        #   initial_weight=1 → bias=10 → sigmoid(10) ≈ 1
        # bias: 直接加到输出上
        #   initial_bias=0 → 输出为0
        with torch.no_grad():
            if self.initial_weight < 0.5:
                self.fc[-1].bias[:self.freq_bins].fill_(-10.0)  # weight 初始化为 0
            else:
                self.fc[-1].bias[:self.freq_bins].fill_(10.0)   # weight 初始化为 1
            self.fc[-1].bias[self.freq_bins:].fill_(self.initial_bias)  # bias 初始化为 0

    def forward(self, cqi: torch.Tensor) -> tuple:
        """
        Args:
            cqi: (B, num_subbands) - CQI 向量
        Returns:
            weight: (B, num_patch_rows, num_patch_cols) - Patch 频率权重
            bias: (B, num_patch_rows, num_patch_cols) - Patch 频率偏置
        """
        B = cqi.shape[0]

        # 处理 wideband 模式：扩展到 13 维
        if cqi.dim() == 2 and cqi.shape[1] == 1:
            cqi_expanded = cqi.expand(-1, self.num_subbands)
        else:
            cqi_expanded = cqi

        # CQI → 频率权重 + 偏置
        out = self.fc(cqi_expanded.float())  # (B, freq_bins * 2)

        weight_raw = out[:, :self.freq_bins]   # (B, 52)
        bias_raw = out[:, self.freq_bins:]      # (B, 52)

        weight = torch.sigmoid(weight_raw)      # (0, 1) 范围
        bias = bias_raw                         # 偏置保持原值

        # 将 52 维向量 reshape 为 num_patch_rows × num_patch_cols 的 patch 格式
        # CSI (32, 52) → (32/PATCH_ROWS)行 patch × (52/PATCH_COLS)列 patch
        # 根据配置的 PATCH_ROWS, PATCH_COLS 自动计算
        num_patch_rows = CSI_SHAPE[0] // PATCH_ROWS
        num_patch_cols = CSI_SHAPE[1] // PATCH_COLS
        weight = weight.view(B, num_patch_rows, num_patch_cols)
        bias = bias.view(B, num_patch_rows, num_patch_cols)

        return weight, bias


class LightweightSuperResolution(nn.Module):
    """多支路轻量级超分辨率增强模块（增强版）

    结构设计思路：
    - 主支路（3x3）：深层残差结构，负责捕获局部细节特征，具有最深的学习能力
    - 次要支路1（1x1）：多层感知机结构，负责捕获通道间关系，增强特征表达能力
    - 次要支路2（5x5）：浅层结构，负责捕获较大范围的上下文信息

    特点：
    - 主支路采用双层卷积+残差连接，深层特征提取
    - 次要支路1使用更宽的隐层维度，增强信息流
    - 使用可学习的融合权重（而非固定 concatenation），更好地平衡各支路贡献

         ┌─────────────────────────────────────────┐
         ↓                                         │
    输入 ──┬── 主支路: 3x3×2层+残差 ──┬──→ 加权融合 → 残差 ──→ 输出
          │                          ↑
          ├── 次支路1: 1x1 MLP ──────┤
          │                          │
          ├── 次支路2: 5x5 浅层 ──────┘
          │
          └─────── 恒等残差连接 ──────┘
    """

    def __init__(self, in_channels=2, hidden_channels=16):
        super().__init__()

        # ============ 主支路：深层残差结构 ============
        # 两层 3x3 卷积提取深层特征，残差连接保证梯度流动
        self.main_path = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
        )
        self.main_proj = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False)
        self.main_act = nn.LeakyReLU(0.3, inplace=True)

        # ============ 次支路1：1x1 MLP（捕获通道关系） ============
        # 使用更宽的隐层维度，增强通道间的信息流动
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels * 2, kernel_size=1),
            nn.BatchNorm2d(hidden_channels * 2),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv2d(hidden_channels * 2, hidden_channels, kernel_size=1),
            nn.BatchNorm2d(hidden_channels),
        )
        self.branch1_proj = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False)
        self.branch1_act = nn.LeakyReLU(0.3, inplace=True)

        # ============ 次支路2：5x5 浅层（捕获上下文） ============
        # 单层大卷积核，捕获更大范围的上下文信息
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(hidden_channels),
            nn.LeakyReLU(0.3, inplace=True),
        )
        self.branch2_proj = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False)
        self.branch2_act = nn.LeakyReLU(0.3, inplace=True)

        # ============ 融合层：可学习加权 ============
        # 将三个分支的输出按通道拼接后，通过 1x1 卷积融合
        # 融合层权重初始化为接近恒等映射（输出权重为0）
        self.fusion = nn.Sequential(
            nn.Conv2d(hidden_channels * 3, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=1, bias=False),
        )
        self.fusion_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)

        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_uniform_(m.weight)
                if m.bias is not None:
                    constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)
        # 恒等映射初始化：融合层输出权重置0
        constant_(self.fusion[-1].weight, 0.0)
        if self.fusion[-1].bias is not None:
            constant_(self.fusion[-1].bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        # 主支路：深层残差
        main_out = self.main_path(x)
        main_proj = self.main_proj(x)
        main_out = self.main_act(main_out + main_proj)  # 残差连接

        # 次支路1：1x1 MLP
        b1_out = self.branch1(x)
        b1_proj = self.branch1_proj(x)
        b1_out = self.branch1_act(b1_out + b1_proj)

        # 次支路2：5x5 浅层
        b2_out = self.branch2(x)
        b2_proj = self.branch2_proj(x)
        b2_out = self.branch2_act(b2_out + b2_proj)

        # 融合
        fused = torch.cat([main_out, b1_out, b2_out], dim=1)
        out = self.fusion(fused)

        # 恒等残差
        identity_proj = self.fusion_proj(identity)
        out = out + identity_proj

        return out


class CRNet_CQI(nn.Module):
    """CRNet with CQI embedding module for fine-tuning
    
    使用与 TransNet_H 相同的嵌入模块：CQIAlphaModulator + LightweightSuperResolution
    """

    def __init__(self, reduction=4, sr_hidden_channels=16):
        super(CRNet_CQI, self).__init__()
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

        # ==================== CQI 嵌入模块 ====================
        # 输出 52 维频率权重 + 52 维偏置，初始时为零改动（直接输出 CSI）
        self.cqi_modulator = CQIAlphaModulator(num_subbands=1, freq_bins=52,
                                                initial_weight=0.1, initial_bias=0.1)
        self.sr_module = LightweightSuperResolution(in_channels=in_channel, hidden_channels=sr_hidden_channels)

        # ==================== 解码器部分（与原始 CRNet 完全一致） ====================
        decoder = OrderedDict([
            ("conv5x5_bn", ConvBN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("CRBlock1", CRBlock()),
            ("CRBlock2", CRBlock())
        ])
        self.decoder_feature = nn.Sequential(decoder)#cqi_fc

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
            cqi: (B, CQI_dim) - CQI 向量，可选
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
        out = self.decoder_feature(out)
        # CQI 嵌入（如果提供了 CQI）
        if cqi is not None:
            B = x.shape[0]
            # 1. 超分辨率增强
            sr_out = self.sr_module(out)  # (B, 2, 32, 52)

            # 2. CQI → num_patch_rows×num_patch_cols patch 权重 + 偏置
            freq_weight, freq_bias = self.cqi_modulator(cqi)  # (B, num_patch_rows, num_patch_cols) each

            # 3. CSI (B, 2, 32, 52) → 按 patch 展开
            # CSI: (B, 2, CSI_ROWS, CSI_COLS) → patch: (B, 2, num_patch_rows, PATCH_ROWS, num_patch_cols, PATCH_COLS)
            num_patch_rows = CSI_SHAPE[0] // PATCH_ROWS
            num_patch_cols = CSI_SHAPE[1] // PATCH_COLS
            out_reshaped = out.view(B, 2, num_patch_rows, PATCH_ROWS, num_patch_cols, PATCH_COLS)
            sr_reshaped = sr_out.view(B, 2, num_patch_rows, PATCH_ROWS, num_patch_cols, PATCH_COLS)

            # 将 weight/bias 从 (B, num_patch_rows, num_patch_cols) 扩展为 (B, 1, num_patch_rows, 1, num_patch_cols, 1) 以便广播
            w = freq_weight.unsqueeze(1).unsqueeze(3).unsqueeze(5)
            b = freq_bias.unsqueeze(1).unsqueeze(3).unsqueeze(5)

            # 4. 调制：每个 patch × weight + bias
            # output = weight * SR(patch) + (1 - weight) * CSI(patch) + bias
            # 初始时 weight≈0, bias≈0 → 输出 CSI（零改动）
            out_patch = w * (sr_reshaped + b) + (1 - w) * out_reshaped

            # 恢复原始形状
            out = out_patch.view(B, 2, CSI_SHAPE[0], CSI_SHAPE[1])

        # 解码器（与原始 CRNet 一致）
        #out = self.decoder_feature(out)

        return out


def crnet_cqi(reduction=4, sr_hidden_channels=16):
    r""" Create a CRNet with CQI embedding module.

    :param reduction: the reciprocal of compression ratio
    :param sr_hidden_channels: hidden channels for super-resolution module
    :return: an instance of CRNet_CQI
    """

    model = CRNet_CQI(reduction=reduction, sr_hidden_channels=sr_hidden_channels)#cqi_fc
    return model
