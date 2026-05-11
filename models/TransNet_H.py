import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_
from typing import Optional
import math
from .TransNet_QModPlusD import (
    Transformer_QModPlusD,
    TransformerEncoderLayer_QModPlusD,
    TransformerEncoder_QModPlusD,
    TransformerDecoderLayer_QModPlusD,
    TransformerDecoder_QModPlusD
)


class CQIAlphaModulator(nn.Module):
    """CQI 调制器：将13维CQI向量转换为因子α（0~1之间）

    使用 sigmoid 激活确保 α ∈ (0, 1)
    通过设置初始偏置控制初始值较小
    """

    def __init__(self, num_subbands=13, initial_value=0.1):
        super().__init__()
        self.num_subbands = num_subbands
        self.fc = nn.Linear(num_subbands, 1)
        self.initial_value = initial_value
        self._reset_parameters()

    def _reset_parameters(self):
        # 初始化权重为很小的值
        xavier_uniform_(self.fc.weight, gain=1e-2)
        # 初始偏置设为负值，使得初始 alpha ≈ self.initial_value
        # sigmoid(x) ≈ initial_value => x ≈ log(initial_value / (1 - initial_value))
        if self.initial_value < 0.5:
            bias = math.log(self.initial_value / (1 - self.initial_value))
        else:
            bias = -3.0  # 默认下限
        constant_(self.fc.bias, bias)

    def forward(self, cqi: torch.Tensor) -> torch.Tensor:
        B = cqi.shape[0]

        # 处理 wideband 模式：扩展到 13 维
        if cqi.dim() == 2 and cqi.shape[1] == 1:
            cqi_expanded = cqi.expand(-1, self.num_subbands)
        else:
            cqi_expanded = cqi

        alpha = self.fc(cqi_expanded.float())  # (B, 1)
        alpha = torch.sigmoid(alpha)  # (0, 1) 范围，确保为正且不大于1
        alpha = alpha.view(B, 1, 1, 1)

        return alpha


class LightweightSuperResolution(nn.Module):
    """
    多支路轻量级超分辨率增强模块（基站端）

    设计思路：多路径残差结构
    - 主路径：3×3 卷积，处理细节
    - 支路1：1×1 卷积，处理全局信息
    - 支路2：5×5 大卷积核，扩大感受野
    - 残差连接：输入直接加到输出，保证不劣化

    结构：
         ┌─────────────────────────────────────┐
         ↓                                      │
    输入 ──┼── 主路: Conv3×3 ──┬──→ 融合 ──→ 输出
         │                  ↑
         └── 支路1: Conv1×1 ─┘
         │
         └── 支路2: Conv5×5 ──┘
         │
         └────── 残差连接 ───┘
    """

    def __init__(self, in_channels=2, hidden_channels=16):
        super().__init__()

        # 主路径：3×3 卷积
        self.main_path = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=3, padding=1),
        )

        # 支路1：1×1 卷积（全局信息）
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=1),
        )

        # 支路2：5×5 大卷积核（扩大感受野）
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=5, padding=2),
        )

        # 融合层
        self.fusion = nn.Conv2d(in_channels * 3, in_channels, kernel_size=1)

        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_uniform_(m.weight)
                if m.bias is not None:
                    constant_(m.bias, 0.0)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        residual = H

        # 三条路径并行处理
        main_out = self.main_path(H)      # (B, 2, 32, 52)
        branch1_out = self.branch1(H)      # (B, 2, 32, 52)
        branch2_out = self.branch2(H)      # (B, 2, 32, 52)

        # 融合三条路径
        combined = torch.cat([main_out, branch1_out, branch2_out], dim=1)  # (B, 6, 32, 52)
        output = self.fusion(combined)  # (B, 2, 32, 52)

        # 残差连接
        output = output + residual

        return output


class Transformer_H(Transformer_QModPlusD):
    """消融实验模型：继承TransNet_QModPlusD，但编码器/译码器不使用CQI

    设计思路：
    - 继承 TransNet_QModPlusD 的架构
    - 编码器/译码器：传入 cqi=None（相当于标准TransNet，无CQI引导注意力）
    - 基站端：添加与QH一致的轻量级超分辨率增强（接收正常CQI输入）

    消融实验对比：
    - TransNet_QModPlusD: 原始模型（CQI引导注意力 + 无基站端增强）
    - TransNet_H: 无CQI引导注意力 + 基站端超分辨率增强
    - TransNet_QH: CQI引导注意力 + 基站端超分辨率增强

    这样可以公平对比：
    1. CQI引导注意力的贡献
    2. 基站端超分辨率增强的贡献
    """

    def __init__(self, mode='wide', d_model: int = 64, nhead: int = 8, num_encoder_layers: int = 2,
                 num_decoder_layers: int = 2, dim_feedforward: int = 2048, dropout: float = 0.0,
                 activation=F.relu, layer_norm_eps: float = 1e-5, batch_first: bool = False,
                 reduction=64, sr_hidden_channels=16) -> None:

        super(Transformer_QModPlusD, self).__init__()

        # 初始化父类组件（与TransNet_QModPlusD相同）
        encoder_layer = TransformerEncoderLayer_QModPlusD(d_model, nhead, mode=mode, dim_feedforward=dim_feedforward,
                                                          dropout=dropout, activation=activation, layer_norm_eps=layer_norm_eps, batch_first=batch_first)
        self.encoder = TransformerEncoder_QModPlusD(encoder_layer, num_encoder_layers, norm=nn.LayerNorm(d_model, eps=layer_norm_eps))
        decoder_layer = TransformerDecoderLayer_QModPlusD(d_model, nhead, mode=mode, dim_feedforward=dim_feedforward,
                                                          dropout=dropout, activation=activation, layer_norm_eps=layer_norm_eps, batch_first=batch_first)
        self.decoder = TransformerDecoder_QModPlusD(decoder_layer, num_decoder_layers, norm=nn.LayerNorm(d_model, eps=layer_norm_eps))
        self.d_model = d_model
        assert not (3328 % self.d_model)
        self.feature_shape = (3328 // self.d_model, self.d_model)
        self.batch_first = batch_first

        target_compressed_dim = 3328 // reduction
        self.compressed_dim = max(1, target_compressed_dim)
        self.actual_reduction = 3328 / self.compressed_dim

        print(f"目标压缩率: 1/{reduction:.1f}, 实际压缩率: 1/{self.actual_reduction:.1f} (维度: {self.compressed_dim})")

        self.fc_encoder = nn.Linear(3328, self.compressed_dim)
        self.fc_decoder = nn.Linear(self.compressed_dim, 3328)
        self._reset_parameters()
        self.output_scale = nn.Parameter(torch.tensor(1.0))

        self.transformer = nn.Module()
        self.transformer.encoder = self.encoder
        self.transformer.decoder = self.decoder

        # 基站端超分辨率增强模块
        self.cqi_modulator = CQIAlphaModulator(num_subbands=13)
        self.sr_module = LightweightSuperResolution(in_channels=2, hidden_channels=sr_hidden_channels)

        # 历史记录
        self._alpha_history = []
        self._alpha_save_dir = None  # 保存目录
        self._alpha_print_freq = 50  # 每多少个 batch 打印一次 alpha

    def forward(self, src: torch.Tensor, cqi: Optional[torch.Tensor] = None, tgt: Optional[torch.Tensor] = None, src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None, memory_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None, tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:

        B = src.shape[0]

        # ========== 编码器/译码器处理（关键：不使用CQI，相当于标准TransNet） ==========
        x = src.permute(0, 1, 3, 2).contiguous()
        src_reshaped = x.view(B, self.feature_shape[0], self.feature_shape[1])

        # 传入 cqi=None，禁用 CQI 引导注意力机制
        memory = self.encoder(src_reshaped.transpose(0, 1), cqi=None, mask=src_mask, src_key_padding_mask=src_key_padding_mask)

        # 不缓存 corr_data，因为没有使用 CQI
        self.corr_cache = None

        memory_encoder = self.fc_encoder(memory.transpose(0, 1).contiguous().view(B, -1))
        memory_decoder = self.fc_decoder(memory_encoder).view(B, self.feature_shape[0], self.feature_shape[1]).transpose(0, 1)

        # 译码器也传入 cqi=None
        output = self.decoder(memory_decoder, memory_decoder, cqi=None, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)

        output = output.transpose(0, 1).contiguous()
        output = output.view(B, 2, 52, 32).permute(0, 1, 3, 2).contiguous()  # (B, 2, 32, 52)

        # ========== 超分辨率增强（基站端） ==========
        if cqi is not None:
            # 1. CQI → α 调制因子
            alpha = self.cqi_modulator(cqi)  # (B, 1, 1, 1)

            # 2. F(H) 超分辨率增强残差
            sr_residual = self.sr_module(output)  # (B, 2, 32, 52)

            # 3. H_enhanced = H + α * F(H)
            output = output + alpha * sr_residual

            # 记录 alpha 值
            if self.training:
                alpha_val = alpha.mean().item()
                self._alpha_history.append(alpha_val)

                # 实时打印 alpha 值（每50个batch打印一次）
                batch_idx = len(self._alpha_history) - 1
                if (batch_idx + 1) % self._alpha_print_freq == 0:
                    print(f"  [BS] Batch {batch_idx + 1}: α = {alpha_val:.6f}")

        output = self.output_scale * output

        return output

    def get_model_info(self):
        total_params = sum(p.numel() for p in self.parameters())
        linear_params = sum(p.numel() for p in self.fc_encoder.parameters()) + sum(p.numel() for p in self.fc_decoder.parameters())

        # 基站端模块参数
        sr_params = sum(p.numel() for p in self.sr_module.parameters()) + sum(p.numel() for p in self.cqi_modulator.parameters())

        info = {
            'use_cqi_attention': False,   # 无CQI引导注意力
            'use_sr_enhancement': True,   # 有基站端超分辨率增强
            'use_ue_preprocessing': False, # 无用户端微调
            'total_params': total_params,
            'linear_params': linear_params,
            'sr_params': sr_params,
            'alpha_history': self._alpha_history.copy() if self._alpha_history else [],
        }
        return info

    def init_alpha_save_dir(self, save_dir: str):
        """初始化 alpha 历史保存目录

        Args:
            save_dir: 保存目录（与模型保存目录一致）
        """
        import os
        from datetime import datetime

        self._alpha_save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        filepath = os.path.join(save_dir, 'alpha_history.txt')
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # 创建/清空文件并写入表头
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"{'='*60}\n")
            f.write(f"# TransNet_H Alpha History\n")
            f.write(f"# Model: 无CQI引导注意力 + 基站端超分辨率增强\n")
            f.write(f"# 创建时间: {timestamp}\n")
            f.write(f"{'='*60}\n\n")

    def set_alpha_print_freq(self, freq: int):
        """设置 alpha 实时打印频率

        Args:
            freq: 每多少个 batch 打印一次（设为 0 则不打印）
        """
        self._alpha_print_freq = freq

    def save_alpha_history(self, epoch: int = None):
        """保存本轮 alpha 历史记录到统一的 txt 文件（追加模式）

        Args:
            epoch: 当前 epoch 编号
        """
        import os
        from datetime import datetime

        if self._alpha_save_dir is None:
            print("警告: 请先调用 init_alpha_save_dir() 设置保存目录")
            return

        if not self._alpha_history:
            print("警告: alpha_history 为空，没有记录可保存")
            return

        filepath = os.path.join(self._alpha_save_dir, 'alpha_history.txt')
        timestamp = datetime.now().strftime('%H:%M:%S')

        with open(filepath, 'a', encoding='utf-8') as f:
            # 写入分隔线和本轮信息
            f.write(f"\n{'='*40}\n")
            f.write(f"Epoch: {epoch} | Time: {timestamp}\n")
            f.write(f"Batches: {len(self._alpha_history)}\n")
            f.write(f"Mean: {sum(self._alpha_history) / len(self._alpha_history):.6f} | ")
            f.write(f"Range: [{min(self._alpha_history):.6f}, {max(self._alpha_history):.6f}]\n")
            f.write(f"{'='*40}\n")

            # 写入每个 batch 的 alpha 值
            for i, alpha in enumerate(self._alpha_history):
                f.write(f"{alpha:.6f}\n")

        print(f"Alpha history 已保存到: {filepath}")
        print(f"  - Epoch: {epoch}, Batches: {len(self._alpha_history)}")

    def clear_alpha_history(self):
        """清空 alpha 历史记录（每个 epoch 开始时调用）"""
        self._alpha_history.clear()


def create_transnet_h(reduction=64, d_model=64, mode='wide', num_encoder_layers=2, num_decoder_layers=2,
                      nhead=2, dropout=0.0, sr_hidden_channels=16):
    """创建消融实验模型：标准Transformer编码器/译码器 + 基站端超分辨率增强

    Args:
        reduction: 压缩比
        d_model: 模型维度
        mode: CQI模式 ('wide' 或 'sub')
        num_encoder_layers: 编码器层数
        num_decoder_layers: 译码器层数
        nhead: 注意力头数
        dropout: Dropout比例
        sr_hidden_channels: 超分辨率模块的隐藏通道数（默认16，轻量级）

    Returns:
        TransNet_H 模型

    消融实验对比：
    - TransNet_QModPlusD: CQI引导注意力 + 无基站端增强
    - TransNet_H: 无CQI引导注意力 + 基站端超分辨率增强
    - TransNet_QH: CQI引导注意力 + 基站端超分辨率增强

    这样可以分别评估：
    1. CQI引导注意力机制的贡献
    2. 基站端超分辨率增强的贡献
    """
    model = Transformer_H(mode=mode, d_model=d_model, num_encoder_layers=num_encoder_layers,
                         num_decoder_layers=num_decoder_layers, nhead=nhead, dropout=dropout,
                         reduction=reduction, sr_hidden_channels=sr_hidden_channels)
    return model
