r""" TransNet_QModPlusD with CQI embedding module for fine-tuning
"""

import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_, constant_
from typing import Optional
import math


# ==================== CQI 嵌入模块（与 TransNet_H 相同） ====================

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
        xavier_uniform_(self.fc.weight, gain=1e-2)
        if self.initial_value < 0.5:
            bias = math.log(self.initial_value / (1 - self.initial_value))
        else:
            bias = -3.0
        constant_(self.fc.bias, bias)

    def forward(self, cqi: torch.Tensor) -> torch.Tensor:
        B = cqi.shape[0]

        # 处理 wideband 模式：扩展到 13 维
        if cqi.dim() == 2 and cqi.shape[1] == 1:
            cqi_expanded = cqi.expand(-1, self.num_subbands)
        else:
            cqi_expanded = cqi

        alpha = self.fc(cqi_expanded.float())  # (B, 1)
        alpha = torch.sigmoid(alpha)  # (0, 1) 范围
        alpha = alpha.view(B, 1, 1, 1)

        return alpha


class LightweightSuperResolution(nn.Module):
    """多支路轻量级超分辨率增强模块

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
        # 融合层初始化：使初始输出接近恒等映射
        # 将 fusion 层的输出通道权重初始化为 0，相当于 out ≈ 0 + identity = identity
        constant_(self.fusion.weight, 0.0)
        constant_(self.fusion.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        # 三条路径
        main_out = self.main_path(x)
        branch1_out = self.branch1(x)
        branch2_out = self.branch2(x)

        # 融合
        fused = torch.cat([main_out, branch1_out, branch2_out], dim=1)
        out = self.fusion(fused)

        # 残差连接
        out = out + identity

        return out


class TransNet_QModPlusD_CQI(nn.Module):
    """TransNet_QModPlusD with CQI embedding module for fine-tuning
    
    使用与 TransNet_H 相同的嵌入模块：CQIAlphaModulator + LightweightSuperResolution
    """

    def __init__(self, mode='wide', d_model: int = 64, nhead: int = 8, num_encoder_layers: int = 2,
                 num_decoder_layers: int = 2, dim_feedforward: int = 2048, dropout: float = 0.0,
                 activation = torch.nn.functional.relu, layer_norm_eps: float = 1e-5, 
                 batch_first: bool = False, reduction=64, sr_hidden_channels=16):
        super().__init__()
        
        # ==================== 原始 TransNet_QModPlusD（完全一致） ====================
        from models.TransNet_QModPlusD import (
            Transformer_QModPlusD,
            TransformerEncoderLayer_QModPlusD,
            TransformerEncoder_QModPlusD,
            TransformerDecoderLayer_QModPlusD,
            TransformerDecoder_QModPlusD
        )
        
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

        # 支持近似压缩率
        target_compressed_dim = 3328 // reduction
        self.compressed_dim = max(1, target_compressed_dim)
        self.actual_reduction = 3328 / self.compressed_dim

        self.fc_encoder = nn.Linear(3328, self.compressed_dim)
        self.fc_decoder = nn.Linear(self.compressed_dim, 3328)
        self._reset_parameters()
        self.output_scale = nn.Parameter(torch.tensor(1.0))

        self.transformer = nn.Module()
        self.transformer.encoder = self.encoder
        self.transformer.decoder = self.decoder

        # ==================== CQI 嵌入模块（与 TransNet_H 相同） ====================
        self.cqi_modulator = CQIAlphaModulator(num_subbands=13)
        self.sr_module = LightweightSuperResolution(in_channels=2, hidden_channels=sr_hidden_channels)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, src: torch.Tensor, cqi: Optional[torch.Tensor] = None, 
                wideband_cqi: Optional[torch.Tensor] = None,
                subband_cqi: Optional[torch.Tensor] = None,
                tgt: Optional[torch.Tensor] = None, src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None, memory_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None, tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        
        # 处理 CQI 参数
        input_cqi = cqi
        if input_cqi is None:
            if wideband_cqi is not None:
                input_cqi = wideband_cqi
            elif subband_cqi is not None:
                input_cqi = subband_cqi
        if input_cqi is None:
            if 'cqi' in kwargs:
                input_cqi = kwargs['cqi']
            elif 'wideband_cqi' in kwargs:
                input_cqi = kwargs['wideband_cqi']
            elif 'subband_cqi' in kwargs:
                input_cqi = kwargs['subband_cqi']
        
        # ==================== 原始 TransNet_QModPlusD 前向 ====================
        B = src.shape[0]
        x = src.permute(0, 1, 3, 2).contiguous()
        src_reshaped = x.view(B, self.feature_shape[0], self.feature_shape[1])

        memory = self.encoder(src_reshaped.transpose(0, 1), None, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        memory_encoder = self.fc_encoder(memory.transpose(0, 1).contiguous().view(B, -1))
        memory_decoder = self.fc_decoder(memory_encoder).view(B, self.feature_shape[0], self.feature_shape[1]).transpose(0, 1)
        output = self.decoder(memory_decoder, memory_decoder, None, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
        output = output.transpose(0, 1).contiguous()
        output = output.view(B, 2, 52, 32).permute(0, 1, 3, 2).contiguous()
        output = self.output_scale * output

        # ==================== CQI 嵌入（与 TransNet_H 相同） ====================
        if input_cqi is not None:
            # 1. CQI → α 调制因子
            alpha = self.cqi_modulator(input_cqi)  # (B, 1, 1, 1)

            # 2. 超分辨率增强
            sr_out = self.sr_module(output)  # (B, 2, 32, 52)

            # 3. 调制：output = α * SR(x) + (1-α) * x
            output = alpha * sr_out + (1 - alpha) * output
        
        return output

    def get_model_info(self):
        total_params = sum(p.numel() for p in self.parameters())
        cqi_params = sum(p.numel() for p in self.cqi_modulator.parameters()) + sum(p.numel() for p in self.sr_module.parameters())
        info = {
            'use_cqi_fusion': True,
            'fusion_params': cqi_params,
            'cqi_embedding_params': cqi_params,
            'total_params': total_params,
            'fusion_ratio': cqi_params / max(1, total_params),
        }
        return info


def transnet_qmodplusd_cqi(reduction=64, d_model=64, mode='wide', num_encoder_layers=2, 
                          num_decoder_layers=2, nhead=2, dropout=0.0, sr_hidden_channels=16):
    r""" Create a TransNet_QModPlusD with CQI embedding module.

    :param reduction: the reciprocal of compression ratio
    :param d_model: model dimension
    :param mode: CQI mode ('wide' or 'sub')
    :param num_encoder_layers: number of encoder layers
    :param num_decoder_layers: number of decoder layers
    :param nhead: number of attention heads
    :param dropout: dropout rate
    :param sr_hidden_channels: hidden channels for super-resolution module
    :return: an instance of TransNet_QModPlusD_CQI
    """
    model = TransNet_QModPlusD_CQI(
        mode=mode, d_model=d_model, num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers, nhead=nhead, dropout=dropout, 
        reduction=reduction, sr_hidden_channels=sr_hidden_channels
    )
    return model
