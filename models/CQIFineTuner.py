import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_
import math


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
        if cqi.dim() == 2 and cqi.shape[1] == 1:
            cqi_expanded = cqi.expand(-1, self.num_subbands)
        else:
            cqi_expanded = cqi
        alpha = self.fc(cqi_expanded.float())
        alpha = torch.sigmoid(alpha)
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
        self.main_path = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=3, padding=1),
        )
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=1),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=5, padding=2),
        )
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
        main_out = self.main_path(H)
        branch1_out = self.branch1(H)
        branch2_out = self.branch2(H)
        combined = torch.cat([main_out, branch1_out, branch2_out], dim=1)
        output = self.fusion(combined)
        output = output + residual
        return output


class CQIFineTuner(nn.Module):
    """
    CQI 微调模块 - 置于 Decoder 之后
    
    设计思路（参考 TransNet_QC）：
        1. 将 CQI 映射到 32x52 空间矩阵
        2. 通过卷积网络生成特征
        3. 应用调制：H_out = H * weight + bias
    
    输入：
        csi: (B, 2, 32, 52) - Decoder 输出的 CSI
        cqi: (B, 13) - Sub CQI 或 (B, 1) - Wide CQI
    
    输出：
        output: (B, 2, 32, 52) - 调制后的 CSI
    """

    def __init__(self, mode='wide', num_subbands=13, hidden_channels=8):
        super().__init__()
        self.mode = mode
        self.num_subbands = num_subbands
        
        # CQI 特征提取卷积
        # 输入: (B, 1, 32, 52), 输出: (B, hidden_channels, 32, 52)
        self.cqi_conv = nn.Sequential(
            nn.Conv2d(1, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        # 权重和偏置生成
        self.weight_conv = nn.Conv2d(hidden_channels * 2, 2, kernel_size=1)   # (B, 2, 32, 52)
        self.bias_conv = nn.Conv2d(hidden_channels * 2, 2, kernel_size=1)     # (B, 2, 32, 52)
        
        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_uniform_(m.weight)
                if m.bias is not None:
                    constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                xavier_uniform_(m.weight)
                if m.bias is not None:
                    constant_(m.bias, 0.0)

    def _cqi_to_spatial_map(self, cqi: torch.Tensor) -> torch.Tensor:
        """
        将 CQI 向量映射到 32x52 空间矩阵
        
        Args:
            cqi: (B, 13) 或 (B, 1)
            
        Returns:
            spatial_map: (B, 32, 52)
        """
        B = cqi.shape[0]
        
        if self.mode == 'wide':
            # Wide 模式：所有元素相同，直接复制成 32x52 矩阵
            spatial_map = cqi[:, 0:1].unsqueeze(-1).expand(-1, 32, 52)
        elif self.mode == 'sub':
            # Sub 模式：13个值映射到 32x52 矩阵
            # 每4列对应一个CQI值：列0-3 -> cqi[0], ..., 列48-51 -> cqi[12]
            spatial_map = torch.zeros(B, 32, 52, device=cqi.device, dtype=cqi.dtype)
            
            for i in range(self.num_subbands):
                start_col = i * 4
                end_col = min((i + 1) * 4, 52)
                spatial_map[:, :, start_col:end_col] = cqi[:, i:i+1].unsqueeze(-1)
        else:
            raise ValueError(f"不支持的CQI模式: {self.mode}")
        
        return spatial_map

    def forward(self, csi: torch.Tensor, cqi: torch.Tensor) -> torch.Tensor:
        """
        Args:
            csi: (B, 2, 32, 52) - Decoder 输出的 CSI
            cqi: (B, 13) - Sub CQI 或 (B, 1) - Wide CQI
            
        Returns:
            output: (B, 2, 32, 52) - 调制后的 CSI
        """
        B, C, H_dim, W_dim = csi.shape
        
        # 1. CQI 映射到空间矩阵
        cqi_spatial = self._cqi_to_spatial_map(cqi)  # (B, 32, 52)
        cqi_spatial = cqi_spatial.unsqueeze(1)       # (B, 1, 32, 52)
        
        # 2. CQI 特征提取
        cqi_features = self.cqi_conv(cqi_spatial)     # (B, 16, 32, 52)
        
        # 3. 生成权重和偏置
        weight = self.weight_conv(cqi_features)      # (B, 2, 32, 52)
        bias = self.bias_conv(cqi_features)         # (B, 2, 32, 52)
        
        # 4. 应用调制: H_out = H * weight + bias
        output = (csi * weight + bias)*0.0001 + csi
        #output = csi
        
        return output


class CQIAlphaSR(nn.Module):
    """CQI Alpha + Super-Resolution 微调模块（来自 TransNet_H 的设计）
    
    设计思路：
        1. CQIAlphaModulator: 将 CQI 向量转换为因子 α（0~1之间）
        2. LightweightSuperResolution: 轻量级超分辨率增强
    
    输入：
        csi: (B, 2, 32, 52) - Decoder 输出的 CSI
        cqi: (B, 13) - Sub CQI 或 (B, 1) - Wide CQI
    
    输出：
        output: (B, 2, 32, 52) - 增强后的 CSI
    """
    
    def __init__(self, mode='wide', num_subbands=13, sr_hidden_channels=16):
        super().__init__()
        self.mode = mode
        self.num_subbands = num_subbands
        
        # CQI -> alpha 调制器
        self.cqi_modulator = CQIAlphaModulator(num_subbands=num_subbands, initial_value=0.1)
        
        # 轻量级超分辨率模块
        self.sr_module = LightweightSuperResolution(in_channels=2, hidden_channels=sr_hidden_channels)
        
        # 可学习的输出缩放因子
        self.output_scale = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, csi: torch.Tensor, cqi: torch.Tensor) -> torch.Tensor:
        """Args:
            csi: (B, 2, 32, 52) - Decoder 输出的 CSI
            cqi: (B, 13) - Sub CQI 或 (B, 1) - Wide CQI
            
        Returns:
            output: (B, 2, 32, 52) - 增强后的 CSI
        """
        # 1. CQI 转换为 alpha 因子
        alpha = self.cqi_modulator(cqi)  # (B, 1, 1, 1)
        
        # 2. 超分辨率增强
        sr_features = self.sr_module(csi)  # (B, 2, 32, 52)
        
        # 3. 融合: H_out = H + alpha * SR(H)
        output = csi + alpha * sr_features
        output = output * self.output_scale
        
        return output


class CRNetWithFineTuner(nn.Module):
    """CRNet + CQI 微调模块"""
    
    def __init__(self, reduction=4, use_cqi_tuner=False, cqi_mode='wide'):
        super().__init__()
        from .crnet import CRNet
        
        self.base_model = CRNet(reduction=reduction)
        self.use_cqi_tuner = use_cqi_tuner
        
        if use_cqi_tuner:
            self.cqi_tuner = CQIFineTuner(mode=cqi_mode)
        
    def forward(self, x, cqi=None):
        # 基础模型前向
        out = self.base_model(x)
        
        # CQI 微调
        if self.use_cqi_tuner and cqi is not None:
            out = self.cqi_tuner(out, cqi)
        
        return out


class CLNetWithFineTuner(nn.Module):
    """CLNet + CQI 微调模块"""
    
    def __init__(self, reduction=4, use_cqi_tuner=False, cqi_mode='wide'):
        super().__init__()
        from .clnet import CLNet
        
        self.base_model = CLNet(reduction=reduction)
        self.use_cqi_tuner = use_cqi_tuner
        
        if use_cqi_tuner:
            self.cqi_tuner = CQIFineTuner(mode=cqi_mode)
        
    def forward(self, x, cqi=None):
        out = self.base_model(x)
        
        if self.use_cqi_tuner and cqi is not None:
            out = self.cqi_tuner(out, cqi)
        
        return out


class TransNetWithFineTuner(nn.Module):
    """TransNet + CQI 微调模块"""
    
    def __init__(self, reduction=64, d_model=64, nhead=8, num_encoder_layers=2, 
                 num_decoder_layers=2, mode='wide', use_cqi_tuner=False):
        super().__init__()
        from .TransNet_QModPlusD import Transformer_QModPlusD
        
        self.base_model = Transformer_QModPlusD(
            cqi_vocab_size=16, mode=mode, d_model=d_model, nhead=nhead,
            num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers,
            reduction=reduction
        )
        self.use_cqi_tuner = use_cqi_tuner
        
        if use_cqi_tuner:
            self.cqi_tuner = CQIFineTuner(mode=mode)
        
    def forward(self, src, cqi=None, tgt=None, **kwargs):
        out = self.base_model(src, cqi=cqi, tgt=tgt, **kwargs)
        
        if self.use_cqi_tuner and cqi is not None:
            # TransNet 输出是 (B, 2, 32, 52)，需要与 cqi 一起输入
            out = self.cqi_tuner(out, cqi)
        
        return out
