import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_uniform_, constant_, xavier_normal_
from typing import Optional, Tuple, Any, List
import math
import warnings

__all__ = ["TransNet_MoE", "Transformer_MoE", "Transformer_C_MoE"]

Tensor = torch.Tensor


class CQIRouter(nn.Module):
    """
    CQI路由器，根据CQI值生成专家网络权重
    支持wide_cqi（单个值）和sub_cqi（8个值）两种模式
    """
    
    def __init__(self, cqi_vocab_size: int = 16, num_experts: int = 8, 
                 hidden_dim: int = 64, mode: str = 'wide'):
        super(CQIRouter, self).__init__()
        self.cqi_vocab_size = cqi_vocab_size
        self.num_experts = num_experts
        self.mode = mode  # 'wide' or 'sub'
        self.hidden_dim = hidden_dim
        
        if mode == 'wide':
            # Wide模式：单个CQI值生成权重
            self.cqi_embedding = nn.Embedding(cqi_vocab_size, hidden_dim)
            # 注意：移除Softmax，改为在forward里手动加偏置后再Softmax
            self.router_net = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, num_experts)
            )
        elif mode == 'sub':
            # Sub模式：8个CQI值生成权重
            self.cqi_embedding = nn.Embedding(cqi_vocab_size, hidden_dim)
            # 同样移除Softmax，forward中统一处理
            self.router_net = nn.Sequential(
                nn.Linear(hidden_dim * 8, hidden_dim),  # 8个CQI嵌入拼接
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, num_experts)
            )
        else:
            raise ValueError("Mode must be 'wide' or 'sub'")

        # 额外的初始化偏置表：为每个CQI提供一行针对各专家的logit偏置
        # 用于产生“故意差异很大”的初始专家权重分布
        self.init_bias_strength = 1.5  # 可调强度，值越大初始越尖锐
        self.router_bias_table = nn.Parameter(torch.zeros(cqi_vocab_size, num_experts))
        # 将相邻的CQI分到同一专家（例如 0-1→E0，2-3→E1，…，14-15→E7）
        self.cqi_group_size = 2
        self._init_router_bias_table()

    def _init_router_bias_table(self):
        """初始化每个CQI的专家logit偏置，制造强非均匀分布"""
        with torch.no_grad():
            # 形状: (cqi_vocab_size, num_experts)
            bias = torch.full_like(self.router_bias_table, -self.init_bias_strength)
            for cqi_idx in range(self.cqi_vocab_size):
                # 分组分配：相邻的CQI映射到同一专家，增强局部连续性
                expert_idx = (cqi_idx // self.cqi_group_size) % self.num_experts
                bias[cqi_idx, expert_idx] = self.init_bias_strength
            # 可选：加入微小噪声，避免完全对称
            bias = bias + 0.01 * torch.randn_like(bias)
            self.router_bias_table.copy_(bias)
    
    def forward(self, cqi: torch.Tensor):
        """
        Args:
            cqi: CQI张量
                - Wide模式: (batch_size,) 或 (batch_size, 1)
                - Sub模式: (batch_size, 8)
        Returns:
            expert_weights: (batch_size, num_experts) 专家权重
        """
        if self.mode == 'wide':
            # Wide模式处理
            if cqi.dim() > 1:
                cqi = cqi.squeeze(-1)  # (batch_size, 1) -> (batch_size,)
            
            # 确保CQI值在有效范围内
            cqi = torch.clamp(cqi.long(), 0, self.cqi_vocab_size - 1)
            
            # 嵌入与路由，得到logits
            cqi_emb = self.cqi_embedding(cqi)  # (batch_size, hidden_dim)
            logits = self.router_net(cqi_emb)  # (batch_size, num_experts)
            # 加入每个CQI的专属偏置（放大初始差异）
            bias = self.router_bias_table[cqi]  # (batch_size, num_experts)
            logits = logits + bias
            # 归一化
            expert_weights = F.softmax(logits, dim=-1)
            
        elif self.mode == 'sub':
            # Sub模式处理
            if cqi.dim() == 1:
                # 如果输入是1D，假设是8个连续值
                cqi = cqi.view(-1, 8)  # (batch_size*8) -> (batch_size, 8)
            elif cqi.dim() > 2:
                cqi = cqi.view(cqi.size(0), -1)  # 展平多余维度
            
            # 确保CQI值在有效范围内
            cqi = torch.clamp(cqi.long(), 0, self.cqi_vocab_size - 1)
            
            # 嵌入8个CQI值
            cqi_emb_8 = self.cqi_embedding(cqi)  # (batch_size, 8, hidden_dim)
            cqi_emb = cqi_emb_8.view(cqi_emb_8.size(0), -1)  # (batch_size, 8*hidden_dim)
            # 路由得到logits
            logits = self.router_net(cqi_emb)  # (batch_size, num_experts)
            # 将8个CQI对应的偏置做平均（或和）作为该样本的额外偏置
            bias = self.router_bias_table[cqi]  # (batch_size, 8, num_experts)
            bias = bias.mean(dim=1)  # (batch_size, num_experts)
            logits = logits + bias
            expert_weights = F.softmax(logits, dim=-1)
        
        return expert_weights
    
    def get_cqi_expert_matrix(self):
        """
        获取所有CQI值对应的专家权重矩阵
        Returns:
            cqi_expert_matrix: (cqi_vocab_size, num_experts) 每个CQI对应的专家权重
        """
        if self.mode != 'wide':
            # 只在wide模式下有效
            return None
        
        # 创建所有CQI值（0到cqi_vocab_size-1）
        all_cqi = torch.arange(self.cqi_vocab_size).to(next(self.parameters()).device)
        
        # 获取每个CQI对应的专家权重
        with torch.no_grad():
            cqi_emb = self.cqi_embedding(all_cqi)
            logits = self.router_net(cqi_emb)
            logits = logits + self.router_bias_table  # 加入偏置
            expert_weights = F.softmax(logits, dim=-1)
        
        return expert_weights.cpu().numpy()
    
    def get_cqi_top_k_experts(self, top_k=None):
        """
        获取每个CQI值对应的Top-K专家序号
        Returns:
            cqi_top_k_experts: (cqi_vocab_size, top_k) 每个CQI激活的专家序号列表
        """
        if self.mode != 'wide':
            return None
        
        top_k = top_k if top_k is not None else self.num_experts
        
        # 创建所有CQI值
        all_cqi = torch.arange(self.cqi_vocab_size).to(next(self.parameters()).device)
        
        # 获取每个CQI对应的专家权重
        with torch.no_grad():
            cqi_emb = self.cqi_embedding(all_cqi)
            logits = self.router_net(cqi_emb)
            logits = logits + self.router_bias_table
            expert_weights = F.softmax(logits, dim=-1)
        
        # 获取Top-K专家序号
        top_k_weights, top_k_indices = torch.topk(expert_weights, k=min(top_k, self.num_experts), dim=-1)
        
        return top_k_indices.cpu().numpy()


class MoEFFN(nn.Module):
    """
    基于MoE的前馈网络，替换原有的FFN层
    每个专家网络的结构与原始FFN相同
    """
    
    def __init__(self, d_model: int, dim_feedforward: int, num_experts: int,
                 cqi_vocab_size: int = 16, mode: str = 'wide', 
                 dropout: float = 0.1, activation=F.relu, top_k: int = None,
                 expert_hidden_dim: int = None, base_alpha: float = 0.9):
        super(MoEFFN, self).__init__()
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.num_experts = num_experts
        self.mode = mode
        self.dropout = dropout
        self.activation = activation
        self.top_k = top_k if top_k is not None else num_experts  # Top-K稀疏激活
        # 轻量专家的隐藏维度（默认使用原FFN隐藏维度的1/8，向上取至少等于d_model*2）
        if expert_hidden_dim is None:
            default_ehd = max(d_model * 2, dim_feedforward // 8)
            self.expert_hidden_dim = default_ehd
        else:
            self.expert_hidden_dim = expert_hidden_dim
        # 主路径（原TransNet FFN）占比
        self.base_alpha = base_alpha
        
        # CQI路由器
        self.router = CQIRouter(cqi_vocab_size, num_experts, d_model, mode)
        
        # 可学习的噪声参数：用于在训练时增强探索能力
        # 这个参数控制添加到专家权重上的高斯噪声的强度
        # 初始化为一个小的正值，允许所有专家都有机会被选择
        self.noise_scale = nn.Parameter(torch.tensor(0.1))
        
        # 主路径：保持与 TransNet FFN 一致的结构与维度
        self.base_linear1 = nn.Linear(d_model, dim_feedforward)
        self.base_dropout = nn.Dropout(dropout)
        self.base_linear2 = nn.Linear(dim_feedforward, d_model)

        # 专家网络：轻量FFN（隐藏维度更小，降低参数开销）
        self.experts = nn.ModuleList([
            nn.ModuleDict({
                'linear1': nn.Linear(d_model, self.expert_hidden_dim),
                'linear2': nn.Linear(self.expert_hidden_dim, d_model),
                'dropout': nn.Dropout(dropout)
            }) for _ in range(num_experts)
        ])
        
        # 门控网络（已移除，只使用CQI指导的专家选择）
        # self.gate_net = nn.Sequential(
        #     nn.Linear(d_model, d_model),
        #     nn.GELU(),
        #     nn.Linear(d_model, num_experts),
        #     nn.Softmax(dim=-1)
        # )
        
        # 输出投影（可选，但标准FFN没有这一步，先注释掉）
        # self.output_proj = nn.Linear(d_model, d_model)
        
        # 残差权重（标准FFN不包含这个，由外部Transformer层处理）
        # self.residual_weight = nn.Parameter(torch.tensor(0.8))
        
        # 初始化：专家权重使用较小的方差
        self._init_experts()
        
    def _init_experts(self):
        """初始化主路径与专家网络参数"""
        # 主路径（与TransNet.py一致的FFN初始化）
        nn.init.xavier_uniform_(self.base_linear1.weight)
        nn.init.zeros_(self.base_linear1.bias)
        nn.init.xavier_uniform_(self.base_linear2.weight)
        nn.init.zeros_(self.base_linear2.bias)

        # 专家网络：使用Xavier初始化（轻量隐藏维度）
        for expert in self.experts:
            nn.init.xavier_uniform_(expert['linear1'].weight)
            nn.init.zeros_(expert['linear1'].bias)
            nn.init.xavier_uniform_(expert['linear2'].weight)
            nn.init.zeros_(expert['linear2'].bias)
        
        # 路由器：纯随机初始化
        # 初始化embedding层
        nn.init.xavier_uniform_(self.router.cqi_embedding.weight)
        
        # 初始化router_net（所有层使用标准初始化）
        for module in self.router.router_net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # 初始化门控网络（已移除，不需要）
        # for module in self.gate_net.modules():
        #     if isinstance(module, nn.Linear):
        #         nn.init.xavier_uniform_(module.weight)
        #         if module.bias is not None:
        #             nn.init.zeros_(module.bias)
        
        # 初始化输出投影（已移除，不需要）
        # nn.init.xavier_uniform_(self.output_proj.weight)
        # nn.init.zeros_(self.output_proj.bias)
        
    def forward(self, x: torch.Tensor, cqi: torch.Tensor):
        """
        Args:
            x: 输入张量 (seq_len, batch_size, d_model)
            cqi: CQI张量，根据mode不同格式不同
        Returns:
            output: 输出张量 (seq_len, batch_size, d_model)
        """
        seq_len, batch_size, d_model = x.shape
        
        # 获取专家权重（只使用CQI指导，与标准Transformer最接近）
        expert_weights = self.router(cqi)  # (batch_size, num_experts)，已通过Softmax归一化
        
        # 训练时添加可学习的高斯噪声，增强探索能力，防止专家死亡
        if self.training and self.top_k < self.num_experts:
            # 添加高斯噪声：标准正态分布 × 可学习的噪声强度
            noise = torch.randn_like(expert_weights) * self.noise_scale
            # 将噪声添加到专家权重上
            noisy_weights = expert_weights + noise
            # 重新归一化以保证权重和为1
            expert_weights = F.softmax(noisy_weights, dim=-1)
        
        # Top-K稀疏激活：只保留权重最高的K个专家
        if self.top_k < self.num_experts:
            top_k_weights, top_k_indices = torch.topk(expert_weights, k=self.top_k, dim=-1)
            # 创建稀疏权重矩阵，只激活Top-K个专家
            sparse_weights = torch.zeros_like(expert_weights)
            sparse_weights.scatter_(1, top_k_indices, top_k_weights)
            # 重新归一化（保证权重和为1）
            sparse_weights = sparse_weights / (sparse_weights.sum(dim=-1, keepdim=True) + 1e-8)
            expert_weights = sparse_weights
        
        # 关键修复：缩放权重使平均权重为1.0，保持与标准FFN相同的信号强度
        # Softmax归一化后，权重之和=1，平均权重=1/num_experts
        # 通过乘以num_experts，使平均权重=1.0，信号不被稀释
        combined_weights = expert_weights * self.num_experts 
        
        # 先计算主路径（原TransNet FFN）：linear1 → activation → dropout → linear2
        base_out = self.base_linear1(x)
        base_out = self.activation(base_out)
        base_out = self.base_dropout(base_out)
        base_out = self.base_linear2(base_out)  # (seq_len, batch_size, d_model)

        # 为每个专家计算输出（轻量FFN）
        expert_outputs = []
        for i in range(self.num_experts):
            expert = self.experts[i]
            expert_out = expert['linear1'](x)  # (seq_len, batch_size, expert_hidden_dim)
            expert_out = self.activation(expert_out)
            expert_out = expert['dropout'](expert_out)  # dropout在activation之后
            expert_out = expert['linear2'](expert_out)  # (seq_len, batch_size, d_model)
            expert_outputs.append(expert_out)
        
        # 加权组合专家输出
        expert_outputs = torch.stack(expert_outputs, dim=-1)  # (seq_len, batch_size, d_model, num_experts)
        # combined_weights: (batch_size, num_experts)
        # 需要broadcast到 (seq_len, batch_size, d_model, num_experts)
        combined_weights = combined_weights.unsqueeze(0).unsqueeze(2)  # (1, batch_size, 1, num_experts)
        expert_mix = torch.sum(expert_outputs * combined_weights, dim=-1)  # (seq_len, batch_size, d_model)

        # 按照需求组合：0.9 * x_base + m（m为专家加权和）
        output = self.base_alpha * base_out + 0.1*expert_mix
        
        # 输出投影（标准FFN没有这一步，移除）
        # output = self.output_proj(output)
        
        # 注意：MoEFFN不包含残差连接！
        # 残差连接在外部 TransformerEncoderLayer_MoE 中处理
        # 这里只返回纯粹的MoE输出，与标准FFN保持一致
        
        # 返回输出和expert_weights（用于计算负载平衡损失）
        # 注意：这里不修改forward签名，通过attribute传递
        self._current_expert_weights = expert_weights
        
        return output


class TransformerEncoderLayer_MoE(nn.Module):
    """
    集成MoE FFN的编码器层
    """
    
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, 
                 activation=F.relu, layer_norm_eps=1e-5, batch_first=False,
                 num_experts=8, cqi_vocab_size=16, mode='wide', top_k: int = None):
        super(TransformerEncoderLayer_MoE, self).__init__()
        
        # 自注意力层（保持不变）
        from models.TransNet import MultiheadAttention
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        
        # MoE FFN层（替换原有FFN）
        self.moe_ffn = MoEFFN(
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            num_experts=num_experts,
            cqi_vocab_size=cqi_vocab_size,
            mode=mode,
            dropout=dropout,
            activation=activation,
            top_k=top_k
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = activation
        
    def forward(self, src: torch.Tensor, cqi: torch.Tensor, 
                src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 自注意力子层
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # MoE FFN子层（与标准TransNet保持一致）
        # 标准TransNet第256行：src2 = linear2(dropout(activation(linear1(src))))
        # MoE内部每个专家已经应用了dropout（第194行）
        src2 = self.moe_ffn(src, cqi)
        # 但标准TransNet第257行在残差连接前还有dropout
        # 为了完全一致，我们需要在这再加一次dropout
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src


class TransformerDecoderLayer_MoE(nn.Module):
    """
    集成MoE FFN的解码器层
    """
    
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation=F.relu, layer_norm_eps=1e-5, batch_first=False,
                 num_experts=8, cqi_vocab_size=16, mode='wide', top_k: int = None):
        super(TransformerDecoderLayer_MoE, self).__init__()
        
        # 注意力层（保持不变）
        from models.TransNet import MultiheadAttention
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        
        # MoE FFN层（替换原有FFN）
        self.moe_ffn = MoEFFN(
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            num_experts=num_experts,
            cqi_vocab_size=cqi_vocab_size,
            mode=mode,
            dropout=dropout,
            activation=activation,
            top_k=top_k
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        self.activation = activation
        
    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, cqi: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 自注意力子层
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # 交叉注意力子层
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # MoE FFN子层
        tgt2 = self.moe_ffn(tgt, cqi)
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt


class TransformerEncoder_MoE(nn.Module):
    """
    集成MoE的编码器
    """
    
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder_MoE, self).__init__()
        self.layer = encoder_layer
        self.num_layers = num_layers
        self.norm = norm
        
    def forward(self, src: torch.Tensor, cqi: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        output = src
        for _ in range(self.num_layers):
            output = self.layer(output, cqi, src_mask=mask, 
                              src_key_padding_mask=src_key_padding_mask)
        
        if self.norm is not None:
            output = self.norm(output)
        
        return output


class TransformerDecoder_MoE(nn.Module):
    """
    集成MoE的解码器
    """
    
    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder_MoE, self).__init__()
        self.layer = decoder_layer
        self.num_layers = num_layers
        self.norm = norm
        
    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, cqi: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        output = tgt
        for _ in range(self.num_layers):
            output = self.layer(output, memory, cqi, tgt_mask=tgt_mask,
                              memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)
        
        if self.norm is not None:
            output = self.norm(output)
        
        return output


class Transformer_MoE(nn.Module):
    """
    集成MoE的基础Transformer类
    """
    
    def __init__(self, d_model: int = 64, nhead: int = 8, num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation=F.relu, layer_norm_eps: float = 1e-5, batch_first: bool = False,
                 reduction=64, num_experts=8, cqi_vocab_size=16, mode='wide', top_k: int = None):
        super(Transformer_MoE, self).__init__()
        
        self.d_model = d_model
        self.nhead = nhead
        self.batch_first = batch_first
        self.mode = mode
        self.output_scale = nn.Parameter(torch.tensor(1.0))
        assert not (3328 % self.d_model), 'd_model needs to be divisible by the size of the entire csi matrix (3328)'
        self.feature_shape = (3328//self.d_model, self.d_model)
        
        # 编码器层
        encoder_layer = TransformerEncoderLayer_MoE(
            d_model, nhead, dim_feedforward, dropout, activation, 
            layer_norm_eps, batch_first, num_experts, cqi_vocab_size, mode, top_k
        )
        encoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.encoder = TransformerEncoder_MoE(encoder_layer, num_encoder_layers, encoder_norm)
        
        # 解码器层
        decoder_layer = TransformerDecoderLayer_MoE(
            d_model, nhead, dim_feedforward, dropout, activation,
            layer_norm_eps, batch_first, num_experts, cqi_vocab_size, mode, top_k
        )
        decoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.decoder = TransformerDecoder_MoE(decoder_layer, num_decoder_layers, decoder_norm)
        
        # 线性层
        self.fc_encoder = nn.Linear(3328, 3328//reduction)
        self.fc_decoder = nn.Linear(3328//reduction, 3328)
        
        # 为了兼容main.py中的代码，添加transformer属性
        # 这里我们创建一个虚拟的transformer对象，包含编码器和解码器
        self.transformer = nn.Module()
        self.transformer.encoder = self.encoder
        self.transformer.decoder = self.decoder
        
        self._reset_parameters()
        
    def forward(self, src: torch.Tensor, cqi: torch.Tensor, tgt: Optional[torch.Tensor] = None,
                src_mask: Optional[torch.Tensor] = None, tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None, src_key_padding_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        # 实时输出模型配置信息（仅在训练时输出一次）
        if self.training and not hasattr(self, '_config_printed'):
            print(f"[TransNet MoE] 模型配置监测:")
            print(f"  - 编码器层数: {self.encoder.num_layers}")
            print(f"  - 解码器层数: {self.decoder.num_layers}")
            print(f"  - CQI模式: {self.mode.upper()}")
            print(f"  - d_model: {self.d_model}")
            print(f"  - 注意力头数: {self.nhead}")
            print(f"  - 信道仿真: 关闭")
            # 获取专家数量（从编码器层的MoE FFN获取）
            expert_count = 'N/A'
            if hasattr(self.encoder, 'layer') and hasattr(self.encoder.layer, 'moe_ffn'):
                expert_count = self.encoder.layer.moe_ffn.num_experts
            print(f"  - 专家数量: {expert_count}")
        #print(f"  - 压缩比: {3328 // self.fc_encoder.out_features}")
        self._config_printed = True
        
        # 编码器处理
        # src: (batch_size, 2, 32, 52)
        B = src.shape[0]  # batch_size = 200
        # 将src从 (B, 3328) reshape为 (B, seq_len, d_model)
        src_reshaped = src.view(B, self.feature_shape[0], self.feature_shape[1])  # (200, 32, 64)
        
        # transpose 以适应 (seq_len, batch_size, d_model) 格式 = (32, 200, 64)
        memory = self.encoder(
            src_reshaped.transpose(0, 1), 
            cqi, mask=src_mask, src_key_padding_mask=src_key_padding_mask
        )
        
        # 压缩编码
        # memory: (seq_len, batch_size, d_model) -> transpose -> (batch_size, seq_len, d_model) -> view -> (batch_size, -1)
        memory_encoder = self.fc_encoder(memory.transpose(0, 1).contiguous().view(B, -1))
        # 压缩后再 reshape 和 transpose 回 (seq_len, batch_size, d_model)
        memory_decoder = self.fc_decoder(memory_encoder).view(B, self.feature_shape[0], self.feature_shape[1]).transpose(0, 1)
        
        # 解码器处理
        output = self.decoder(
            memory_decoder, memory_decoder, cqi, tgt_mask=tgt_mask, memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        
        # 输出reshape
        # output: (seq_len, batch_size, d_model) -> transpose -> (batch_size, seq_len, d_model) -> view -> (batch_size, 2, 32, 52)
        output = output.transpose(0, 1).contiguous().view(B, 2, 32, 52)
        output = output * self.output_scale
        return output
    
    def _reset_parameters(self):
        """初始化模型参数 - 跳过MoE参数（已在MoEFFN中初始化）"""
        for name, p in self.named_parameters():
            # 跳过MoE相关参数（已经在MoEFFN._init_experts中初始化）
            if 'moe_ffn' not in name:
                if p.dim() > 1:
                    xavier_uniform_(p)
    
    def get_model_info(self):
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        linear_params = sum(p.numel() for p in self.fc_encoder.parameters()) + \
                       sum(p.numel() for p in self.fc_decoder.parameters())
        
        # 计算MoE相关参数（CQI融合参数）
        moe_params = 0
        if hasattr(self.encoder, 'layer') and hasattr(self.encoder.layer, 'moe_ffn'):
            # 编码器MoE参数
            moe_params += sum(p.numel() for p in self.encoder.layer.moe_ffn.parameters())
            # 解码器MoE参数
            if hasattr(self.decoder, 'layer') and hasattr(self.decoder.layer, 'moe_ffn'):
                moe_params += sum(p.numel() for p in self.decoder.layer.moe_ffn.parameters())
        
        info = {
            'cqi_type': self.mode,
            'use_cqi_fusion': self.mode != 'nan',  # MoE使用CQI进行专家选择
            'total_params': total_params,
            'transformer_params': encoder_params + decoder_params,
            'transformer_ratio': (encoder_params + decoder_params) / max(1, total_params),
            'fusion_params': moe_params,
            'fusion_ratio': moe_params / max(1, total_params),
            'd_model': self.d_model,
            'nhead': self.nhead,
            'num_encoder_layers': self.encoder.num_layers,
            'num_decoder_layers': self.decoder.num_layers,
            'reduction': 3328 // self.fc_encoder.out_features,
            'mode': self.mode,
            'device': 'cpu',  # 基础版本无设备信息
            'encoder_params': encoder_params,
            'decoder_params': decoder_params,
            'linear_params': linear_params,
            'has_channel_simulation': False,
            'has_moe': True
        }
        
        return info


class Transformer_C_MoE(nn.Module):
    """
    集成MoE的带信道模拟的Transformer类
    """
    
    def __init__(self, d_model: int = 64, nhead: int = 8, num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation=F.relu, layer_norm_eps: float = 1e-5, batch_first: bool = False,
                 reduction=64, num_experts=8, cqi_vocab_size=16, mode='wide',
                 snr: float = -10.0, device: str = 'cuda', top_k: int = None):
        super(Transformer_C_MoE, self).__init__()
        
        self.d_model = d_model
        self.nhead = nhead
        self.batch_first = batch_first
        self.mode = mode
        self.snr = snr
        self.device = device
        
        assert not (3328 % self.d_model), 'd_model needs to be divisible by the size of the entire csi matrix (3328)'
        self.feature_shape = (3328//self.d_model, self.d_model)
        
        # 编码器层
        encoder_layer = TransformerEncoderLayer_MoE(
            d_model, nhead, dim_feedforward, dropout, activation,
            layer_norm_eps, batch_first, num_experts, cqi_vocab_size, mode, top_k
        )
        encoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.encoder = TransformerEncoder_MoE(encoder_layer, num_encoder_layers, encoder_norm)
        
        # 解码器层
        decoder_layer = TransformerDecoderLayer_MoE(
            d_model, nhead, dim_feedforward, dropout, activation,
            layer_norm_eps, batch_first, num_experts, cqi_vocab_size, mode, top_k
        )
        decoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.decoder = TransformerDecoder_MoE(decoder_layer, num_decoder_layers, decoder_norm)
        
        # 线性层
        self.fc_encoder = nn.Linear(3328, 3328//reduction)
        self.fc_decoder = nn.Linear(3328//reduction, 3328)
        
        # 为了兼容main.py中的代码，添加transformer属性
        # 这里我们创建一个虚拟的transformer对象，包含编码器和解码器
        self.transformer = nn.Module()
        self.transformer.encoder = self.encoder
        self.transformer.decoder = self.decoder
        
        self._reset_parameters()
    
    def normalize(self, x, power=1):
        """归一化函数"""
        power_emp = torch.mean(x ** 2)
        x = (power / power_emp) ** 0.5 * x
        return power_emp, x
    
    def awgn(self, snr, x):
        """加性白高斯噪声"""
        n = 1 / (10 ** (snr / 10))
        sqrt_n = n ** 0.5
        noise = torch.randn_like(torch.abs(x)) * sqrt_n
        noise = noise.to(self.device)
        x_hat = x + noise
        return x_hat
    
    def channel_simulation(self, x, snr=None):
        """信道模拟"""
        if snr is None:
            snr = self.snr
        
        power_emp, norm_x = self.normalize(x)
        print("snr为：", snr)
        chan_x = self.awgn(snr, norm_x)
        
        return chan_x, power_emp
    
    def forward(self, src: torch.Tensor, cqi: torch.Tensor, tgt: Optional[torch.Tensor] = None,
                src_mask: Optional[torch.Tensor] = None, tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None, src_key_padding_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None, snr: Optional[float] = None) -> torch.Tensor:
        
        # 实时输出模型配置信息（仅在训练时输出一次）
        if self.training and not hasattr(self, '_config_printed'):
            print(f"[TransNet MoE] 模型配置监测:")
            print(f"  - 编码器层数: {self.encoder.num_layers}")
            print(f"  - 解码器层数: {self.decoder.num_layers}")
            print(f"  - CQI模式: {self.mode.upper()}")
            print(f"  - d_model: {self.d_model}")
            print(f"  - 注意力头数: {self.nhead}")
            print(f"  - 信道仿真: 开启 (SNR={self.snr})")
            # 获取专家数量（从编码器层的MoE FFN获取）
            expert_count = 'N/A'
            if hasattr(self.encoder, 'layer') and hasattr(self.encoder.layer, 'moe_ffn'):
                expert_count = self.encoder.layer.moe_ffn.num_experts
        print(f"  - 专家数量: {expert_count}")
        print(f"  - 压缩比: {2048 // self.fc_encoder.out_features}")
        self._config_printed = True
        
        # 编码器处理
        # src: (batch_size, 2, 32, 52)
        B = src.shape[0]  # batch_size = 200
        # 将src从 (B, 3328) reshape为 (B, seq_len, d_model)
        src_reshaped = src.view(B, self.feature_shape[0], self.feature_shape[1])  # (200, 32, 64)
        
        # transpose 以适应 (seq_len, batch_size, d_model) 格式 = (32, 200, 64)
        memory = self.encoder(
            src_reshaped.transpose(0, 1), 
            cqi, mask=src_mask, src_key_padding_mask=src_key_padding_mask
        )
        
        # 压缩编码
        # memory: (seq_len, batch_size, d_model) -> transpose -> (batch_size, seq_len, d_model) -> view -> (batch_size, -1)
        memory_flat = memory.transpose(0, 1).contiguous().view(B, -1)
        memory_encoder = self.fc_encoder(memory_flat)
        
        # 信道模拟
        chan_memory, power_emp = self.channel_simulation(memory_encoder, snr)
        print("经过信道，")
        
        # 解压缩
        # 压缩后再 reshape 和 transpose 回 (seq_len, batch_size, d_model)
        memory_decoder = self.fc_decoder(chan_memory).view(B, self.feature_shape[0], self.feature_shape[1]).transpose(0, 1)
        
        # 解码器处理
        output = self.decoder(
            memory_decoder, memory_decoder, cqi, tgt_mask=tgt_mask, memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        
        # 输出reshape
        # output: (seq_len, batch_size, d_model) -> transpose -> (batch_size, seq_len, d_model) -> view -> (batch_size, 2, 32, 52)
        output = output.transpose(0, 1).contiguous().view(B, 2, 32, 52)
        return output
    
    def _reset_parameters(self):
        """初始化模型参数 - 跳过MoE参数（已在MoEFFN中初始化）"""
        for name, p in self.named_parameters():
            # 跳过MoE相关参数（已经在MoEFFN._init_experts中初始化）
            if 'moe_ffn' not in name:
                if p.dim() > 1:
                    xavier_uniform_(p)
    
    def set_snr(self, snr: float):
        """设置信噪比"""
        self.snr = snr
    
    def get_snr(self) -> float:
        """获取当前信噪比"""
        return self.snr
    
    def get_model_info(self):
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        linear_params = sum(p.numel() for p in self.fc_encoder.parameters()) + \
                       sum(p.numel() for p in self.fc_decoder.parameters())
        
        # 计算MoE相关参数（CQI融合参数）
        moe_params = 0
        if hasattr(self.encoder, 'layer') and hasattr(self.encoder.layer, 'moe_ffn'):
            # 编码器MoE参数
            moe_params += sum(p.numel() for p in self.encoder.layer.moe_ffn.parameters())
            # 解码器MoE参数
            if hasattr(self.decoder, 'layer') and hasattr(self.decoder.layer, 'moe_ffn'):
                moe_params += sum(p.numel() for p in self.decoder.layer.moe_ffn.parameters())
        
        info = {
            'cqi_type': self.mode,
            'use_cqi_fusion': self.mode != 'nan',  # MoE使用CQI进行专家选择
            'total_params': total_params,
            'transformer_params': encoder_params + decoder_params,
            'transformer_ratio': (encoder_params + decoder_params) / max(1, total_params),
            'fusion_params': moe_params,
            'fusion_ratio': moe_params / max(1, total_params),
            'd_model': self.d_model,
            'nhead': self.nhead,
            'num_encoder_layers': self.encoder.num_layers,
            'num_decoder_layers': self.decoder.num_layers,
            'reduction': 3328 // self.fc_encoder.out_features,
            'mode': self.mode,
            'device': self.device,
            'encoder_params': encoder_params,
            'decoder_params': decoder_params,
            'linear_params': linear_params,
            'has_channel_simulation': True,
            'snr': self.snr,
            'has_moe': True
        }
        
        return info


def create_transnet_moe(reduction=64, d_model=64, num_experts=8, mode='wide',
                       num_encoder_layers=2, num_decoder_layers=2, nhead=2, 
                       dropout=0.0, snr=-10.0, device='cuda', enable_channel=False,
                       top_k: int = None):
    """
    创建TransNet MoE模型
    
    Args:
        reduction: 压缩比
        d_model: 模型维度
        num_experts: 专家网络数量
        mode: 'wide' 或 'sub'
        num_encoder_layers: 编码器层数
        num_decoder_layers: 解码器层数
        nhead: 注意力头数
        dropout: Dropout概率
        snr: 信噪比
        device: 设备
        enable_channel: 是否开启信道仿真（默认False）
        top_k: Top-K稀疏激活，只激活前K个专家（None表示激活所有专家）
    """
    if enable_channel:
        # 使用带信道仿真的版本
        model = Transformer_C_MoE(
            d_model=d_model, 
            num_encoder_layers=num_encoder_layers, 
            num_decoder_layers=num_decoder_layers, 
            nhead=nhead, 
            reduction=reduction, 
            dropout=dropout,
            num_experts=num_experts,
            mode=mode,
            snr=snr,
            device=device,
            top_k=top_k
        )
        print(f"已使用信道仿真，MoE模式: {mode}, 专家数量: {num_experts}")
    else:
        # 使用基础版本（无信道仿真）
        model = Transformer_MoE(
            d_model=d_model, 
            num_encoder_layers=num_encoder_layers, 
            num_decoder_layers=num_decoder_layers, 
            nhead=nhead, 
            reduction=reduction, 
            dropout=dropout,
            num_experts=num_experts,
            mode=mode,
            top_k=top_k
        )
        print(f"未使用信道仿真，MoE模式: {mode}, 专家数量: {num_experts}")
    
    print(f"编码器层数: {num_encoder_layers}, 解码器层数: {num_decoder_layers}, 注意力头数: {nhead}")
    return model
