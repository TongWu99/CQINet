import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_
from typing import Optional, Any
import math

class CQIQueryMLP_D(nn.Module):
    """
    多层线性映射：直接将13维CQI向量映射到A/B矩阵
    无Embedding，单/子带共享同一流程
    A/B 各自拥有独立的MLP
    """

    def __init__(
        self,
        d_model: int,
        mode: str = "wide",
        num_subbands: int = 13,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.mode = mode
        self.num_subbands = num_subbands
        self.hidden_dim = hidden_dim or max(d_model*2, 32)

        self.input_proj_A = nn.Sequential(
            nn.Linear(num_subbands, d_model),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        )
        self.input_proj_B = nn.Sequential(
            nn.Linear(num_subbands, d_model),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        )

        self.mlp_A = nn.Sequential(
            nn.Linear(d_model, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(self.hidden_dim, d_model * d_model)
        )
        self.mlp_B = nn.Sequential(
            nn.Linear(d_model, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(self.hidden_dim, d_model)
        )
        self.mix_logit = nn.Parameter(torch.tensor([math.log(0.9), math.log(0.1)], dtype=torch.float32))
        self._reset_parameters()

    def _reset_parameters(self):
        for seq in (self.input_proj_A, self.input_proj_B, self.mlp_A, self.mlp_B):
            for m in seq:
                if isinstance(m, nn.Linear):
                    xavier_uniform_(m.weight, gain=1e-2)
                    constant_(m.bias, 0.0)

    def _prepare_cqi(self, cqi: torch.Tensor) -> torch.Tensor:
        if cqi is None:
            raise ValueError("CQIQueryMLP_D requires CQI input.")
        if cqi.dim() == 1:
            cqi = cqi.unsqueeze(1)
        if cqi.size(1) == 1:
            cqi = cqi.expand(-1, self.num_subbands)
        if cqi.size(1) != self.num_subbands:
            raise ValueError(
                f"CQI维度应为{self.num_subbands}，但得到{cqi.size(1)}，"
                "请确认sub模式下的子带数或wide模式的扩展策略。"
            )
        return cqi.float()

    def forward(self, q: torch.Tensor, cqi: torch.Tensor) -> torch.Tensor:
        #print(self.mode)
        original_layout = q.shape
        if q.dim() == 3 and q.shape[0] != q.shape[1]:
            q = q.transpose(0, 1)
        bsz, tgt_len, d = q.shape
        #print("CQI:",cqi)
        cqi_vec = self._prepare_cqi(cqi)
        feat_A = self.mlp_A(self.input_proj_A(cqi_vec))
        feat_B = self.mlp_B(self.input_proj_B(cqi_vec))
        A_raw = feat_A.view(bsz, d, d)
        A = F.softplus(A_raw) + torch.eye(d, device=q.device, dtype=q.dtype).unsqueeze(0)
        B = feat_B
        # 新增：强度标量
        a_strength = (A - torch.eye(d, device=A.device, dtype=A.dtype).unsqueeze(0)).pow(2).sum(dim=[1, 2]).sqrt()
        b_strength = B.pow(2).sum(dim=1).sqrt()
        q_aff = torch.matmul(q, A) + B.unsqueeze(1)

        w = F.softmax(self.mix_logit, dim=0)#后加的
        q_mix = w[0] * q + w[1] * q_aff #后加的
        #q_mix = w[0] * q + w[1] * q_aff #后加的
        if original_layout[0] != original_layout[1]:
            #q_aff = q_aff.transpose(0, 1)
            q_mix = q_mix.transpose(0, 1)#后加的
        # 缓存用于相关性正则的统计量（保持梯度）
        self._corr_data = {
            'a_strength': a_strength,
            'b_strength': b_strength,
            'cqi_scalar': cqi_vec.mean(dim=1)
        }
        return q_mix


def _in_projection_packed(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, w: torch.Tensor, b: Optional[torch.Tensor] = None):
    E = q.size(-1)
    if k is v:
        if q is k:
            return F.linear(q, w, b).chunk(3, dim=-1)
        else:
            w_q, w_kv = w.split([E, E * 2])
            if b is None:
                b_q = b_kv = None
            else:
                b_q, b_kv = b.split([E, E * 2])
            return (F.linear(q, w_q, b_q),) + F.linear(k, w_kv, b_kv).chunk(2, dim=-1)
    else:
        w_q, w_k, w_v = w.chunk(3)
        if b is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = b.chunk(3)
        return F.linear(q, w_q, b_q), F.linear(k, w_k, b_k), F.linear(v, w_v, b_v)


def multi_head_attention_forward(query, key, value, num_heads, in_proj_weight, in_proj_bias, dropout_p,
                                 out_proj_weight, out_proj_bias, training=True, key_padding_mask=None,
                                 need_weights=True, attn_mask=None, use_separate_proj_weight=None):
    from models.TransNet import multi_head_attention_forward as core_forward
    return core_forward(query, key, value, num_heads, in_proj_weight, in_proj_bias, dropout_p,
                        out_proj_weight, out_proj_bias, training, key_padding_mask, need_weights, attn_mask,
                        use_separate_proj_weight)


class MultiheadAttention_CQIPlusD(nn.Module):
    def __init__(self, embed_dim, num_heads, mode='wide', dropout=0., bias=True, batch_first=False):
        super().__init__()
        from models.TransNet import MultiheadAttention
        self.core_attn = MultiheadAttention(embed_dim, num_heads, dropout=dropout, bias=bias, batch_first=batch_first)
        self.q_mapper = CQIQueryMLP_D(d_model=embed_dim, mode=mode)
        self.batch_first = batch_first

    def forward(self, query, key, value, cqi=None, key_padding_mask=None, need_weights=True, attn_mask=None):
        #print()
        if cqi is None:
            #print("nan")
            return self.core_attn(query, key, value, key_padding_mask=key_padding_mask, need_weights=need_weights, attn_mask=attn_mask)
        from models.TransNet import _in_projection_packed, multi_head_attention_forward
        if self.core_attn.batch_first:
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]
        q, k, v = _in_projection_packed(query, key, value, self.core_attn.in_proj_weight, self.core_attn.in_proj_bias)
        q = self.q_mapper(q, cqi)
        out, attn_w = multi_head_attention_forward(
            q, k, v, self.core_attn.num_heads,
            self.core_attn.in_proj_weight, self.core_attn.in_proj_bias,
            self.core_attn.dropout, self.core_attn.out_proj.weight, self.core_attn.out_proj.bias,
            training=self.training, key_padding_mask=key_padding_mask, need_weights=need_weights, attn_mask=attn_mask,
            use_separate_proj_weight=None,
        )
        if self.batch_first:
            out = out.transpose(1, 0)
        return out, attn_w


class TransformerEncoderLayer_QModPlusD(nn.Module):
    def __init__(self, d_model, nhead, mode='wide', dim_feedforward=2048, dropout=0.1, activation=F.relu,
                 layer_norm_eps=1e-5, batch_first=False) -> None:
        super().__init__()
        self.self_attn = MultiheadAttention_CQIPlusD(d_model, nhead, mode=mode, dropout=dropout, batch_first=batch_first)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, src: torch.Tensor, cqi: torch.Tensor, src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        src2 = self.self_attn(src, src, src, cqi=cqi, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerEncoder_QModPlusD(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.norm = norm

    def forward(self, src: torch.Tensor, cqi: torch.Tensor, mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        output = src
        #print("CQI:",cqi)
        for layer in self.layers:
            output = layer(output, cqi, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output


class TransformerDecoderLayer_QModPlusD(nn.Module):
    def __init__(self, d_model, nhead, mode='wide', dim_feedforward=2048, dropout=0.1, activation=F.relu,
                 layer_norm_eps=1e-5, batch_first=False) -> None:
        super().__init__()
        self.self_attn = MultiheadAttention_CQIPlusD(d_model, nhead, mode=mode, dropout=dropout, batch_first=batch_first)
        self.multihead_attn = MultiheadAttention_CQIPlusD(d_model, nhead, mode=mode, dropout=dropout, batch_first=batch_first)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, cqi: torch.Tensor, tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None, tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        tgt2 = self.self_attn(tgt, tgt, tgt, cqi=cqi, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory, cqi=cqi, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class TransformerDecoder_QModPlusD(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.norm = norm

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, cqi: torch.Tensor, tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None, tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        output = tgt
        for layer in self.layers:
            output = layer(output, memory, cqi, tgt_mask=tgt_mask, memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output


class Transformer_QModPlusD(nn.Module):
    def __init__(self, mode='wide', d_model: int = 64, nhead: int = 8, num_encoder_layers: int = 2,
                 num_decoder_layers: int = 2, dim_feedforward: int = 2048, dropout: float = 0.0,
                 activation = F.relu, layer_norm_eps: float = 1e-5, batch_first: bool = False, reduction=64) -> None:
        super().__init__()
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

        # 支持近似压缩率：计算最接近目标压缩率的整数维度
        target_compressed_dim = 3328 // reduction
        # 如果不能整除，使用最接近的整数值
        self.compressed_dim = max(1, target_compressed_dim)  # 确保至少为1
        self.actual_reduction = 3328 / self.compressed_dim

        print(f"目标压缩率: 1/{reduction:.1f}, 实际压缩率: 1/{self.actual_reduction:.1f} (维度: {self.compressed_dim})")

        self.fc_encoder = nn.Linear(3328, self.compressed_dim)
        self.fc_decoder = nn.Linear(self.compressed_dim, 3328)
        self._reset_parameters()
        self.output_scale = nn.Parameter(torch.tensor(1.0))

        self.transformer = nn.Module()
        self.transformer.encoder = self.encoder
        self.transformer.decoder = self.decoder

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, src: torch.Tensor, cqi: Optional[torch.Tensor] = None, tgt: Optional[torch.Tensor] = None, src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None, memory_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None, tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        if cqi is None:
            if 'wideband_cqi' in kwargs and kwargs['wideband_cqi'] is not None:
                cqi = kwargs['wideband_cqi']
            elif 'subband_cqi' in kwargs and kwargs['subband_cqi'] is not None:
                cqi = kwargs['subband_cqi']
        #print("src.shape:",src.shape)     
        B = src.shape[0]
        #print("成功被调用转置！")
        x = src.permute(0, 1, 3, 2).contiguous() #后加的111
        #src_reshaped = x.view(B, 2 * 52, 32) #后加的000
        src_reshaped = x.view(B, self.feature_shape[0], self.feature_shape[1])#后加的111
        #src_reshaped = src.view(B, self.feature_shape[0], self.feature_shape[1])# 源代码

        # print("src_reshape.shape:",src_reshaped.shape)
        # print("src_reshape.transpose(0,1).shape:",src_reshaped.transpose(0, 1).shape)


        memory = self.encoder(src_reshaped.transpose(0, 1), None, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        try:
            first_q_mapper = self.encoder.layers[0].self_attn.q_mapper
            if hasattr(first_q_mapper, '_corr_data'):
                self.corr_cache = first_q_mapper._corr_data
                #print("成功启用")
        except Exception:
            self.corr_cache = None
        memory_encoder = self.fc_encoder(memory.transpose(0, 1).contiguous().view(B, -1))
        memory_decoder = self.fc_decoder(memory_encoder).view(B, self.feature_shape[0], self.feature_shape[1]).transpose(0, 1)
        output = self.decoder(memory_decoder, memory_decoder, None, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
        #output = output.transpose(0, 1).contiguous().view(B, 2, 32, 52) # 源代码
        # (104, B, 32) -> (B, 104, 32)
        output = output.transpose(0, 1).contiguous() #后加的111
        # 还原到 (B, 2, 32, 52): (B, 104, 32) -> (B, 2, 52, 32) -> (B, 2, 32, 52)
        #output = output.view(B, 2, 52, 32).permute(0, 1, 3, 2).contiguous() #后加的000
        output = output.view(B, 2, 52, 32).permute(0, 1, 3, 2).contiguous() #后加的111
        output = self.output_scale * output
        
        return output

    def get_model_info(self):
        total_params = sum(p.numel() for p in self.parameters())
        linear_params = sum(p.numel() for p in self.fc_encoder.parameters()) + sum(p.numel() for p in self.fc_decoder.parameters())
        info = {
            'use_cqi_fusion': True,
            'fusion_params': total_params - linear_params,
            'fusion_ratio': (total_params - linear_params) / max(1, total_params),
        }
        return info


def create_transnet_qmodplusd(reduction=64, d_model=64, mode='wide', num_encoder_layers=2, num_decoder_layers=2, nhead=2, dropout=0.0):
    model = Transformer_QModPlusD(mode=mode, d_model=d_model, num_encoder_layers=num_encoder_layers,
                                  num_decoder_layers=num_decoder_layers, nhead=nhead, dropout=dropout, reduction=reduction)
    return model


