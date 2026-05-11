"""
模型测试脚本
===========
加载已保存的模型，对指定路径的数据集进行测试。

支持模型类型：
- TransCQA/TransNet 系列：multiscale, transnet_moe, transnet_qmod, transnet_qmodplus, transnet_qmodplusd, transnet_qc
- 基础模型：crnet, clnet
- CQI微调模型：crnet_cqi, clnet_cqi

重要：测试集的比例和划分方式必须与训练时完全一致，才能公平评估模型性能。
- 随机种子：42（与训练时一致）
- 数据划分：80%训练，10%验证，10%测试
- stratify_by_cqi：与训练时保持一致
- cqi_bins：与训练时保持一致
"""

import torch
import torch.nn as nn
import argparse
import os
import sys
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，避免GUI问题

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from utils import logger, init_device
from utils.solver import Tester
from dataloader.losangeles import LosAngelesDataLoader


def create_model(config):
    """创建模型（从 init_model 复制关键逻辑，避免依赖 config.py 的 parse_args）"""
    #from models.TransCQA_Advanced import transcqa_multiscale
    from models.TransNet_MoE import create_transnet_moe
    #from models.TransNet_QMod import create_transnet_qmod
    #from models.TransNet_QModPlus import create_transnet_qmodplus
    from models.TransNet_QModPlusD import create_transnet_qmodplusd
    #from models.TransNet_QC import create_transnet_qc
    from models.crnet import crnet
    from models.clnet import clnet
    from models.crnet_cqi import crnet_cqi
    from models.clnet_cqi import clnet_cqi

    
    model_type = getattr(config, 'model_type', 'multiscale')
    
    if model_type == 'multiscale':
        model = transcqa_multiscale(
            snr=getattr(config, 'snr', 0),
            cqi_type=config.cqi_type,
            reduction=config.cr,
            d_model=config.d_model,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            nhead=config.nhead,
            dropout=config.dropout
        )
    elif model_type == 'transnet_moe':
        model = create_transnet_moe(
            reduction=config.cr,
            d_model=config.d_model,
            num_experts=getattr(config, 'moe_experts', 8),
            mode=config.cqi_type,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            nhead=config.nhead,
            dropout=config.dropout,
            snr=getattr(config, 'snr', 0),
            device=str(config.device),
            enable_channel=getattr(config, 'moe_enable_channel', False),
            top_k=getattr(config, 'moe_top_k', None)
        )
    elif model_type == 'transnet_qmod':
        model = create_transnet_qmod(
            reduction=config.cr,
            d_model=config.d_model,
            mode=config.cqi_type,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            nhead=config.nhead,
            dropout=config.dropout,
            cqi_vocab_size=getattr(config, 'cqi_vocab_size', 16)
        )
    elif model_type == 'transnet_qmodplus':
        model = create_transnet_qmodplus(
            reduction=config.cr,
            d_model=config.d_model,
            mode=config.cqi_type,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            nhead=config.nhead,
            dropout=config.dropout
        )
    elif model_type == 'transnet_qmodplusd':
        model = create_transnet_qmodplusd(
            reduction=config.cr,
            d_model=config.d_model,
            mode=config.cqi_type,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            nhead=config.nhead,
            dropout=config.dropout
        )
    elif model_type == 'transnet_qc':
        model = create_transnet_qc(
            reduction=config.cr,
            d_model=config.d_model,
            mode=config.cqi_type,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            nhead=config.nhead,
            dropout=config.dropout
        )
    elif model_type == 'crnet':
        model = crnet(reduction=config.cr)
    elif model_type == 'clnet':
        model = clnet(reduction=config.cr)
    elif model_type == 'crnet_cqi':
        model = crnet_cqi(reduction=config.cr, sr_hidden_channels=getattr(config, 'sr_hidden_channels', 16))
    elif model_type == 'clnet_cqi':
        model = clnet_cqi(reduction=config.cr, sr_hidden_channels=getattr(config, 'sr_hidden_channels', 16))
    else:
        logger.warning(f"未知的model_type='{model_type}'，回退为'multiscale'")
        model = transcqa_multiscale(
            snr=getattr(config, 'snr', 0),
            cqi_type=config.cqi_type,
            reduction=config.cr,
            d_model=config.d_model,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            nhead=config.nhead,
            dropout=config.dropout
        )
    
    return model


def load_model(model, checkpoint_path, device, logger):
    """加载模型检查点"""
    if not os.path.exists(checkpoint_path):
        logger.error(f'❌ 模型文件不存在: {checkpoint_path}')
        return None, None, None
    
    try:
        logger.info(f'📦 加载模型检查点: {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 检查checkpoint格式
        if 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
            epoch = checkpoint.get('epoch', '未知')
            best_rho = checkpoint.get('best_rho', '未知')
            best_nmse = checkpoint.get('best_nmse', '未知')
        elif 'state_dict' in checkpoint:
            model_state = checkpoint['state_dict']
            epoch = checkpoint.get('epoch', '未知')
            best_rho = checkpoint.get('best_rho', '未知')
            best_nmse = checkpoint.get('best_nmse', '未知')
        else:
            model_state = checkpoint
            epoch = '未知'
            best_rho = '未知'
            best_nmse = '未知'
        
        # 加载模型参数
        model.load_state_dict(model_state, strict=False)
        
        logger.info(f'✅ 模型加载成功!')
        logger.info(f'   训练轮数: {epoch}')
        logger.info(f'   最佳Rho: {best_rho}')
        logger.info(f'   最佳NMSE: {best_nmse}')
        
        return checkpoint, epoch, best_rho
        
    except Exception as e:
        logger.error(f'❌ 模型加载失败: {e}')
        return None, None, None


class TestConfig:
    """测试配置类（不依赖 config.py 的 parse_args）"""
    def __init__(self):
        # 数据相关
        self.data_dir = './losangeles_data'
        self.cqi_type = 'nan'  # CRNet/CLNet 不使用 CQI，默认为 nan
        self.stratify_by_cqi = False
        self.cqi_bins = None

        # 训练相关（用于模型创建）
        self.batch_size = 200
        self.num_workers = 4
        self.seed = 42

        # 模型相关
        self.d_model = 64
        self.cr = 4
        self.num_encoder_layers = 4
        self.num_decoder_layers = 4
        self.nhead = 8
        self.dropout = 0.0
        self.model_type = 'crnet'  # 默认使用 crnet
        self.snr = 0
        self.sr_hidden_channels = 16  # CQI模型的超分辨率模块隐藏层通道数

        # 硬件相关
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 其他
        self.model_path = None


def print_dataset_info(train_size, val_size, test_size, total_size):
    """打印数据集信息"""
    logger.info('=' * 60)
    logger.info('数据集划分信息（确保与训练时一致）')
    logger.info('=' * 60)
    logger.info(f'   总样本数: {total_size:,}')
    logger.info(f'   训练集: {train_size:,} ({train_size/total_size*100:.1f}%)')
    logger.info(f'   验证集: {val_size:,} ({val_size/total_size*100:.1f}%)')
    logger.info(f'   测试集: {test_size:,} ({test_size/total_size*100:.1f}%)')
    logger.info('=' * 60)
    logger.info(f'   随机种子: 42')
    logger.info(f'   划分策略: 8:1:1 (80% 训练, 10% 验证, 10% 测试)')
    logger.info('=' * 60)


def plot_alpha_beta_heatmaps(alpha_matrix, beta_matrix, cqi_counts, output_dir):
    """绘制α和β的二维热力图

    Args:
        alpha_matrix: (16, 52) α均值矩阵，行为CQI值(0-15)，列为频率维度(0-51)
        beta_matrix: (16, 52) β均值矩阵
        cqi_counts: (16,) 每个CQI值的样本数
        output_dir: 输出目录
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # 计算每行（每个CQI）的平均值
    alpha_row_means = alpha_matrix.mean(axis=1)  # (16,) 每个CQI对应的平均alpha
    beta_row_means = beta_matrix.mean(axis=1)     # (16,) 每个CQI对应的平均beta

    fig, axes = plt.subplots(1, 2, figsize=(22, 8))

    # 设置字体
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
    plt.rcParams['axes.unicode_minus'] = False

    # ========== α热力图（根据数据自动确定颜色范围）=========
    ax1 = axes[0]
    im1 = ax1.imshow(alpha_matrix, aspect='auto', cmap='viridis', origin='lower')
    ax1.set_title(r'$\alpha$ Heatmap (Mean per CQI $\times$ Frequency)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Frequency Dimension (0-51)', fontsize=12)
    ax1.set_ylabel('CQI Value (0-15)', fontsize=12)
    ax1.set_xticks(np.arange(0, 52, 4))
    ax1.set_xticklabels([str(i) for i in range(0, 52, 4)])
    ax1.set_yticks(np.arange(0, 16, 1))
    ax1.set_yticklabels([str(i) for i in range(0, 16)])
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label(r'$\alpha$ Value', fontsize=11)

    # 在热力图右侧添加行均值条
    ax1_right = ax1.twinx()
    ax1_right.set_ylim(ax1.get_ylim())
    ax1_right.set_yticks(np.arange(0, 16, 1))
    ax1_right.set_yticklabels([f'{v:.3f}' for v in alpha_row_means], fontsize=9)
    ax1_right.set_ylabel(r'Mean $\alpha$ per CQI', fontsize=11, rotation=270, labelpad=20)

    # ========== β热力图（根据数据自动确定颜色范围）=========
    ax2 = axes[1]
    im2 = ax2.imshow(beta_matrix, aspect='auto', cmap='RdBu_r', origin='lower')
    ax2.set_title(r'$\beta$ Heatmap (Mean per CQI $\times$ Frequency)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Frequency Dimension (0-51)', fontsize=12)
    ax2.set_ylabel('CQI Value (0-15)', fontsize=12)
    ax2.set_xticks(np.arange(0, 52, 4))
    ax2.set_xticklabels([str(i) for i in range(0, 52, 4)])
    ax2.set_yticks(np.arange(0, 16, 1))
    ax2.set_yticklabels([str(i) for i in range(0, 16)])
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
    cbar2.set_label(r'$\beta$ Value', fontsize=11)

    # 在热力图右侧添加行均值条
    ax2_right = ax2.twinx()
    ax2_right.set_ylim(ax2.get_ylim())
    ax2_right.set_yticks(np.arange(0, 16, 1))
    ax2_right.set_yticklabels([f'{v:.3f}' for v in beta_row_means], fontsize=9)
    ax2_right.set_ylabel(r'Mean $\beta$ per CQI', fontsize=11, rotation=270, labelpad=20)

    plt.tight_layout()

    # 保存图像
    output_path = os.path.join(output_dir, 'alpha_beta_heatmap.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


def extract_cqi_parameters(model, test_loader, device, logger):
    """提取CQI嵌入模块的实际输出α和β
    
    Args:
        model: 已加载的模型
        test_loader: 测试数据加载器
        device: 设备
        logger: 日志记录器
    
    Returns:
        dict: 包含α和β实际输出的统计信息
    """
    # 检查模型是否包含cqi_modulator
    if not hasattr(model, 'cqi_modulator'):
        return None
    
    # 获取cqi_modulator的num_subbands参数
    num_subbands = model.cqi_modulator.num_subbands  # 应该是13
    
    # 收集所有样本的CQI、α和β值
    # 归一化CQI（用于模型输入）
    all_cqi_normalized = []
    # 原始CQI（用于分组，1-15的整数）
    all_cqi_original = []
    # 完整的α和β向量 (每个样本的52维向量)
    all_alpha_full = []  # (B, 52) 每个样本完整的α向量
    all_beta_full = []   # (B, 52) 每个样本完整的β向量
    
    logger.info('=> 开始提取CQI嵌入模块的实际输出...')
    logger.info(f'=> cqi_modulator.num_subbands = {num_subbands}')
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_loader):
            # 获取数据 - 数据格式可能是 (H_ini, H_norm) 或 (H_ini, H_norm, cqi)
            if len(batch_data) == 3:
                h_ini, h_norm, cqi = batch_data
            elif len(batch_data) == 2:
                h_ini, h_norm = batch_data
                cqi = None
            else:
                logger.warning(f'未知的数据格式，长度为 {len(batch_data)}')
                continue
            
            # 处理CQI数据
            if cqi is not None:
                cqi = cqi.to(device)
                batch_size = cqi.shape[0]
                
                # 归一化CQI（用于模型输入）
                cqi_norm_np = cqi.cpu().numpy()
                
                # 如果CQI是标量 (B, 1)，扩展为 (B, num_subbands)
                if cqi.dim() == 2 and cqi.shape[1] == 1:
                    cqi_expanded = cqi.expand(-1, num_subbands)
                    logger.info(f'=> 检测到标量CQI，已扩展为 {num_subbands} 维向量')
                elif cqi.shape[1] == num_subbands:
                    cqi_expanded = cqi
                else:
                    # 其他情况，尝试扩展或截断
                    if cqi.shape[1] < num_subbands:
                        cqi_expanded = torch.cat([cqi] + [cqi[:, -1:]] * (num_subbands - cqi.shape[1]), dim=1)
                    else:
                        cqi_expanded = cqi[:, :num_subbands]
                
                # 将CQI传入cqi_modulator获取α和β
                alpha, beta = model.cqi_modulator(cqi_expanded)
                # alpha: (B, 1, 1, 52) -> squeeze后 (B, 52)
                # beta: (B, 1, 1, 52) -> squeeze后 (B, 52)
                
                alpha_np = alpha.squeeze(1).squeeze(1).cpu().numpy()  # (B, 52)
                beta_np = beta.squeeze(1).squeeze(1).cpu().numpy()    # (B, 52)
                
                # 计算原始CQI值（用于分组）
                # 归一化公式: cqi_norm = (cqi_orig - mean) / std
                # 反归一化: cqi_orig = cqi_norm * std + mean
                # 从测试集获取归一化参数
                test_dataset = test_loader.dataset
                if hasattr(test_dataset, 's_mean') and hasattr(test_dataset, 's_std'):
                    s_mean = test_dataset.s_mean
                    s_std = test_dataset.s_std
                    if np.isscalar(s_mean):
                        cqi_orig_np = cqi_norm_np * s_std + s_mean
                    else:
                        cqi_orig_np = cqi_norm_np * s_std[0] + s_mean[0]
                else:
                    # 如果没有归一化参数，假设归一化CQI就是原始CQI
                    cqi_orig_np = cqi_norm_np
                
                all_cqi_normalized.append(cqi_norm_np)
                all_cqi_original.append(cqi_orig_np)
                all_alpha_full.append(alpha_np)
                all_beta_full.append(beta_np)
    
    # 合并所有批次
    all_cqi_normalized = np.concatenate(all_cqi_normalized, axis=0)     # (N, CQI_dim)
    all_cqi_original = np.concatenate(all_cqi_original, axis=0)          # (N, CQI_dim)
    all_alpha_full = np.concatenate(all_alpha_full, axis=0)             # (N, 52)
    all_beta_full = np.concatenate(all_beta_full, axis=0)              # (N, 52)
    
    N = len(all_cqi_normalized)
    num_dims = all_alpha_full.shape[1]  # 应该是52
    logger.info(f'=> 共提取了 {N} 个样本的CQI嵌入输出')
    logger.info(f'=> α/β 向量维度: {num_dims}')
    
    # 如果CQI是一维的，转换为可分析的格式
    if all_cqi_normalized.ndim == 1:
        cqi_norm_flat = all_cqi_normalized
        cqi_orig_flat = all_cqi_original
    elif all_cqi_normalized.shape[1] == 1:
        cqi_norm_flat = all_cqi_normalized.flatten()
        cqi_orig_flat = all_cqi_original.flatten()
    else:
        # 多维CQI，取第一个维度作为主要分析维度
        cqi_norm_flat = all_cqi_normalized[:, 0]
        cqi_orig_flat = all_cqi_original[:, 0]
    
    # 将CQI四舍五入到整数并限制在0-15范围内
    cqi_rounded = np.round(cqi_orig_flat).astype(int)
    cqi_rounded = np.clip(cqi_rounded, 0, 15)
    
    # ==================== 分析CQI与α、β的关系 ====================
    
    # 统计信息
    stats = {
        'alpha': {
            'mean': all_alpha_full.mean(),
            'std': all_alpha_full.std(),
            'min': all_alpha_full.min(),
            'max': all_alpha_full.max(),
            'median': np.median(all_alpha_full),
        },
        'beta': {
            'mean': all_beta_full.mean(),
            'std': all_beta_full.std(),
            'min': all_beta_full.min(),
            'max': all_beta_full.max(),
            'median': np.median(all_beta_full),
        },
        'cqi': {
            'normalized': {
                'mean': all_cqi_normalized.mean(axis=0) if all_cqi_normalized.ndim > 1 else all_cqi_normalized.mean(),
                'std': all_cqi_normalized.std(axis=0) if all_cqi_normalized.ndim > 1 else all_cqi_normalized.std(),
                'min': all_cqi_normalized.min(axis=0) if all_cqi_normalized.ndim > 1 else all_cqi_normalized.min(),
                'max': all_cqi_normalized.max(axis=0) if all_cqi_normalized.ndim > 1 else all_cqi_normalized.max(),
            },
            'original': {
                'mean': all_cqi_original.mean(axis=0) if all_cqi_original.ndim > 1 else all_cqi_original.mean(),
                'std': all_cqi_original.std(axis=0) if all_cqi_original.ndim > 1 else all_cqi_original.std(),
                'min': all_cqi_original.min(axis=0) if all_cqi_original.ndim > 1 else all_cqi_original.min(),
                'max': all_cqi_original.max(axis=0) if all_cqi_original.ndim > 1 else all_cqi_original.max(),
            }
        }
    }
    
    # 打印统计信息
    logger.info('=' * 70)
    logger.info('CQI 嵌入模块实际输出 α 和 β 统计信息')
    logger.info('=' * 70)
    logger.info(f'样本总数: {N}')
    logger.info(f'调制公式: output = α * SR(x) + (1 - α) * x + β')
    logger.info(f'α/β 向量维度: {num_dims}')
    logger.info('-' * 70)
    
    logger.info('[α 实际输出统计]')
    logger.info(f'   均值: {stats["alpha"]["mean"]:.6f}')
    logger.info(f'   标准差: {stats["alpha"]["std"]:.6f}')
    logger.info(f'   范围: [{stats["alpha"]["min"]:.6f}, {stats["alpha"]["max"]:.6f}]')
    logger.info(f'   中位数: {stats["alpha"]["median"]:.6f}')
    
    logger.info('[β 实际输出统计]')
    logger.info(f'   均值: {stats["beta"]["mean"]:.6f}')
    logger.info(f'   标准差: {stats["beta"]["std"]:.6f}')
    logger.info(f'   范围: [{stats["beta"]["min"]:.6f}, {stats["beta"]["max"]:.6f}]')
    logger.info(f'   中位数: {stats["beta"]["median"]:.6f}')
    
    logger.info('-' * 70)
    logger.info('[CQI 输入统计]')
    # 处理mean/std可能是数组的情况
    orig_mean = stats["cqi"]["original"]["mean"]
    orig_min = stats["cqi"]["original"]["min"]
    orig_max = stats["cqi"]["original"]["max"]
    norm_mean = stats["cqi"]["normalized"]["mean"]
    norm_min = stats["cqi"]["normalized"]["min"]
    norm_max = stats["cqi"]["normalized"]["max"]
    
    # 如果是数组，取第一个元素或整体均值
    if np.ndim(orig_mean) > 0:
        orig_mean = orig_mean.flat[0] if orig_mean.size > 0 else 0
    if np.ndim(orig_min) > 0:
        orig_min = orig_min.flat[0] if orig_min.size > 0 else 0
    if np.ndim(orig_max) > 0:
        orig_max = orig_max.flat[0] if orig_max.size > 0 else 0
    if np.ndim(norm_mean) > 0:
        norm_mean = norm_mean.flat[0] if norm_mean.size > 0 else 0
    if np.ndim(norm_min) > 0:
        norm_min = norm_min.flat[0] if norm_min.size > 0 else 0
    if np.ndim(norm_max) > 0:
        norm_max = norm_max.flat[0] if norm_max.size > 0 else 0
    
    logger.info(f'   原始值范围: [{int(orig_min)}, {int(orig_max)}]')
    logger.info(f'   原始值均值: {orig_mean:.2f}')
    logger.info(f'   归一化值范围: [{norm_min:.2f}, {norm_max:.2f}]')
    logger.info(f'   归一化值均值: {norm_mean:.2f}')
    
    logger.info('=' * 70)
    
    # ==================== 分析α与CQI的关系 ====================
    logger.info('=' * 70)
    logger.info('α 与 CQI 的关系分析 (按原始CQI值 1-15 分组)')
    logger.info('=' * 70)
    
    # 按原始CQI值（0-15）分组统计
    unique_cqi = np.unique(cqi_rounded)
    logger.info(f'[原始CQI值范围: {unique_cqi.min()}-{unique_cqi.max()}, 共 {len(unique_cqi)} 种值]')
    logger.info(f'[按CQI值分组的α统计 - 每个维度的均值]')
    logger.info(f'   CQI值   样本数    各维度α均值(前10维)                           各维度α均值(后10维)')

    stats['correlation_alpha_cqi'] = {}
    stats['correlation_beta_cqi'] = {}
    stats['cqi_groups'] = {}

    # 构建完整的α和β均值矩阵 (16 CQI值 x 52 维度)
    alpha_matrix = np.zeros((16, num_dims))  # 索引0对应CQI=0
    beta_matrix = np.zeros((16, num_dims))
    cqi_counts = np.zeros(16, dtype=int)

    for cqi_val in range(0, 16):
        mask = cqi_rounded == cqi_val
        count = int(mask.sum())
        cqi_counts[cqi_val - 1] = count
        
        if count > 0:
            alpha_group = all_alpha_full[mask]  # (count, 52)
            beta_group = all_beta_full[mask]
            
            alpha_dim_mean = alpha_group.mean(axis=0)
            beta_dim_mean = beta_group.mean(axis=0)
            alpha_dim_std = alpha_group.std(axis=0)
            beta_dim_std = beta_group.std(axis=0)
            
            alpha_matrix[cqi_val - 1] = alpha_dim_mean
            beta_matrix[cqi_val - 1] = beta_dim_mean
            
            stats['cqi_groups'][int(cqi_val)] = {
                'count': count,
                'alpha_dim_mean': alpha_dim_mean,
                'alpha_dim_std': alpha_dim_std,
                'beta_dim_mean': beta_dim_mean,
                'beta_dim_std': beta_dim_std,
            }
    
    # ==================== 打印完整的52维度热力图式总览 ====================
    logger.info('=' * 80)
    logger.info('不同CQI下 α 在所有维度上的均值 (完整52维总览)')
    logger.info('=' * 80)
    logger.info(f'       CQI值:    0     1     2     3     4     5     6     7     8     9    10    11    12    13    14    15')
    cqi_counts_str = '  '.join([f'{cqi_counts[i]:4d}' for i in range(16)])
    logger.info(f'       样本数: {cqi_counts_str}')
    logger.info('-' * 80)

    # 分块打印52个维度（每8维一行，便于阅读）
    for start in range(0, num_dims, 8):
        end = min(start + 8, num_dims)
        dim_labels = [f'Dim{i:02d}' for i in range(start, end)]
        logger.info(f'  维度:  ' + '  '.join([f'{l:>6s}' for l in dim_labels]))
        for cqi_idx, cqi_val in enumerate(range(0, 16)):
            if cqi_counts[cqi_idx] > 0:
                values = alpha_matrix[cqi_idx, start:end]
                row_str = '  '.join([f'{v:>6.4f}' for v in values])
                logger.info(f'  CQI={cqi_val:2d}: {row_str}')
            else:
                row_str = '  '.join(['   N/A' for _ in range(start, end)])
                logger.info(f'  CQI={cqi_val:2d}: {row_str}')
        logger.info('')
    
    # 打印每列（CQI）的均值
    alpha_col_means = alpha_matrix.mean(axis=1)
    logger.info('-' * 80)
    logger.info('  α均值: ' + '  '.join([f'{v:>6.4f}' for v in alpha_col_means]))
    
    logger.info('=' * 80)
    logger.info('不同CQI下 β 在所有维度上的均值 (完整52维总览)')
    logger.info('=' * 80)
    logger.info(f'       CQI值:    0     1     2     3     4     5     6     7     8     9    10    11    12    13    14    15')
    cqi_counts_str = '  '.join([f'{cqi_counts[i]:4d}' for i in range(16)])
    logger.info(f'       样本数: {cqi_counts_str}')
    logger.info('-' * 80)

    for start in range(0, num_dims, 8):
        end = min(start + 8, num_dims)
        dim_labels = [f'Dim{i:02d}' for i in range(start, end)]
        logger.info(f'  维度:  ' + '  '.join([f'{l:>6s}' for l in dim_labels]))
        for cqi_idx, cqi_val in enumerate(range(0, 16)):
            if cqi_counts[cqi_idx] > 0:
                values = beta_matrix[cqi_idx, start:end]
                row_str = '  '.join([f'{v:>6.3f}' for v in values])
                logger.info(f'  CQI={cqi_val:2d}: {row_str}')
            else:
                row_str = '  '.join(['   N/A' for _ in range(start, end)])
                logger.info(f'  CQI={cqi_val:2d}: {row_str}')
        logger.info('')
    
    beta_col_means = beta_matrix.mean(axis=1)
    logger.info('-' * 80)
    logger.info('  β均值: ' + '  '.join([f'{v:>6.3f}' for v in beta_col_means]))
    
    logger.info('=' * 70)
    
    # 计算全局α/β均值与CQI的相关性
    alpha_global_mean = all_alpha_full.mean(axis=1)  # (N,) 每个样本的全局均值
    beta_global_mean = all_beta_full.mean(axis=1)
    
    correlation_alpha_cqi = np.corrcoef(cqi_rounded, alpha_global_mean)[0, 1]
    correlation_beta_cqi = np.corrcoef(cqi_rounded, beta_global_mean)[0, 1]
    
    logger.info(f'[全局α均值与CQI的相关性]')
    logger.info(f'   Pearson相关系数: {correlation_alpha_cqi:.4f}')
    logger.info(f'[全局β均值与CQI的相关性]')
    logger.info(f'   Pearson相关系数: {correlation_beta_cqi:.4f}')
    
    # 计算每个维度上α与CQI的相关性
    logger.info(f'[各维度α与CQI的相关性]')
    alpha_corr_per_dim = []
    for dim_idx in range(num_dims):
        corr = np.corrcoef(cqi_rounded, all_alpha_full[:, dim_idx])[0, 1]
        alpha_corr_per_dim.append(corr)
    alpha_corr_per_dim = np.array(alpha_corr_per_dim)
    
    logger.info(f'   各维度相关性: min={alpha_corr_per_dim.min():.4f}, max={alpha_corr_per_dim.max():.4f}, mean={alpha_corr_per_dim.mean():.4f}')
    logger.info(f'   前10维相关性: {[f"{v:.3f}" for v in alpha_corr_per_dim[:10]]}')
    logger.info(f'   后10维相关性: {[f"{v:.3f}" for v in alpha_corr_per_dim[-10:]]}')
    
    logger.info(f'[各维度β与CQI的相关性]')
    beta_corr_per_dim = []
    for dim_idx in range(num_dims):
        corr = np.corrcoef(cqi_rounded, all_beta_full[:, dim_idx])[0, 1]
        beta_corr_per_dim.append(corr)
    beta_corr_per_dim = np.array(beta_corr_per_dim)
    
    logger.info(f'   各维度相关性: min={beta_corr_per_dim.min():.4f}, max={beta_corr_per_dim.max():.4f}, mean={beta_corr_per_dim.mean():.4f}')
    logger.info(f'   前10维相关性: {[f"{v:.3f}" for v in beta_corr_per_dim[:10]]}')
    logger.info(f'   后10维相关性: {[f"{v:.3f}" for v in beta_corr_per_dim[-10:]]}')
    
    logger.info('=' * 70)

    # 解释结论
    logger.info('[结论]')
    if correlation_alpha_cqi > 0.3:
        logger.info(f'   ✓ 全局α均值与CQI呈正相关 (r={correlation_alpha_cqi:.3f})')
        logger.info(f'     → CQI越大，α越大，模型越倾向于使用SR增强')
    elif correlation_alpha_cqi < -0.3:
        logger.info(f'   ✓ 全局α均值与CQI呈负相关 (r={correlation_alpha_cqi:.3f})')
        logger.info(f'     → CQI越大，α越小，模型越倾向于保留原始CSI')
    else:
        logger.info(f'   △ 全局α均值与CQI相关性较弱 (r={correlation_alpha_cqi:.3f})')
        logger.info(f'     → CQI对α的调节作用不明显')

    if stats['alpha']['mean'] < 0.1:
        logger.info(f'   △ α 均值很低 ({stats["alpha"]["mean"]:.4f})，模型主要保留原始CSI')
    elif stats['alpha']['mean'] > 0.5:
        logger.info(f'   △ α 均值较高 ({stats["alpha"]["mean"]:.4f})，模型主要使用SR增强')
    else:
        logger.info(f'   △ α 均值适中 ({stats["alpha"]["mean"]:.4f})，模型在两者间平衡')

    logger.info('=' * 70)

    # 返回热力图所需的数据
    return stats, alpha_matrix, beta_matrix, cqi_counts


def run_test(args):
    """运行测试流程"""
    
    # 创建配置对象
    config = TestConfig()
    
    # 覆盖命令行参数
    if args.data_dir:
        config.data_dir = args.data_dir
    if args.model_path:
        config.model_path = args.model_path
    if args.cqi_type:
        config.cqi_type = args.cqi_type
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.seed:
        config.seed = args.seed
    if args.stratify_by_cqi:
        config.stratify_by_cqi = True
    if args.cqi_bins is not None:
        config.cqi_bins = args.cqi_bins
    
    # 设置模型类型（需要从模型路径推断或手动指定）
    if args.model_type:
        config.model_type = args.model_type

    # CRNet/CLNet 不使用 CQI，强制设置为 nan
    if config.model_type in ['crnet', 'clnet']:
        config.cqi_type = 'nan'
        logger.info(f'=> 检测到 {config.model_type.upper()} 模型，自动设置 cqi_type=nan')
    elif config.model_type in ['crnet_cqi', 'clnet_cqi']:
        # CQI 模型需要设置正确的 cqi_type
        if config.cqi_type == 'nan':
            logger.warning(f'=> 检测到 {config.model_type.upper()} 模型但 cqi_type=nan，将自动设置为 wide')
            config.cqi_type = 'wide'
        logger.info(f'=> 检测到 {config.model_type.upper()} 模型，CQI嵌入模块已启用')
    if args.sr_hidden_channels:
        config.sr_hidden_channels = args.sr_hidden_channels
    if args.d_model:
        config.d_model = args.d_model
    if args.cr:
        config.cr = args.cr
    if args.num_encoder_layers:
        config.num_encoder_layers = args.num_encoder_layers
    if args.num_decoder_layers:
        config.num_decoder_layers = args.num_decoder_layers
    if args.nhead:
        config.nhead = args.nhead
    if args.dropout is not None:
        config.dropout = args.dropout
    
    # 设置时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 打印配置信息
    logger.info('=' * 80)
    logger.info('TransCQA/TransNet/CRNet/CLNet 模型测试')
    logger.info('=' * 80)
    logger.info(f'=> 时间戳: {timestamp}')
    logger.info(f'=> PyTorch版本: {torch.__version__}')
    logger.info(f'=> 模型类型: {config.model_type.upper()}')
    logger.info(f'=> CQI类型: {config.cqi_type.upper()} (CRNet/CLNet 不使用 CQI，固定为 nan)')
    logger.info(f'=> 压缩比: 1/{config.cr}')
    logger.info(f'=> 数据目录: {config.data_dir}')
    logger.info(f'=> 模型路径: {config.model_path}')
    logger.info('=> 测试配置:')
    logger.info(f'     stratify_by_cqi: {config.stratify_by_cqi}')
    logger.info(f'     cqi_bins: {config.cqi_bins}')
    logger.info(f'     seed: {config.seed}')
    logger.info(f'     batch_size: {config.batch_size}')
    logger.info(f'     d_model: {config.d_model}')
    logger.info(f'     cr: {config.cr}')
    logger.info('=' * 80)
    
    # 环境初始化（使用与训练时相同的随机种子）
    device, pin_memory = init_device(
        seed=config.seed,
        cpu=config.device.type == 'cpu',
        gpu=config.device.index if config.device.type == 'cuda' else None
    )
    config.device = device
    
    # 设置随机种子（确保数据划分一致）
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    logger.info(f'=> 设备: {device}')
    logger.info(f'=> 随机种子: {config.seed}（确保数据划分与训练时一致）')
    
    # 创建数据加载器（确保使用相同的 stratify_by_cqi 和 cqi_bins 设置）
    logger.info('=> 创建数据加载器...')
    logger.info(f'   stratify_by_cqi: {config.stratify_by_cqi}')
    logger.info(f'   cqi_bins: {config.cqi_bins}')
    
    train_loader, val_loader, test_loader = LosAngelesDataLoader(
        root=config.data_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        cqi_type=config.cqi_type,
        stratify_by_cqi=config.stratify_by_cqi,
        cqi_bins=config.cqi_bins
    )()
    
    # 获取数据集大小
    train_size = len(train_loader.dataset)
    val_size = len(val_loader.dataset)
    test_size = len(test_loader.dataset)
    total_size = train_size + val_size + test_size
    
    print_dataset_info(train_size, val_size, test_size, total_size)
    
    # 创建模型
    logger.info('=> 创建模型...')
    logger.info(f'=> 模型类型: {config.model_type}')
    model = create_model(config)
    model.to(device)
    
    # 加载预训练模型
    checkpoint, epoch, best_rho = load_model(model, config.model_path, device, logger)
    if checkpoint is None:
        logger.error('❌ 模型加载失败，测试终止')
        return
    
    # 提取并打印 CQI 嵌入参数（仅对 crnet_cqi 和 clnet_cqi 模型）
    cqi_stats = None
    alpha_matrix = None
    beta_matrix = None
    cqi_counts = None
    if config.model_type in ['crnet_cqi', 'clnet_cqi']:
        logger.info(f'=> 提取 {config.model_type.upper()} 模型的 CQI 嵌入参数...')
        result = extract_cqi_parameters(model, test_loader, device, logger)
        if result is not None:
            cqi_stats, alpha_matrix, beta_matrix, cqi_counts = result
    
    # 定义损失函数
    criterion = nn.MSELoss().to(device)
    
    # 创建测试器
    logger.info('=> 开始测试...')
    tester = Tester(model, device, criterion, config.cqi_type)
    
    # 运行测试
    loss, rho, nmse = tester(test_loader)
    
    # 打印测试结果
    logger.info('=' * 80)
    logger.info('测试结果汇总')
    logger.info('=' * 80)
    logger.info(f'=> 模型: {config.model_path}')
    logger.info(f'=> 训练轮数: {epoch}')
    logger.info(f'=> 测试样本数: {test_size}')
    logger.info(f'=> Test Loss: {loss:.6e}')
    logger.info(f'=> Test Rho: {rho:.6f}')
    if nmse > 0:
        logger.info(f'=> Test NMSE: {nmse:.6f} ({10 * np.log10(nmse):.3f} dB)')
    else:
        logger.info(f'=> Test NMSE: {nmse:.6f} (无法计算dB，NMSE非正值)')
    logger.info('=' * 80)
    
    # 保存测试结果
    results_dir = os.path.dirname(config.model_path)
    if not results_dir:
        results_dir = '.'
    results_file = os.path.join(
        results_dir,
        f'test_results_{config.cqi_type}_{timestamp}.txt'
    )
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write('TransCQA/TransNet 测试结果\n')
        f.write('=' * 60 + '\n')
        f.write(f'时间戳: {timestamp}\n')
        f.write(f'模型路径: {config.model_path}\n')
        f.write(f'训练轮数: {epoch}\n')
        f.write(f'CQI类型: {config.cqi_type}\n')
        f.write(f'测试样本数: {test_size}\n')
        f.write(f'随机种子: {config.seed}\n')
        f.write(f'stratify_by_cqi: {config.stratify_by_cqi}\n')
        f.write(f'cqi_bins: {config.cqi_bins}\n')
        f.write('-' * 60 + '\n')
        f.write(f'Test Loss: {loss:.6e}\n')
        f.write(f'Test Rho: {rho:.6f}\n')
        f.write(f'Test NMSE: {nmse:.6f}')
        if nmse > 0:
            f.write(f' ({10 * np.log10(nmse):.3f} dB)\n')
        else:
            f.write(f' (无法计算dB)\n')

        # 如果是CQI模型，保存嵌入参数统计
        if cqi_stats is not None:
            f.write('=' * 60 + '\n')
            f.write('CQI 嵌入模块 α 和 β 输出统计\n')
            f.write('=' * 60 + '\n')
            f.write(f'调制公式: output = α * SR(x) + (1 - α) * x + β\n')
            f.write('-' * 60 + '\n')
            f.write('[α 输出统计]\n')
            f.write(f'  均值: {cqi_stats["alpha"]["mean"]:.6f}\n')
            f.write(f'  标准差: {cqi_stats["alpha"]["std"]:.6f}\n')
            f.write(f'  范围: [{cqi_stats["alpha"]["min"]:.6f}, {cqi_stats["alpha"]["max"]:.6f}]\n')
            f.write(f'  中位数: {cqi_stats["alpha"]["median"]:.6f}\n')
            f.write('-' * 60 + '\n')
            f.write('[β 输出统计]\n')
            f.write(f'  均值: {cqi_stats["beta"]["mean"]:.6f}\n')
            f.write(f'  标准差: {cqi_stats["beta"]["std"]:.6f}\n')
            f.write(f'  范围: [{cqi_stats["beta"]["min"]:.6f}, {cqi_stats["beta"]["max"]:.6f}]\n')
            f.write(f'  中位数: {cqi_stats["beta"]["median"]:.6f}\n')
            f.write('=' * 60 + '\n')
        
        f.write('=' * 60 + '\n')
    
    logger.info(f'=> 测试结果已保存到: {results_file}')

    # ==================== 绘制 α 和 β 热力图（仅对 CQI 模型） ====================
    if alpha_matrix is not None and beta_matrix is not None:
        logger.info('=> 开始绘制 α 和 β 热力图...')
        heatmap_path = plot_alpha_beta_heatmaps(
            alpha_matrix=alpha_matrix,
            beta_matrix=beta_matrix,
            cqi_counts=cqi_counts,
            output_dir=results_dir
        )
        logger.info(f'=> 热力图已保存到: {heatmap_path}')

    return loss, rho, nmse


def main():
    parser = argparse.ArgumentParser(description='TransCQA/TransNet 模型测试')
    
    # 必需参数
    parser.add_argument('--model-path', type=str, default="/data/wutong/Wireless_AI_Research_DateSet/01105/DFT_120deg/01105_mapping2_N-135_120deg/CRNet/Result_2_32_52_d16/BaseModel_Finetuned_CQI/cr_64/wide/best_nmse_wide.pth",
                        help='模型检查点路径')
    
    # 数据相关参数
    parser.add_argument('--data-dir', type=str, default='/data/wutong/Wireless_AI_Research_DateSet/01105/DFT_120deg/01105_mapping2_N-135_120deg/Wireless_AI_Research_Dataset',
                        help='数据目录路径')
    parser.add_argument('--cqi-type', type=str, default='wide',
                        choices=['wide', 'sub', 'nan'],
                        help='CQI类型（crnet/clnet使用nan）')
    parser.add_argument('--stratify-by-cqi', type=bool,default=True,
                        help='是否按CQI分层划分（必须与训练时一致）')
    parser.add_argument('--cqi-bins', type=int, default=None,
                        help='CQI分组数量（必须与训练时一致）')
    
    # 模型参数（用于创建模型）
    parser.add_argument('--model-type', type=str, default='crnet_cqi',
                        choices=['multiscale', 'transnet_moe', 'transnet_qmod', 
                                 'transnet_qmodplus', 'transnet_qmodplusd', 'transnet_qc',
                                 'crnet', 'clnet', 'crnet_cqi', 'clnet_cqi'],
                        help='模型类型')
    parser.add_argument('--d-model', type=int, default=16,
                        help='Transformer特征维度')
    parser.add_argument('--cr', type=int, default=64,
                        help='压缩比（CRNet/CLNet/CRNet_CQI/CLNet_CQI 常用 4, 8, 16, 32）')
    parser.add_argument('--sr-hidden-channels', type=int, default=32,
                        help='CQI模型超分辨率模块隐藏层通道数')
    parser.add_argument('--num-encoder-layers', type=int, default=2,
                        help='编码器层数')
    parser.add_argument('--num-decoder-layers', type=int, default=2,
                        help='解码器层数')
    parser.add_argument('--nhead', type=int, default=4,
                        help='注意力头数')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout率')
    
    # 训练参数
    parser.add_argument('-b', '--batch-size', type=int, default=256,
                        help='批大小')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子（必须与训练时一致）')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='数据加载工作进程数')
    
    # 设备参数
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU ID')
    parser.add_argument('--cpu', action='store_true',
                        help='使用CPU')
    
    args = parser.parse_args()
    
    # 设置设备
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    if args.cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    # 运行测试
    run_test(args)


if __name__ == "__main__":
    main()
