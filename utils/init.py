import os
import random
import thop
import torch
import torch.nn as nn
import copy


def init_device(seed=None, cpu=None, gpu=None, affinity=None):
    """初始化设备和环境"""
    # 设置CPU亲和性
    if affinity is not None:
        os.system(f'taskset -p {affinity} {os.getpid()}')

    # 设置随机种子
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    # 设置GPU ID
    if gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    # 设备选择
    if cpu:
        device = torch.device('cpu')
        pin_memory = False
    elif gpu is not None and torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu}')
        #device = torch.device('cuda:0')
        pin_memory = True
    elif torch.cuda.is_available():
        device = torch.device('cuda:0')
        pin_memory = True
    else:
        device = torch.device('cpu')
        pin_memory = False

    return device, pin_memory


def clear_thop_hooks(model):
    """清理thop注册的所有hook"""
    def remove_hooks(module):
        # 移除所有forward hook
        if hasattr(module, '_forward_hooks'):
            hooks_to_remove = []
            for hook_id, hook in module._forward_hooks.items():
                # 检查是否是thop相关的hook
                if hasattr(hook, '__module__') and 'thop' in str(hook.__module__):
                    hooks_to_remove.append(hook_id)
            for hook_id in hooks_to_remove:
                del module._forward_hooks[hook_id]
        
        # 递归清理子模块
        for child in module.children():
            remove_hooks(child)
    
    remove_hooks(model)


def safe_flops_calculation(model, config):
    """安全的FLOPs计算，避免训练时设备冲突"""
    try:
        # 创建模型副本用于FLOPs计算
        model_copy = copy.deepcopy(model)
        device = config.device
        
        # 将副本移动到指定设备
        model_copy = model_copy.to(device)
        
        # 创建示例输入
        H_sample = torch.randn([1, 2, 32, 52]).to(device)
        
        # 根据CQI类型创建合适的输入（这里只需要形状，具体值不重要）
        inputs = (H_sample,)
        
        # 计算FLOPs
        flops, params = thop.profile(model_copy, inputs=inputs, verbose=False)
        flops, params = thop.clever_format([flops, params], "%.3f")
        
        # 清理副本上的hook
        clear_thop_hooks(model_copy)
        
        # 删除副本释放内存
        del model_copy
        
        return flops, params
        
    except Exception as e:
        total_params = sum(p.numel() for p in model.parameters())
        params = f"{total_params/1e6:.3f}M"
        flops = "N/A"
        return flops, params


from models.TransNet import *
from models.TransNet_MoE import Transformer_MoE
from models.TransNet_H import *
from models.TransNet_QModPlusD import *
from models.crnet import crnet
from models.clnet import clnet
from models.CQIFineTuner import CQIFineTuner

# 导入 logger 模块的函数（使用相对导入避免循环依赖）
from .logger import debug, info, emph, warning, error, fatal, line_seg

# 创建 logger 对象供兼容使用
class _LoggerProxy:
    """logger 代理类，兼容 logger.info() 等调用方式"""
    debug = staticmethod(debug)
    info = staticmethod(info)
    emph = staticmethod(emph)
    warning = staticmethod(warning)
    error = staticmethod(error)
    fatal = staticmethod(fatal)

logger = _LoggerProxy()


def init_model(config):
    """模型初始化函数
    
    支持的模型类型：
    - 'crnet': CRNet 压缩恢复网络
    - 'clnet': CLNet 网络  
    - 'transnet': TransNet 基础模型
    - 'transnet_moe': TransNet MoE 变体
    - 'transnet_h': TransNet_H 变体
    - 'transnet_qmodplusd': TransNet_QModPlusD 变体
    - 'crnet_cqi': CRNet + CQI 嵌入模块（用于微调）
    - 'clnet_cqi': CLNet + CQI 嵌入模块（用于微调）
    - 'transnet_qmodplusd_cqi': TransNet_QModPlusD + CQI 嵌入模块（用于微调）
    """
    model_type = getattr(config, 'model_type', 'crnet')
    print("当前使用模型：", model_type)
    
    # 先创建基础模型
    base_model = None
    
    if model_type == 'crnet':
        from models.crnet import crnet
        base_model = crnet(reduction=config.cr)
    elif model_type == 'clnet':
        from models.clnet import clnet
        base_model = clnet(reduction=config.cr)
    elif model_type == 'transnet_moe':
        from models.TransNet_MoE import create_transnet_moe
        base_model = create_transnet_moe(
            reduction=config.cr,
            d_model=config.d_model,
            num_experts=getattr(config, 'moe_experts', 8),
            mode=config.cqi_type,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            nhead=config.nhead,
            dropout=config.dropout,
            snr=config.snr,
            device=str(config.device),
            enable_channel=getattr(config, 'moe_enable_channel', False),
            top_k=getattr(config, 'moe_top_k', None)
        )
    elif model_type == 'transnet_h':
        from models.TransNet_H import create_transnet_h
        base_model = create_transnet_h(
            reduction=config.cr,
            d_model=config.d_model,
            mode=config.cqi_type,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            nhead=config.nhead,
            dropout=config.dropout
        )
    elif model_type == 'transnet':
        from models.TransNet import transnet
        base_model = transnet(
            reduction=config.cr,
            d_model=config.d_model
        )
    elif model_type == 'transnet_qmodplusd':
        from models.TransNet_QModPlusD import create_transnet_qmodplusd
        base_model = create_transnet_qmodplusd(
            reduction=config.cr,
            d_model=config.d_model,
            mode=config.cqi_type,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            nhead=config.nhead,
            dropout=config.dropout
        )
    elif model_type == 'crnet_cqi':
        from models.crnet_cqi import crnet_cqi
        base_model = crnet_cqi(
            reduction=config.cr,
            sr_hidden_channels=getattr(config, 'sr_hidden_channels', 16)
        )
    elif model_type == 'crnet_cqi_film':
        from models.crnet_cqi_film import crnet_cqi_film
        base_model = crnet_cqi_film(
            reduction=config.cr,
            film_hidden_dim=getattr(config, 'film_hidden_dim', 32)
        )
    elif model_type == 'clnet_cqi':
        from models.clnet_cqi import clnet_cqi
        base_model = clnet_cqi(
            reduction=config.cr,
            sr_hidden_channels=getattr(config, 'sr_hidden_channels', 16)
        )
    elif model_type == 'transnet_qmodplusd_cqi':
        from models.transnet_qmodplusd_cqi import transnet_qmodplusd_cqi
        base_model = transnet_qmodplusd_cqi(
            reduction=config.cr,
            d_model=config.d_model,
            mode=config.cqi_type,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            nhead=config.nhead,
            dropout=config.dropout,
            sr_hidden_channels=getattr(config, 'sr_hidden_channels', 16)
        )
    else:
        logger.warning(f"未知的model_type='{model_type}'，回退为'crnet'")
        from models.crnet import crnet
        base_model = crnet(reduction=config.cr)

    model = base_model

    # 为避免训练时设备冲突，默认跳过FLOPs计算
    try:
        total_params = sum(p.numel() for p in model.parameters())
        params = f"{total_params/1e6:.3f}M"
        flops = "N/A (跳过以避免训练时设备冲突)"
        logger.info("为避免训练时设备冲突，跳过FLOPs计算")
    except Exception as e:
        logger.warning(f"参数计算失败: {e}")
        params = "N/A"
        flops = "N/A"

    # 获取模型信息（各模型需实现get_model_info）
    model_info = model.get_model_info() if hasattr(model, 'get_model_info') else {'use_cqi_fusion': False}

    # 记录基本信息
    logger.info(f"=> 模型名称: {model_type}")
    logger.info(f"=> CQI类型: {config.cqi_type.upper()}")
    logger.info(f"=> 模型配置: 压缩比=1/{config.cr}, d_model={config.d_model}")
    logger.info(f"=> 编码器层数: {config.num_encoder_layers}, 解码器层数: {config.num_decoder_layers}")
    logger.info(f"=> 注意力头数: {config.nhead}, Dropout: {config.dropout}")
    logger.info(f"=> 模型FLOPs: {flops}")
    logger.info(f"=> 模型参数数量: {params}")

    if model_info.get('use_cqi_fusion', False):
        logger.info(f"=> CQI融合参数: {model_info.get('fusion_params', 0):,} ({model_info.get('fusion_ratio', 0.0):.2%})")
    
    logger.info(f'{line_seg}\n{model}\n{line_seg}\n')

    ## 测试预训练模型加载是否正确
    

    return model, params, flops


def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """加载检查点"""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
    
    logger.info(f'=> 正在加载检查点 {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path)
    
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    
    best_rho = checkpoint.get('best_rho', None)
    best_nmse = checkpoint.get('best_nmse', None)
    
    logger.info(f"=> 成功加载检查点 {checkpoint_path} (epoch {checkpoint['epoch']})")
    
    return start_epoch + 1, best_rho, best_nmse


def save_checkpoint(state, save_path, filename):
    """保存检查点"""
    if save_path is None:
        logger.warning('没有指定保存路径')
        return

    os.makedirs(save_path, exist_ok=True)
    filepath = os.path.join(save_path, filename)
    torch.save(state, filepath)
    logger.info(f'检查点已保存: {filepath}')


def count_parameters(model):
    """统计模型参数"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }


def print_model_summary(model, input_shape=(1, 2, 32, 52)):
    """打印模型摘要"""
    logger.info("模型结构摘要:")
    logger.info("-" * 60)
    
    param_counts = count_parameters(model)
    logger.info(f"总参数数量: {param_counts['total']:,}")
    logger.info(f"可训练参数: {param_counts['trainable']:,}")
    logger.info(f"不可训练参数: {param_counts['non_trainable']:,}")
    
    # 尝试打印每层参数数量
    try:
        logger.info("\n各模块参数统计:")
        for name, module in model.named_children():
            module_params = sum(p.numel() for p in module.parameters())
            logger.info(f"  {name}: {module_params:,} parameters")
    except Exception as e:
        logger.warning(f"无法打印详细参数统计: {e}")
    
    logger.info("-" * 60)


__all__ = ["init_device", "init_model", "load_checkpoint", "save_checkpoint", 
           "count_parameters", "print_model_summary",
           "logger", "line_seg", "debug", "info", "emph", "warning", "error", "fatal"]
