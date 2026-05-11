import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免显示问题
import matplotlib.pyplot as plt
from config import parse_args
from utils import logger, Trainer, Tester
from utils import init_device, init_model, FakeLR, WarmUpCosineAnnealingLR, LinearDecayLR
from dataloader.losangeles import LosAngelesDataLoader
from tensorboardX import SummaryWriter
import os
import time
#os.environ['CUDA_VISIBLE_DEVICES'] = '3' 
# tensorboard --logdir /data/wutong/Wireless_AI_Research_DateSet/BaseModel/base_model/cr_16/sites_25/20260201_154604/tensorboard
# tensorboard --logdir /data/wutong/WDataResult/TransNet_Q/cr_32/wide/tensorboard

def load_pretrained_model(model, pretrained_path, cqi_type, logger):
    """加载预训练模型"""
    if not os.path.exists(pretrained_path):
        logger.warning(f'❌ 预训练模型不存在: {pretrained_path}')
        return False
    
    try:
        logger.info(f'📦 加载预训练模型: {pretrained_path}')
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        
        # 检查checkpoint格式
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            logger.info(f'   训练轮数: {checkpoint.get("epoch", "未知")}')
            logger.info(f'   最佳Rho: {checkpoint.get("best_rho", "未知")}')
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # 处理键名映射（如果模型被包装过）
        if hasattr(model, 'base_model'):
            new_state_dict = {}
            model_keys = set(model.state_dict().keys())
            for k, v in state_dict.items():
                if k in model_keys:
                    new_state_dict[k] = v
                elif f'base_model.{k}' in model_keys:
                    new_state_dict[f'base_model.{k}'] = v
                else:
                    new_state_dict[k] = v
            state_dict = new_state_dict
        
        # 加载模型参数
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        
        if missing:
            logger.warning(f'   缺失键: {missing[:5]}...' if len(missing) > 5 else f'   缺失键: {missing}')
        if unexpected:
            logger.warning(f'   多余键: {unexpected[:5]}...' if len(unexpected) > 5 else f'   多余键: {unexpected}')
        
        logger.info(f'✅ 预训练模型加载成功，将用于训练 {cqi_type.upper()} CQI')
        
        return True
        
    except Exception as e:
        logger.error(f'❌ 预训练模型加载失败: {e}')
        return False


def main():
    # 解析配置
    config = parse_args()
    # 确保config有timestamp属性
    if not hasattr(config, 'timestamp') or config.timestamp is None:
        from datetime import datetime
        config.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 设置日志文件名包含时间戳
    log_filename = f'transcqa_{config.cqi_type}_{config.timestamp}.log'
    
    # 打印配置信息 
    logger.info('=' * 80)
    logger.info('TransCQA Training Started')
    logger.info('=' * 80)
    logger.info(f'=> 时间戳: {config.timestamp}')
    logger.info(f'=> PyTorch版本: {torch.__version__}')
    logger.info(f'=> CQI类型: {config.cqi_type.upper()}')
    logger.info(f'=> 使用预训练: {"是" if config.use_pretrained else "否"}')
    if config.use_pretrained:
        logger.info(f'=> 预训练模型: {config.pretrained}')
    logger.info('=> 训练配置:')
    for key, value in vars(config).items():
        if key not in ['timestamp']:  # 跳过已显示的参数
            logger.info(f'     {key}: {value}')
    logger.info('=' * 80)
    
    # 环境初始化
    device, pin_memory = init_device(
        seed=config.seed, 
        cpu=(config.device.type == 'cpu'), 
        gpu=(config.device.index if config.device.type == 'cuda' else None)
    )
    config.device = device
    
    # 创建数据加载器
    logger.info('=> 创建数据加载器...')
    if hasattr(config, 'stratify_by_cqi') and config.stratify_by_cqi:
        logger.info('=> 使用按CQI分组的数据划分方式（每组内8:1:1）')
    train_loader, val_loader, test_loader = LosAngelesDataLoader(
        root=config.data_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        cqi_type=config.cqi_type,
        stratify_by_cqi=getattr(config, 'stratify_by_cqi', False),
        cqi_bins=getattr(config, 'cqi_bins', None)
    )()
    
    logger.info('=> 数据加载器创建成功')
    
    # 定义模型
    logger.info('=> 创建模型...')
    model,params,flops = init_model(config)
    model.to(device)
    
    # ============================================================
    # CQI 嵌入模型：加载预训练权重并设置冻结策略
    # ============================================================
    is_cqi_model = config.model_type in ['crnet_cqi', 'crnet_cqi_film', 'clnet_cqi', 'transnet_qmodplusd_cqi','crnet','clnet']
    freeze_mode = getattr(config, 'freeze_mode', 'none')
    
    # 编码器相关模块名称集合（兼容不同模型结构）
    ENCODER_MODULES = {'encoder', 'encoder1', 'encoder2', 'encoder_conv', 'encoder_fc'}
    # 解码器相关模块名称集合
    DECODER_MODULES = {'decoder', 'decoder_feature', 'decoder_fc'}
    # 嵌入模块名称集合
    EMBEDDING_MODULES = {'cqi_modulator', 'sr_module'}
    
    print(f"CQI模型: {is_cqi_model}")
    print(f"冻结模式: {freeze_mode}")
    print(f"预训练路径: {config.pretrained}")
    
    if is_cqi_model and config.pretrained is not None and os.path.isfile(config.pretrained):
        logger.info(f"=> 加载预训练模型到 CQI 嵌入模型: {config.pretrained}")
        checkpoint = torch.load(config.pretrained, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # 加载预训练权重（键名完全匹配）
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        logger.info(f"   预训练权重加载完成，缺失键: {len(missing)}, 多余键: {len(unexpected)}")
        
        # 根据冻结模式设置参数冻结策略
        if freeze_mode != 'none':
            logger.info(f"=> 冻结模式: {freeze_mode}")
            
            frozen_params = 0
            trainable_params = 0
            encoder_params = 0
            decoder_params = 0
            embedding_params = 0
            
            for name, param in model.named_parameters():
                # 获取第一层模块名（处理复合名称如 encoder1, encoder_fc）
                first_module = name.split('.')[0]
                
                # 特殊处理 encoder_fc/decoder_fc：它们同时属于编码器和解码器
                # 在 CRNet 中：encoder_fc 是编码器的最后层，decoder_fc 是解码器的第一层
                is_encoder_module = first_module in ENCODER_MODULES
                is_decoder_module = first_module in DECODER_MODULES
                is_embedding_module = first_module in EMBEDDING_MODULES
                
                # encoder_fc/decoder_fc 的归属判断
                # encoder_fc 在编码器侧，decoder_fc 在解码器侧
                if first_module == 'encoder_fc':
                    is_encoder_module = True
                    is_decoder_module = False
                elif first_module == 'decoder_fc':
                    is_encoder_module = False
                    is_decoder_module = True
                
                # 确定冻结策略
                if freeze_mode == 'all':
                    # 模式1: 冻结全部基础模型，只训练嵌入模块
                    if is_embedding_module:
                        param.requires_grad = True
                        trainable_params += param.numel()
                        embedding_params += param.numel()
                    else:
                        param.requires_grad = False
                        frozen_params += param.numel()
                        
                elif freeze_mode == 'encoder_only':
                    # 模式2: 冻结编码器，训练解码器和嵌入模块
                    if is_encoder_module:
                        param.requires_grad = False
                        frozen_params += param.numel()
                        encoder_params += param.numel()
                    else:
                        param.requires_grad = True
                        trainable_params += param.numel()
                        if is_decoder_module:
                            decoder_params += param.numel()
                        elif is_embedding_module:
                            embedding_params += param.numel()
                            
                elif freeze_mode == 'freeze_decoder':
                    # 模式3: 冻结解码器，训练编码器和嵌入模块
                    if is_decoder_module:
                        param.requires_grad = False
                        frozen_params += param.numel()
                        decoder_params += param.numel()
                    else:
                        param.requires_grad = True
                        trainable_params += param.numel()
                        if is_encoder_module:
                            encoder_params += param.numel()
                        elif is_embedding_module:
                            embedding_params += param.numel()
                else:
                    # 默认模式：所有参数可训练
                    param.requires_grad = True
                    trainable_params += param.numel()
            
            # 打印冻结统计信息
            logger.info(f'   总冻结参数: {frozen_params:,}')
            if encoder_params > 0:
                logger.info(f'   └─ 编码器参数: {encoder_params:,}')
            if decoder_params > 0:
                logger.info(f'   └─ 解码器参数: {decoder_params:,}')
            logger.info(f'   总可训练参数: {trainable_params:,}')
            if embedding_params > 0:
                logger.info(f'   └─ 嵌入模块参数: {embedding_params:,}')
            
            mode_descriptions = {
                'all': '冻结全部基础模型',
                'encoder_only': '冻结编码器',
                'freeze_decoder': '冻结解码器'
            }
            logger.info(f'=> 冻结模式详情: {mode_descriptions.get(freeze_mode, freeze_mode)}')
    
    # 验证模型设备一致性
    logger.info('=> 验证模型设备一致性...')
    try:
        model_device = next(model.parameters()).device
        if model_device != device:
            logger.warning(f'模型设备 ({model_device}) 与目标设备 ({device}) 不匹配，重新同步...')
            model = model.to(device)
        logger.info(f'✅ 模型已正确加载到设备: {device}')
    except Exception as e:
        logger.warning(f'设备验证时出现问题: {e}')
    
    # 定义损失函数
    criterion = nn.MSELoss().to(device)
    logger.info('=> 损失函数: MSE Loss')
    
    # 推理模式
    if config.evaluate:
        logger.info('=> 评估模式')
        tester = Tester(model, device, criterion, config.cqi_type)
        loss, rho, nmse = tester(test_loader)
        
        logger.info(f'=> 最终评估结果:')
        logger.info(f'     Test Loss: {loss:.3e}')
        logger.info(f'     Test Rho: {rho:.3e}')
        logger.info(f'     Test NMSE: {nmse:.3e}')
        return
    
    # 定义优化器和调度器
    logger.info('=> 设置优化器和调度器...')
    lr_init = config.lr
    
    # 检查模型类型，为MoE模型的Router设置独立学习率
    if getattr(config, 'model_type', '') == 'transnet_moe':
        # 分离Router参数和Expert参数
        router_params = []
        expert_params = []
        other_params = []
        
        for name, param in model.named_parameters():
            if 'router' in name:
                router_params.append(param)
            elif 'moe_ffn.experts' in name:
                expert_params.append(param)
            else:
                other_params.append(param)
        
        # 设置不同的学习率
        lr_router = getattr(config, 'router_lr', lr_init * 10)  # Router使用10倍学习率（可配置）
        lr_expert = lr_init
        lr_other = lr_init
        
        param_groups = [
            {'params': router_params, 'lr': lr_router, 'name': 'router'},
            {'params': expert_params, 'lr': lr_expert, 'name': 'experts'},
            {'params': other_params, 'lr': lr_other, 'name': 'other'}
        ]
        
        # 优化器选择
        opt_type = getattr(config, 'optimizer', 'adam')
        if opt_type == 'adamw':
            optimizer = torch.optim.AdamW(param_groups, lr=lr_init, weight_decay=getattr(config, 'weight_decay', 0.0))
            logger.info(f'=> 优化器: AdamW（分组学习率）')
        elif opt_type == 'sgd':
            optimizer = torch.optim.SGD(
                param_groups, lr=lr_init,
                momentum=getattr(config, 'momentum', 0.9),
                weight_decay=getattr(config, 'weight_decay', 0.0),
                nesterov=True
            )
            logger.info(f'=> 优化器: SGD（分组学习率），momentum={getattr(config, "momentum", 0.9)}, weight_decay={getattr(config, "weight_decay", 0.0)}')
        else:
            optimizer = torch.optim.Adam(param_groups, lr=lr_init)
            logger.info(f'=> 优化器: Adam（分组学习率）')
        
        logger.info(f'  - Router学习率: {lr_router:.2e}（{lr_router/lr_init:.1f}×基准）')
        logger.info(f'  - Expert学习率: {lr_expert:.2e}')
        logger.info(f'  - 其他参数学习率: {lr_other:.2e}')
    else:
        # 非MoE模型，使用标准方式
        # 根据冻结模式获取可训练参数
        if is_cqi_model and freeze_mode != 'none':
            # 只优化可训练参数
            trainable_params_list = [p for p in model.parameters() if p.requires_grad]
            trainable_count = sum(p.numel() for p in trainable_params_list)
            logger.info(f'=> CQI嵌入模型：冻结模式下可训练参数 ({trainable_count:,})')
            
            mode_descriptions = {
                'all': '冻结全部基础模型',
                'encoder_only': '冻结编码器',
                'freeze_decoder': '冻结解码器'
            }
            logger.info(f'=> 冻结模式: {mode_descriptions.get(freeze_mode, freeze_mode)}')
            
            opt_type = getattr(config, 'optimizer', 'adam')
            if opt_type == 'adamw':
                optimizer = torch.optim.AdamW(trainable_params_list, lr_init, weight_decay=getattr(config, 'weight_decay', 0.0))
                logger.info(f'=> 优化器: AdamW，学习率={lr_init}, weight_decay={getattr(config, "weight_decay", 0.0)}')
            elif opt_type == 'sgd':
                optimizer = torch.optim.SGD(
                    trainable_params_list, lr_init,
                    momentum=getattr(config, 'momentum', 0.9),
                    weight_decay=getattr(config, 'weight_decay', 0.0),
                    nesterov=True
                )
                logger.info(f'=> 优化器: SGD，学习率={lr_init}, momentum={getattr(config, "momentum", 0.9)}, weight_decay={getattr(config, "weight_decay", 0.0)}')
            else:
                optimizer = torch.optim.Adam(trainable_params_list, lr_init)
                logger.info(f'=> 优化器: Adam，学习率={lr_init}')
        else:
            opt_type = getattr(config, 'optimizer', 'adam')
            if opt_type == 'adamw':
                optimizer = torch.optim.AdamW(model.parameters(), lr_init, weight_decay=getattr(config, 'weight_decay', 0.0))
                logger.info(f'=> 优化器: AdamW，学习率={lr_init}, weight_decay={getattr(config, "weight_decay", 0.0)}')
            elif opt_type == 'sgd':
                optimizer = torch.optim.SGD(
                    model.parameters(), lr_init,
                    momentum=getattr(config, 'momentum', 0.9),
                    weight_decay=getattr(config, 'weight_decay', 0.0),
                    nesterov=True
                )
                logger.info(f'=> 优化器: SGD，学习率={lr_init}, momentum={getattr(config, "momentum", 0.9)}, weight_decay={getattr(config, "weight_decay", 0.0)}')
            else:
                optimizer = torch.optim.Adam(model.parameters(), lr_init)
                logger.info(f'=> 优化器: Adam，学习率={lr_init}')
    
    if config.scheduler == 'const':
        scheduler = FakeLR(optimizer=optimizer)
        logger.info('=> 调度器: 恒定学习率')
    elif config.scheduler == 'linear':
        # 预热+线性衰减
        total_steps = config.epochs * len(train_loader)
        warmup_steps = config.warmup_epochs * len(train_loader)
        lr_min = getattr(config, 'lr_eta_min', None) or (lr_init / 100)
        
        scheduler = LinearDecayLR(
            optimizer=optimizer,
            T_max=total_steps,
            T_warmup=warmup_steps,
            eta_min=lr_min,
            lr_max=lr_init
        )
        logger.info(f'=> 调度器: 预热线性衰减 (warmup_epochs={config.warmup_epochs}, eta_min={lr_min})')
    else:
        # 预热+余弦退火 (默认)
        total_steps = config.epochs * len(train_loader)
        warmup_steps = config.warmup_epochs * len(train_loader)
        lr_min = getattr(config, 'lr_eta_min', None) or (lr_init / 100)
        
        scheduler = WarmUpCosineAnnealingLR(
            optimizer=optimizer,
            T_max=total_steps,
            T_warmup=warmup_steps,
            eta_min=lr_min,
            lr_max=lr_init
        )
        logger.info(f'=> 调度器: 预热余弦退火 (warmup_epochs={config.warmup_epochs}, eta_min={lr_min})')
    
    # 创建保存目录 - 按cr和cqi_type分类
    save_root = os.path.join(config.save_path, f'cr_{config.cr}', config.cqi_type)
    config.save_path = save_root
    os.makedirs(config.save_path, exist_ok=True)
    
    logger.info(f'=> 模型将保存到: {config.save_path}')

    # 定义训练管道
    logger.info('=> 设置训练器...')
    # 将梯度裁剪阈值以属性方式挂到 optimizer（最小侵入，不改 Trainer 接口）
    setattr(optimizer, 'grad_clip', getattr(config, 'grad_clip', 0.0))

    # TensorBoard日志目录（支持自定义，避免多个仿真冲突）
    # 默认使用与模型保存目录相关的日志目录
    if not hasattr(config, 'log_dir') or config.log_dir is None:
        log_dir = os.path.join(config.save_path, 'tensorboard')
    else:
        log_dir = config.log_dir
    os.makedirs(log_dir, exist_ok=True)
    logger.info(f'=> TensorBoard日志目录: {log_dir}')
    
    trainer = Trainer(
        model=model,
        device=device,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        cqi_type=config.cqi_type,
        resume=config.resume,
        save_path=config.save_path,
        print_freq=config.print_freq,
        val_freq=config.val_freq,
        test_freq=config.test_freq,
        log_dir=log_dir,
        use_corr_loss=getattr(config, 'use_corr_loss', False),
        lambda_corr_a=getattr(config, 'lambda_corr_a', 0.0),
        lambda_corr_b=getattr(config, 'lambda_corr_b', 0.0)
    )
    
    # 开始训练
    logger.info('=' * 80)
    logger.info(f'开始训练 - {config.epochs} 轮')
    logger.info(f'CQI类型: {config.cqi_type.upper()}')
    logger.info(f'压缩比: 1/{config.cr}')
    logger.info(f'预训练: {"是" if config.use_pretrained else "否"}')
    logger.info('=' * 80)
    
    start_time = time.time()
    
    try:
        trainer.loop(config.epochs, train_loader, val_loader, test_loader)
    except KeyboardInterrupt:
        logger.info('=> 训练被用户中断')
    except Exception as e:
        logger.error(f'=> 训练失败: {e}')
        raise
    
    end_time = time.time()
    training_time = end_time - start_time
    
    # 最终测试
    logger.info('=> 执行最终评估...')
    tester = Tester(model, device, criterion, config.cqi_type)
    loss, rho, nmse = tester(test_loader)
    
    # 打印最终结果
    logger.info('=' * 80)
    logger.info('训练完成')
    logger.info('=' * 80)
    logger.info(f'=> 总训练时间: {training_time/3600:.2f} 小时')
    logger.info(f'=> 最终测试结果:')
    logger.info(f'     Loss: {loss:.3e}')
    logger.info(f'     Rho: {rho:.3e}')
    logger.info(f'     NMSE: {nmse:.3e}')
    if hasattr(trainer, 'best_rho'):
        logger.info(f'=> 训练过程最佳结果:')
        logger.info(f'     Best Rho: {trainer.best_rho.rho:.3e}')
        logger.info(f'     Best NMSE: {trainer.best_nmse.nmse:.3e}')
    if hasattr(trainer, 'best_nmse_1k') and trainer.best_nmse_1k.nmse is not None:
        logger.info(f'     Best NMSE (first 1000 epochs): {trainer.best_nmse_1k.nmse:.3e} at epoch {trainer.best_nmse_1k.epoch}')
    logger.info('=' * 80)
    
    # 保存最终结果
    results_file = os.path.join(config.save_path, f'final_results_{config.cqi_type}_{config.timestamp}.txt')
    with open(results_file, 'w') as f:
        f.write(f'TransCQA Training Results\n')
        f.write(f'========================\n')
        f.write(f'时间戳: {config.timestamp}\n')
        f.write(f'CQI类型: {config.cqi_type}\n')
        f.write(f'压缩比: 1/{config.cr}\n')
        f.write(f'使用预训练: {"是" if config.use_pretrained else "否"}\n')
        if config.use_pretrained:
            f.write(f'预训练模型: {config.pretrained}\n')
        f.write(f'训练轮数: {config.epochs}\n')
        f.write(f'训练时间: {training_time/3600:.2f} 小时\n')
        f.write(f'最终测试Loss: {loss:.3e}\n')
        f.write(f'最终测试Rho: {rho:.3e}\n')
        f.write(f'最终测试NMSE: {nmse:.3e}\n')
        if hasattr(trainer, 'best_rho'):
            f.write(f'最佳Rho: {trainer.best_rho.rho:.3e}\n')
            f.write(f'最佳NMSE: {trainer.best_nmse.nmse:.3e}\n')
        if hasattr(trainer, 'best_nmse_1k') and trainer.best_nmse_1k.nmse is not None:
            f.write(f'前1000轮最佳NMSE: {trainer.best_nmse_1k.nmse:.3e} (Epoch {trainer.best_nmse_1k.epoch})\n')
        f.write(f'模型总参数量: {total_params:}\n')
        f.write(f'模型可训练参数量: {total_trainable_params:}\n')
        
    logger.info(f'=> 结果已保存到: {results_file}')
    
    # 保存训练曲线数据
    try:
        # 保存训练历史到文件供后续绘图使用
        history_file = os.path.join(config.save_path, f'training_history_{config.cqi_type}_{config.timestamp}.txt')
        if hasattr(trainer, 'train_losses') and hasattr(trainer, 'val_losses'):
            with open(history_file, 'w') as f:
                f.write(f'# TransCQA Training History - {config.cqi_type.upper()}\n')
                f.write(f'# Timestamp: {config.timestamp}\n')
                f.write(f'# Epoch,Train_Loss,Val_Loss,Test_Loss,NMSE,Rho\n')
                
                max_len = max(len(trainer.train_losses), len(trainer.val_losses) if hasattr(trainer, 'val_losses') else 0)
                for i in range(max_len):
                    train_loss = trainer.train_losses[i] if i < len(trainer.train_losses) else ''
                    val_loss = trainer.val_losses[i] if hasattr(trainer, 'val_losses') and i < len(trainer.val_losses) else ''
                    test_loss = trainer.test_losses[i] if hasattr(trainer, 'test_losses') and i < len(trainer.test_losses) else ''
                    nmse = trainer.test_nmses[i] if hasattr(trainer, 'test_nmses') and i < len(trainer.test_nmses) else ''
                    rho = trainer.test_rhos[i] if hasattr(trainer, 'test_rhos') and i < len(trainer.test_rhos) else ''
                    f.write(f'{i+1},{train_loss},{val_loss},{test_loss},{nmse},{rho}\n')
            
            logger.info(f'=> 训练历史已保存到: {history_file}')
    except Exception as e:
        logger.warning(f'保存训练历史失败: {e}')
    
    logger.info('=> 🎯 训练完成！')


def quick_test():
    """快速测试函数"""
    logger.info('=> 运行快速测试...')
    
    # 创建测试配置
    from config import Config
    config = Config()
    config.cqi_type = 'wide'
    config.epochs = 2
    config.batch_size = 32
    config.num_workers = 0
    config.d_model = 32
    config.num_encoder_layers = 2
    config.num_decoder_layers = 2
    config.nhead = 4
    
    # 测试模型创建
    device, pin_memory = init_device(seed=42, cpu=True)
    config.device = device
    
    model = init_model(config)
    logger.info('=> 模型创建成功')
    
    # 测试数据形状
    test_input = torch.randn(4, 2, 32, 52)
    test_cqi = torch.randn(4, 1)
    
    model.eval()
    with torch.no_grad():
        if config.cqi_type == 'wide':
            output = model(test_input, wideband_cqi=test_cqi)
        else:
            output = model(test_input)
    
    logger.info(f'=> 前向传播成功: {test_input.shape} -> {output.shape}')
    logger.info('=> 快速测试完成')


if __name__ == "__main__":
    import sys
    
    # if len(sys.argv) > 1 and sys.argv[1] == '--quick-test':
    #     quick_test()
    # else:
    main()