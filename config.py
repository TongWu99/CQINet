import argparse
import torch
import os
from datetime import datetime

class Config:
    def __init__(self):
        # 数据相关
        self.data_dir = './losangeles_data'  # 数据目录
        self.scenario = 'losangeles'  # 场景名称
        self.cqi_type = 'wide'  # CQI类型: 'wide', 'sub', 'nan'
        self.stratify_by_cqi = True  # 是否按wideband CQI分组后再划分数据集（每组内8:1:1）
        self.cqi_bins = None  # CQI分组方式，None表示按CQI值分组（0-15），或指定分组数量/区间
        
        # 训练相关
        self.batch_size = 200
        self.epochs = 1000
        self.lr = 1e-3
        self.scheduler = 'cosine'  # 'const' or 'cosine'
        
         # CQI 相关正则
        self.use_corr_loss = False
        self.lambda_corr_a = 0.01
        self.lambda_corr_b = 0.01

        # 模型相关（仅保留三种）
        self.d_model = 64
        self.cr = 4  # 压缩比
        self.num_encoder_layers = 4
        self.num_decoder_layers = 4
        self.nhead = 8
        self.dropout = 0.0
        self.model_type = 'crnet'  # 可选: 'crnet', 'clnet', 'transnet_moe', 'transnet_h'
        
        # 硬件相关
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.num_workers = 4
        self.pin_memory = True
        self.seed = 42
        
        # 训练流程
        self.evaluate = False
        self.pretrained = None  # 预训练模型路径
        self.resume = None
        self.save_path = './checkpoints'
        self.print_freq = 10
        self.val_freq = 10
        self.test_freq = 10
        self.snr = 0
        
        # 时间戳相关
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.use_pretrained = False
        
        # 优化器/训练细节
        self.optimizer = 'adam'
        self.weight_decay = 0.0
        self.momentum = 0.9  # SGD动量系数
        self.grad_clip = 0.0
        self.warmup_epochs = 0  # 预热轮数（默认0）
        self.lr_eta_min = None  # 余弦退火最小学习率（None表示自动设为lr/100）
        
        # TransNet_QMod 相关
        self.cqi_vocab_size = 16

        # TransNet_MoE 相关
        self.moe_experts = 8
        self.moe_top_k = None
        self.router_lr = None
        self.moe_enable_channel = False

        # 冻结模式相关
        self.freeze_base = False  # 保留兼容性
        self.freeze_mode = 'none'  # 'none', 'all', 'encoder_only', 'freeze_decoder'

        

def parse_args():
    parser = argparse.ArgumentParser(description='TransCQA/TransNet PyTorch Training')
    
    # 必需参数
    parser.add_argument('--data-dir', type=str, default='/data/wutong/Wireless_AI_Research_DateSet/06949/DFT_120deg/06949_mapping2_N-25_120deg/Wireless_AI_Research_Dataset',
                        help='数据目录路径')
    parser.add_argument('--scenario', type=str, default='losangeles',
                        help='场景名称')
    parser.add_argument('--cqi-type', type=str, default='wide', 
                        choices=['wide', 'sub', 'nan'],
                        help='CQI类型')
    parser.add_argument('--stratify-by-cqi', default=True,
                        help='按wideband CQI分组后再划分数据集（每组内8:1:1）')
    parser.add_argument('--cqi-bins', type=int, default=None,
                        help='CQI分组数量（None表示按CQI值0-15分组，整数表示等频分组数量）')
    
    # 训练参数
    parser.add_argument('-b', '--batch-size', type=int, default=256,
                        help='批大小')
    parser.add_argument('--epochs', type=int, default=500,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='学习率')
    parser.add_argument('--snr', type=float, default=0,
                        help='snr')
    parser.add_argument('--scheduler', type=str, default='linear',
                        choices=['const', 'cosine', 'linear'],
                        help='学习率调度器: const=恒定, cosine=余弦退火, linear=线性衰减')
    
    # 模型参数（仅保留三种模型通用超参）
    parser.add_argument('--d-model', type=int, default=16,
                        help='Transformer特征维度')
    parser.add_argument('--cr', type=int, default=8,
                        help='压缩比')
    parser.add_argument('--num-encoder-layers', type=int, default=2,
                        help='编码器层数')
    parser.add_argument('--num-decoder-layers', type=int, default=2,
                        help='解码器层数')
    parser.add_argument('--nhead', type=int, default=3,
                        help='注意力头数')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout率')
    
    # 硬件参数
    parser.add_argument('--gpu', type=int, default=7,
                        help='GPU ID')
    parser.add_argument('--cpu', action='store_true',
                        help='使用CPU训练')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='数据加载工作进程数')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    # 训练流程
    parser.add_argument('-e', '--evaluate', action='store_true',default=False,
                        help='仅评估模型')
    parser.add_argument('--use_pretrained',  type= bool,default=True,#"/data/wutong/Wireless_AI_Research_DateSet/BaseModel/TransNet_QModPlusD/ConstLr/base_model/cr_16/sites_25/best_nmse_nan.pth"
                        help='是否使用预训练模型')#/data/wutong/Wireless_AI_Research_DateSet/BaseModel/CRNet/ConstLr/base_model/cr_64/sites_25/best_nmse_nan.pth
    parser.add_argument('--pretrained', type=str, default="/data/wutong/Wireless_AI_Research_DateSet/BaseModel/CRNet/ConstLr/base_model/cr_8/sites_25/best_nmse_nan.pth",
                        help='预训练模型路径')#/data/wutong/Wireless_AI_Research_DateSet/00873/DFT_120deg/00873_mapping2_N+0_120deg/Result_2_32_52_d16/CRNet/BaseModel_Finetuned_woCQI/cr_64/nan/best_nmse_nan.pth
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复训练检查点路径')
    parser.add_argument('--save-path', type=str, default='/data/wutong/Wireless_AI_Research_DateSet/06949/DFT_120deg/06949_mapping2_N-25_120deg/CRNet/Result_2_32_52_d16/BaseModel_CQA_EFS_Patch16t2',
                        help='模型保存路径')
    
    # 优化器/日志
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'adamw', 'sgd'],
                        help='优化器类型（默认adam）')
    parser.add_argument('--weight-decay', type=float, default=0,
                        help='权重衰减系数（AdamW/SGD有效；默认0不启用）')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='动量系数（SGD有效；默认0.9）')
    parser.add_argument('--grad-clip', type=float, default=0.0,
                        help='梯度裁剪阈值（L2范数；<=0 不启用）')
    parser.add_argument('--log-dir', type=str, default=None,
                        help='TensorBoard日志目录（默认：logs/exp_timestamp）')
    parser.add_argument('--warmup-epochs', type=int, default=10,
                        help='预热轮数（默认0，设置为正数启用预热）')
    
    # 余弦退火参数
    parser.add_argument('--lr-eta-min', type=float, default=1e-5,
                        help='余弦退火的最小学习率（None表示自动设为lr/10）')
    
    # 模型类型（仅保留四种）
    parser.add_argument('--model-type', type=str, default='crnet_cqi',
                        choices=['crnet', 'clnet', 'transnet_moe', 'transnet_qmod', 'transnet_h', 
                                'crnet_cqi', 'crnet_cqi_film', 'clnet_cqi', 'transnet_qmodplusd_cqi'],
                        help='模型类型：crnet/clnet/transnet系列，或带CQI嵌入的微调模型(crnet_cqi/crnet_cqi_film/clnet_cqi/transnet_qmodplusd_cqi)')
    # CQI 相关正则
    parser.add_argument('--use-corr-loss', default=False,
                        help='启用CQI与A/B负相关的正则项')
    parser.add_argument('--lambda-corr-a', type=float, default=0.1,
                        help='L_corr_a 权重（默认0关闭）')
    parser.add_argument('--lambda-corr-b', type=float, default=0.1,
                        help='L_corr_b 权重（默认0关闭）')
    # TransNet_QMod 相关
    parser.add_argument('--cqi-vocab-size', type=int, default=16, help='CQI词汇表大小（0-15）')
    
    # TransNet_MoE 相关
    parser.add_argument('--moe-experts', type=int, default=8, help='TransNet MoE专家数量')
    parser.add_argument('--moe-top-k', type=int, default=None, 
                        help='TransNet MoE Top-K稀疏激活（None表示激活所有专家，2表示只激活2个）')
    parser.add_argument('--router-lr', type=float, default=1e-4,
                        help='Router独立学习率（None表示自动设为10×基准lr）')
    parser.add_argument('--moe-enable-channel', action='store_true', 
                        help='TransNet MoE开启信道仿真（默认禁用）')
    parser.add_argument('--moe-disable-channel', action='store_true', default=True,
                        help='TransNet MoE禁用信道仿真（默认禁用）')


    
    # 时间戳参数
    parser.add_argument('--timestamp', type=str, default=None,
                        help='自定义时间戳')
    # CQI 嵌入模块参数（用于 _cqi 模型）
    parser.add_argument('--sr-hidden-channels', type=int, default=32,
                        help='超分辨率模块的隐藏通道数')
    parser.add_argument('--film-hidden-dim', type=int, default=32,
                        help='FiLM调制器的隐藏层维度（仅用于crnet_cqi_film模型）')
    parser.add_argument('--freeze-base', action='store_true', default=False,
                        help='冻结基础模型参数，只训练 CQI 嵌入模块（仅用于 _cqi 模型）')
    parser.add_argument('--freeze-mode', type=str, default='encoder_only',
                        choices=['none', 'all', 'encoder_only', 'none'],
                        help='冻结模式：none=不冻结, all=冻结全部基础模型, encoder_only=冻结编码器, freeze_decoder=冻结解码器（仅用于 _cqi 模型）')
    args = parser.parse_args()
    
    # 创建配置对象并更新
    config = Config()
    for key, value in vars(args).items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    # 设置设备
    if args.gpu is not None:
        config.device = torch.device(f'cuda:{args.gpu}')
    elif args.cpu:
        config.device = torch.device('cpu')
    
    # 设置时间戳
    if args.timestamp is not None:
        config.timestamp = args.timestamp
    else:
        config.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 处理TransNet MoE信道仿真参数（默认禁用）
    config.moe_enable_channel = False
    if hasattr(config, 'moe_enable_channel') and config.moe_enable_channel:
        config.moe_enable_channel = True
    elif hasattr(config, 'moe_disable_channel') and not config.moe_disable_channel:
        config.moe_enable_channel = True

    config.use_corr_loss = args.use_corr_loss
    config.lambda_corr_a = args.lambda_corr_a
    config.lambda_corr_b = args.lambda_corr_b

    # TransNet_QC 参数
    config.sr_hidden_channels = args.sr_hidden_channels if hasattr(args, 'sr_hidden_channels') else 16
    config.film_hidden_dim = args.film_hidden_dim if hasattr(args, 'film_hidden_dim') else 32
    config.freeze_base = args.freeze_base if hasattr(args, 'freeze_base') else False
    config.freeze_mode = args.freeze_mode if hasattr(args, 'freeze_mode') else 'none'
    # 预训练使用规则
    # if config.pretrained is not None and os.path.exists(config.pretrained):
    #     config.use_pretrained = config.cqi_type in ['wide', 'sub']
    #     if config.cqi_type == 'nan':
    #         config.pretrained = None
    
    return config

if __name__ == "__main__":
    config = parse_args()
    print("配置参数:")
    for key, value in vars(config).items():
        print(f"  {key}: {value}")
