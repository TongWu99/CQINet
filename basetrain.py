"""
训练基础模型（不含CQI嵌入）的脚本

核心思路：
1. 从多个场景目录加载数据
2. 每个场景独立进行8:1:1划分
3. 合并所有场景的训练集、验证集、测试集
4. 使用 TransNet_QPlusD 训练纯基础模型（不包含任何CQI相关模块）
5. 这样可以确保：测试某场景时，该场景的测试数据从未参与过基础模型的训练

使用方法：
    python train_base_model.py --data-dirs "path/to/site1" "path/to/site2" "path/to/site3"
    python train_base_model.py --data-dirs "./losangeles_data" "./new_york_data" --epochs 500 --batch-size 256

基础模型设计：
    - 只包含 Transformer Encoder + Decoder + 压缩/解压层
    - 不包含任何 CQI 相关模块（CQIQueryMLP, CQIAlphaModulator, SR模块等）
    - 参数量小，训练快
    - 后续可无缝加载到 TransNet_H 等完整模型中
"""

import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from config import Config, parse_args
from utils import logger
from utils.solver import Trainer, Tester
from utils import init_device, FakeLR, WarmUpCosineAnnealingLR
from dataloader.losangeles import LosAngelesDataLoader
from tensorboardX import SummaryWriter
import os
import sys
import time
import numpy as np
from collections import defaultdict
from torch.utils.data import ConcatDataset, DataLoader
#from models.PureEncoderDecoder import PureEncoderDecoder, create_pure_encoder_decoder
from models.TransNet_QModPlusD import Transformer_QModPlusD, create_transnet_qmodplusd
from models.crnet import CRNet, crnet
from models.clnet import CLNet, clnet
from models.TransNet import transnet
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch.cuda.device_count():", torch.cuda.device_count())

def combine_dataloaders(data_dirs, batch_size, num_workers, pin_memory, device, cqi_type='wide'):
    """
    从多个场景目录加载数据并合并
    
    关键：每个场景独立进行8:1:1划分，然后合并所有场景的对应划分集
    这样确保：测试某场景时，该场景的测试数据从未参与过训练
    
    Args:
        data_dirs: 场景数据目录列表
        batch_size: 批大小
        num_workers: 工作进程数
        pin_memory: 是否使用固定内存
        device: 计算设备
        cqi_type: CQI类型
        
    Returns:
        train_loader, val_loader, test_loader: 合并后的数据加载器
        site_info: 各场景数据信息
    """
    print("=" * 80)
    print("多场景数据加载与合并")
    print("=" * 80)
    
    # 存储各场景的数据集
    train_datasets = []
    val_datasets = []
    test_datasets = []
    site_info = []
    
    total_train = 0
    total_val = 0
    total_test = 0
    
    # nan模式使用wide数据（nan模式只需要CSI，不需要CQI）
    actual_cqi_type = 'wide' if cqi_type == 'nan' else cqi_type
    
    for i, data_dir in enumerate(data_dirs):
        site_name = os.path.basename(data_dir.rstrip('/'))
        print(f"\n[{i+1}/{len(data_dirs)}] 加载场景: {site_name}")
        print(f"  路径: {data_dir}")
        
        # 检查目录是否存在
        if not os.path.isdir(data_dir):
            print(f"  ⚠️  目录不存在，跳过: {data_dir}")
            continue
        
        # 检查数据文件是否存在（nan模式使用wide数据）
        data_file = os.path.join(data_dir, f'losangeles_adCSI_right_{actual_cqi_type}.mat')
        if not os.path.exists(data_file):
            print(f"  ⚠️  数据文件不存在，跳过: {data_file}")
            continue
        
        # 创建数据加载器（场景内独立划分）
        try:
            loader = LosAngelesDataLoader(
                root=data_dir,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
                cqi_type=actual_cqi_type,  # nan模式使用wide数据
                stratify_by_cqi=True,  # 按CQI分组后再划分
                cqi_bins=None
            )
            
            train_ds, val_ds, test_ds = loader()
            
            train_datasets.append(train_ds)
            val_datasets.append(val_ds)
            test_datasets.append(test_ds)
            
            # 统计信息
            train_size = len(train_ds.dataset)
            val_size = len(val_ds.dataset)
            test_size = len(test_ds.dataset)
            
            total_train += train_size
            total_val += val_size
            total_test += test_size
            
            site_info.append({
                'name': site_name,
                'path': data_dir,
                'train_size': train_size,
                'val_size': val_size,
                'test_size': test_size
            })
            
            print(f"  ✓ 加载成功:")
            print(f"      训练集: {train_size} 样本")
            print(f"      验证集: {val_size} 样本")
            print(f"      测试集: {test_size} 样本")
            
        except Exception as e:
            print(f"  ✗ 加载失败: {e}")
            continue
    
    if not train_datasets:
        raise ValueError("没有成功加载任何场景数据！")
    
    # 打印汇总信息
    print("\n" + "=" * 80)
    print("数据汇总")
    print("=" * 80)
    print(f"成功加载 {len(site_info)} 个场景:")
    for info in site_info:
        print(f"  - {info['name']}: 训练{info['train_size']} / 验证{info['val_size']} / 测试{info['test_size']}")
    print(f"\n总计:")
    print(f"  训练集: {total_train} 样本")
    print(f"  验证集: {total_val} 样本")
    print(f"  测试集: {total_test} 样本")
    print("=" * 80)
    
    return train_datasets, val_datasets, test_datasets, site_info


def create_combined_loader(datasets, batch_size, num_workers, pin_memory, shuffle=True, drop_cqi=False):
    """
    将多个数据集组合成一个数据加载器

    Args:
        datasets: 数据集列表（可以是Dataset或DataLoader）
        batch_size: 批大小
        num_workers: 工作进程数
        pin_memory: 是否使用固定内存
        shuffle: 是否打乱
        drop_cqi: 是否丢弃CQI值（当cqi_type='nan'时需要）

    Returns:
        合并后的DataLoader
    """
    if not datasets:
        raise ValueError("数据集列表为空")

    # 如果是DataLoader列表，提取其数据集
    if len(datasets) > 0 and hasattr(datasets[0], 'dataset'):
        # 这是一个DataLoader列表，获取底层数据集
        underlying_datasets = [dl.dataset for dl in datasets]
    else:
        # 这是一个Dataset列表
        underlying_datasets = datasets

    # 合并数据集
    combined_dataset = ConcatDataset(underlying_datasets)

    # 创建包装数据集（用于丢弃CQI）
    if drop_cqi:
        class DropCQIDataset(torch.utils.data.Dataset):
            def __init__(self, dataset):
                self.dataset = dataset

            def __len__(self):
                return len(self.dataset)

            def __getitem__(self, idx):
                data = self.dataset[idx]
                # data 可能是 (H_ini, H_norm, cqi) 或 (H_ini, H_norm)
                if isinstance(data, tuple) and len(data) == 3:
                    return data[0], data[1]  # 丢弃 CQI
                return data

        combined_dataset = DropCQIDataset(combined_dataset)

    # 创建新的DataLoader
    loader = DataLoader(
        combined_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=shuffle,
        drop_last=True
    )

    return loader


class CombinedDataLoader:
    """
    组合多场景数据加载器
    
    提供统一的接口，返回合并后的train/val/test加载器
    """
    
    def __init__(self, data_dirs, batch_size, num_workers, pin_memory, device, cqi_type='wide'):
        self.data_dirs = data_dirs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.device = device
        self.cqi_type = cqi_type
        
        # 加载并合并数据
        train_loaders, val_loaders, test_loaders, self.site_info = combine_dataloaders(
            data_dirs=data_dirs,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            device=device,
            cqi_type=cqi_type
        )
        
        # 创建合并后的加载器
        self.train_loader = create_combined_loader(
            train_loaders, batch_size, num_workers, pin_memory, shuffle=True, drop_cqi=(cqi_type == 'nan')
        )
        self.val_loader = create_combined_loader(
            val_loaders, batch_size, num_workers, pin_memory, shuffle=False, drop_cqi=(cqi_type == 'nan')
        )
        self.test_loader = create_combined_loader(
            test_loaders, batch_size, num_workers, pin_memory, shuffle=False, drop_cqi=(cqi_type == 'nan')
        )
        
        print(f"\n合并后:")
        print(f"  训练批次数: {len(self.train_loader)}")
        print(f"  验证批次数: {len(self.val_loader)}")
        print(f"  测试批次数: {len(self.test_loader)}")
    
    def __call__(self):
        """返回合并后的数据加载器"""
        return self.train_loader, self.val_loader, self.test_loader
    
    def get_site_info(self):
        """获取各场景信息"""
        return self.site_info


def evaluate_per_site(model, test_loaders, device, criterion, cqi_type='wide'):
    """
    在每个场景的测试集上分别评估模型
    
    Args:
        model: 待评估模型
        test_loaders: 各场景的测试集DataLoader列表
        device: 计算设备
        criterion: 损失函数
        cqi_type: CQI类型
        
    Returns:
        per_site_results: 每个场景的测试结果
    """
    model.eval()
    tester = Tester(model, device, criterion, cqi_type)
    
    per_site_results = {}
    
    for i, test_loader in enumerate(test_loaders):
        site_name = test_loader.dataset.datasets[0].root if hasattr(test_loader.dataset, 'datasets') else f"Site_{i}"
        site_name = os.path.basename(site_name.rstrip('/'))
        
        loss, rho, nmse = tester(test_loader)
        
        per_site_results[site_name] = {
            'loss': loss,
            'rho': rho,
            'nmse': nmse
        }
        
        print(f"  {site_name}: Loss={loss:.3e}, Rho={rho:.3e}, NMSE={nmse:.3e}")
    
    return per_site_results


def main(args):
    """主训练函数"""
    # 解析配置（args 已经是解析好的参数对象）
    
    # 覆盖默认设置以适应基础模型训练
    args.cqi_type = 'nan'  # 基础模型不使用CQI
    args.use_pretrained = False
    args.timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    print("=" * 80)
    print("基础模型训练 (Base Model Training)")
    print("=" * 80)
    print(f"时间戳: {args.timestamp}")
    print(f"CQI类型: {args.cqi_type.upper()} (不使用CQI嵌入)")
    print("=" * 80)
    
    # 检查数据目录参数
    if not hasattr(args, 'data_dirs') or not args.data_dirs or len(args.data_dirs) == 0:
        print("错误: 请使用 --data-dirs 参数指定场景数据目录")
        print("示例: python train_base_model.py --data-dirs './site1' './site2' './site3'")
        sys.exit(1)
    
    data_dirs = args.data_dirs
    print(f"\n加载场景数据: {len(data_dirs)} 个场景")
    for i, d in enumerate(data_dirs):
        print(f"  [{i+1}] {d}")
    
    # 环境初始化
    # device, pin_memory = init_device(
    #     seed=args.seed if hasattr(args, 'seed') else 42,
    #     cpu=(args.device.type == 'cpu'),
    #     gpu=(args.gpu)
    # )
    pin_memory = True
    device=args.device 
    
    # 创建多场景数据加载器
    print("\n=> 创建多场景数据加载器...")
    combined_loader = CombinedDataLoader(
        data_dirs=data_dirs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        device=device,
        cqi_type='nan'  # 基础模型使用nan模式
    )
    
    train_loader, val_loader, test_loader = combined_loader()
    site_info = combined_loader.get_site_info()

    # 定义模型（根据 model_type 选择基础模型）
    print(f"\n=> 创建基础模型 (type: {args.model_type})")
    if args.model_type == 'transnet_qmodplusd':
        model = create_transnet_qmodplusd(
            reduction=args.cr,
            d_model=args.d_model,
            nhead=args.nhead,
            num_encoder_layers=args.num_encoder_layers,
            num_decoder_layers=args.num_decoder_layers,
            dropout=args.dropout,
            mode="nan"
        )
        model_name = "TransNet_QModPlusD"
    elif args.model_type == 'crnet':
        model = crnet(reduction=args.cr)
        model_name = "CRNet"
    elif args.model_type == 'transnet':
        model = transnet(reduction=args.cr,d_model=args.d_model)
        model_name = "TransNet"
    elif args.model_type == 'clnet':
        model = clnet(reduction=args.cr)
        model_name = "CLNet"
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")
    model.to(device)
    print(model)
    print(f"  模型类型: {model_name}")
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  模型参数量: {total_params:,}")
    print(f"  可训练参数量: {total_trainable_params:,}")
    
    # 定义损失函数
    criterion = nn.MSELoss().to(device)
    print("=> 损失函数: MSE Loss")

    # 定义优化器和调度器
    print("=> 设置优化器和调度器...")
    lr_init = args.lr
    
    if args.scheduler == 'const':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr_init)
        scheduler = FakeLR(optimizer=optimizer)
        print(f"=> 优化器: Adam，学习率={lr_init}")
        print("=> 调度器: 恒定学习率")
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr_init)
        total_steps = args.epochs * len(train_loader)
        warmup_steps = args.warmup_epochs * len(train_loader)
        lr_min = args.lr_eta_min if args.lr_eta_min else lr_init / 100
        
        scheduler = WarmUpCosineAnnealingLR(
            optimizer=optimizer,
            T_max=total_steps,
            T_warmup=warmup_steps,
            eta_min=lr_min,
            lr_max=lr_init
        )
        print(f"=> 优化器: Adam，学习率={lr_init}")
        print(f"=> 调度器: 预热余弦退火 (warmup_epochs={args.warmup_epochs}, eta_min={lr_min})")
    
    # 创建保存目录
    # 默认保存路径（根据模型类型自动生成）
    if args.save_path is None:
        default_base = '/data/wutong/Wireless_AI_Research_DateSet/BaseModel'
        args.save_path = os.path.join(default_base, model_name, 'ConstLr')

    save_root = os.path.join(
        args.save_path,
        'base_model',
        f'cr_{args.cr}',
        f'sites_{len(data_dirs)}'

    )
    args.save_path = save_root
    os.makedirs(args.save_path, exist_ok=True)
    
    # 保存场景信息
    site_info_file = os.path.join(save_root, 'site_info.txt')
    with open(site_info_file, 'w', encoding='utf-8') as f:
        f.write(f"基础模型训练 - 场景信息\n")
        f.write(f"模型类型: {model_name}\n")
        f.write("=" * 60 + "\n\n")
        for info in site_info:
            f.write(f"场景: {info['name']}\n")
            f.write(f"  路径: {info['path']}\n")
            f.write(f"  训练集: {info['train_size']} 样本\n")
            f.write(f"  验证集: {info['val_size']} 样本\n")
            f.write(f"  测试集: {info['test_size']} 样本\n\n")
        f.write(f"合并后总计:\n")
        f.write(f"  训练集: {sum(s['train_size'] for s in site_info)} 样本\n")
        f.write(f"  验证集: {sum(s['val_size'] for s in site_info)} 样本\n")
        f.write(f"  测试集: {sum(s['test_size'] for s in site_info)} 样本\n")
    print(f"=> 场景信息已保存到: {site_info_file}")
    
    # TensorBoard日志目录
    log_dir = os.path.join(save_root, 'tensorboard')
    os.makedirs(log_dir, exist_ok=True)
    
    # 设置梯度裁剪
    setattr(optimizer, 'grad_clip', getattr(args, 'grad_clip', 0.0))
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        device=device,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        cqi_type='nan',  # 基础模型不使用CQI
        resume=args.resume,
        save_path=args.save_path,
        print_freq=args.print_freq,
        val_freq=args.val_freq,
        test_freq=args.test_freq,
        log_dir=log_dir,
        use_corr_loss=False
    )
    
    # 开始训练
    print("\n" + "=" * 80)
    print(f"开始训练 - {args.epochs} 轮")
    print(f"训练场景数: {len(data_dirs)}")
    print(f"每轮训练样本数: ~{len(train_loader.dataset)}")
    print("=" * 80)
    
    start_time = time.time()
    
    try:
        trainer.loop(args.epochs, train_loader, val_loader, test_loader)
    except KeyboardInterrupt:
        print("\n训练被用户中断")
    except Exception as e:
        print(f"\n训练失败: {e}")
        raise
    
    end_time = time.time()
    training_time = end_time - start_time
    
    # 最终测试
    print("\n=> 执行最终评估...")
    tester = Tester(model, device, criterion, 'nan')
    loss, rho, nmse = tester(test_loader)
    
    # 打印结果
    print("\n" + "=" * 80)
    print("训练完成")
    print("=" * 80)
    print(f"总训练时间: {training_time/3600:.2f} 小时")
    print(f"\n整体测试结果:")
    print(f"  Loss: {loss:.3e}")
    print(f"  Rho: {rho:.3e}")
    print(f"  NMSE: {nmse:.3e}")
    
    if hasattr(trainer, 'best_rho'):
        print(f"\n训练过程最佳结果:")
        print(f"  Best Rho: {trainer.best_rho.rho:.3e} (Epoch {trainer.best_rho.epoch})")
        print(f"  Best NMSE: {trainer.best_nmse.nmse:.3e} (Epoch {trainer.best_nmse.epoch})")
    
    # 保存最终结果
    results_file = os.path.join(args.save_path, f'base_model_results_{args.timestamp}.txt')
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("基础模型训练结果\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"时间戳: {args.timestamp}\n")
        f.write(f"CQI类型: nan (不使用CQI嵌入)\n")
        f.write(f"场景数: {len(data_dirs)}\n")
        f.write(f"训练轮数: {args.epochs}\n")
        f.write(f"训练时间: {training_time/3600:.2f} 小时\n\n")
        
        f.write("场景信息:\n")
        for info in site_info:
            f.write(f"  - {info['name']}: 训练{info['train_size']} / 验证{info['val_size']} / 测试{info['test_size']}\n")
        
        f.write(f"\n整体测试结果:\n")
        f.write(f"  Loss: {loss:.3e}\n")
        f.write(f"  Rho: {rho:.3e}\n")
        f.write(f"  NMSE: {nmse:.3e}\n\n")
        
        if hasattr(trainer, 'best_rho'):
            f.write(f"最佳结果:\n")
            f.write(f"  Best Rho: {trainer.best_rho.rho:.3e} (Epoch {trainer.best_rho.epoch})\n")
            f.write(f"  Best NMSE: {trainer.best_nmse.nmse:.3e} (Epoch {trainer.best_nmse.epoch})\n")
        
        f.write(f"\n模型参数:\n")
        f.write(f"  总参数量: {total_params:,}\n")
        f.write(f"  可训练参数量: {total_trainable_params:,}\n")
    
    print(f"=> 结果已保存到: {results_file}")
    print("\n=> 基础模型训练完成！")


if __name__ == "__main__":
    # 添加命令行参数支持
    import argparse
    
    parser = argparse.ArgumentParser(description='基础模型训练')
    
    parser.add_argument('--data-dirs', nargs='+', default=['/data/wutong/Wireless_AI_Research_DateSet/00032/DFT_120deg/00032_mapping2_N+90_120deg/Wireless_AI_Research_Dataset',
                                                           '/data/wutong/Wireless_AI_Research_DateSet/00247/DFT_120deg/00247_mapping2_N+180_120deg/Wireless_AI_Research_Dataset',
                                                           '/data/wutong/Wireless_AI_Research_DateSet/00390/DFT_120deg/00390_mapping2_N+90_120deg/Wireless_AI_Research_Dataset',
                                                           '/data/wutong/Wireless_AI_Research_DateSet/00561/DFT_120deg/00561_mapping2_N+90_120deg/Wireless_AI_Research_Dataset',
                                                           '/data/wutong/Wireless_AI_Research_DateSet/00670/DFT_120deg/00670_mapping2_N-45_120deg/Wireless_AI_Research_Dataset',#5
                                                           '/data/wutong/Wireless_AI_Research_DateSet/00743/DFT_120deg/00743_mapping2_N+120_120deg/Wireless_AI_Research_Dataset',
                                                           '/data/wutong/Wireless_AI_Research_DateSet/00812/DFT_120deg/00812_mapping2_N-90_120deg/Wireless_AI_Research_Dataset',
                                                           '/data/wutong/Wireless_AI_Research_DateSet/00873/DFT_120deg/00873_mapping2_N+0_120deg/Wireless_AI_Research_Dataset',
                                                           '/data/wutong/Wireless_AI_Research_DateSet/00873/DFT_120deg/00873_mapping2_N+120_120deg/Wireless_AI_Research_Dataset',
                                                           '/data/wutong/Wireless_AI_Research_DateSet/00873/DFT_120deg/00873_mapping2_N+240_120deg/Wireless_AI_Research_Dataset',#10
                                                           '/data/wutong/Wireless_AI_Research_DateSet/01105/DFT_120deg/01105_mapping2_N-135_120deg/Wireless_AI_Research_Dataset',
                                                           '/data/wutong/Wireless_AI_Research_DateSet/05658/DFT_120deg/05658_mapping2_N-135_120deg/Wireless_AI_Research_Dataset',
                                                           '/data/wutong/Wireless_AI_Research_DateSet/06425/DFT_120deg/06425_mapping2_N+145_120deg/Wireless_AI_Research_Dataset',
                                                           '/data/wutong/Wireless_AI_Research_DateSet/06736/DFT_120deg/06736_mapping2_N-90_120deg/Wireless_AI_Research_Dataset',
                                                           '/data/wutong/Wireless_AI_Research_DateSet/06802/DFT_120deg/06802_mapping2_N+0_120deg/Wireless_AI_Research_Dataset',#15
                                                           '/data/wutong/Wireless_AI_Research_DateSet/06949/DFT_120deg/06949_mapping2_N-25_120deg/Wireless_AI_Research_Dataset',
                                                           '/data/wutong/Wireless_AI_Research_DateSet/07078/DFT_120deg/07078_mapping2_N+60_120deg/Wireless_AI_Research_Dataset',
                                                           '/data/wutong/Wireless_AI_Research_DateSet/07078/DFT_120deg/07078_mapping2_N+300_120deg/Wireless_AI_Research_Dataset',
                                                           '/data/wutong/Wireless_AI_Research_DateSet/07342/DFT_120deg/07342_mapping2_N+135_120deg/Wireless_AI_Research_Dataset',
                                                           '/data/wutong/Wireless_AI_Research_DateSet/07563/DFT_120deg/07563_mapping2_N-15_120deg/Wireless_AI_Research_Dataset',#20
                                                           '/data/wutong/Wireless_AI_Research_DateSet/08532/DFT_120deg/08532_mapping2_N+100_120deg/Wireless_AI_Research_Dataset',
                                                           '/data/wutong/Wireless_AI_Research_DateSet/02220/DFT_120deg/02220_mapping2_N-45_120deg/Wireless_AI_Research_Dataset',
                                                           '/data/wutong/Wireless_AI_Research_DateSet/02220/DFT_120deg/02220_mapping2_N+180_120deg/Wireless_AI_Research_Dataset',
                                                           '/data/wutong/Wireless_AI_Research_DateSet/02311/DFT_120deg/02311_mapping2_N+120_120deg/Wireless_AI_Research_Dataset',
                                                           '/data/wutong/Wireless_AI_Research_DateSet/02311/DFT_120deg/02311_mapping2_N+260_120deg/Wireless_AI_Research_Dataset'],#25
                        help='场景数据目录列表')
    parser.add_argument('--batch-size', type=int, default=1024,
                        help='批大小')
    parser.add_argument('--epochs', type=int, default=10000,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='学习率')
    parser.add_argument('--cr', type=int, default=8,
                        help='压缩比')
    parser.add_argument('--d-model', type=int, default=16,
                        help='Transformer特征维度')
    parser.add_argument('--num-encoder-layers', type=int, default=2,
                        help='编码器层数')
    parser.add_argument('--num-decoder-layers', type=int, default=2,
                        help='解码器层数')
    parser.add_argument('--nhead', type=int, default=4,
                        help='注意力头数')
    parser.add_argument('--gpu', type=int, default=4,
                        help='GPU ID')
    parser.add_argument('--save-path', type=str, default="/data/wutong/Wireless_AI_Research_DateSet/BaseModel/TransNet/ConstLr",
                        help='模型保存路径（默认根据 model-type 自动生成）')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--dim-feedforward', type=int, default=2048, help='前馈网络维度')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout率')
    parser.add_argument('--model-type', type=str, default='transnet', choices=['transnet', 'crnet', 'clnet'],
                        help='基础模型类型: transnet=TransNet_QModPlusD, crnet=CRNet, clnet=CLNet')
    parser.add_argument('--val-freq', type=int, default=10, help='验证频率')
    parser.add_argument('--test-freq', type=int, default=10, help='测试频率')
    parser.add_argument('--print-freq', type=int, default=20, help='打印频率')
    parser.add_argument('--num-workers', type=int, default=4, help='数据加载工作进程数')
    parser.add_argument('--scheduler', type=str, default='const', choices=['const', 'cosine'],
                        help='学习率调度器')
    parser.add_argument('--warmup-epochs', type=int, default=0, help='预热轮数')
    parser.add_argument('--lr-eta-min', type=float, default=1e-5, help='余弦退火最小学习率')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练检查点路径')
    
    args = parser.parse_args()
    if args.gpu is not None:
        args.device = torch.device(f'cuda:{args.gpu}')
    #创建简化的配置对象（兼容后续代码）
    class SimpleArgs:
        def __init__(self, a):
            self.__dict__.update(vars(a))
            # 设置设备
            import torch
            if hasattr(a, 'gpu') and a.gpu is not None:
                self.device = torch.device(f'cuda:{a.gpu}')
            else:
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    args = SimpleArgs(args)
    main(args)

