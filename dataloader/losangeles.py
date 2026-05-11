import os
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import sys
__all__ = ['LosAngelesDataLoader', 'CSI_CQIDataset', 'PreFetcher']


class PreFetcher:
    """数据预取器，加速数据加载"""
    
    def __init__(self, loader, device):
        self.ori_loader = loader
        self.len = len(loader)
        self.device = device
        
        # 只有在使用CUDA时才创建CUDA stream
        if device.type == 'cuda':
            self.stream = torch.cuda.Stream()
            self.use_cuda = True
        else:
            self.stream = None
            self.use_cuda = False
            
        self.next_input = None

    def preload(self):
        try:
            self.next_input = next(self.loader)
        except StopIteration:
            self.next_input = None
            return

        if self.use_cuda:
            with torch.cuda.stream(self.stream):
                for idx, tensor in enumerate(self.next_input):
                    self.next_input[idx] = tensor.to(self.device, non_blocking=True)
        else:
            # CPU情况下直接移动到设备
            for idx, tensor in enumerate(self.next_input):
                self.next_input[idx] = tensor.to(self.device)

    def __len__(self):
        return self.len

    def __iter__(self):
        self.loader = iter(self.ori_loader)
        self.preload()
        return self

    def __next__(self):
        if self.use_cuda:
            torch.cuda.current_stream().wait_stream(self.stream)
        
        input = self.next_input
        if input is None:
            raise StopIteration
            
        if self.use_cuda:
            for tensor in input:
                tensor.record_stream(torch.cuda.current_stream())
                
        self.preload()
        return input


class CSI_CQIDataset(Dataset):
    def __init__(self, H_data, s_data, cqi_type, device='cpu', sample_axis=0):
        """
        H_data: shape (N, 2, H, W)
        s_data: shape (N,) for wideband CQI or (N, S) for subband CQI
        """
        self.device = device
        self.sample_axis = sample_axis
        self.num_samples = H_data.shape[self.sample_axis]

        # 保存原始 CSI
        self.H_initial = H_data

        # CSI normalization: 按样本维以外的维度做 Z-score
        axes = tuple(i for i in range(H_data.ndim) if i != self.sample_axis)
        self.H_data = (H_data - np.mean(H_data, axis=axes, keepdims=True)) / \
                      (np.std(H_data, axis=axes, keepdims=True) + 1e-8)

        self.cqi_type = cqi_type

        # 保存原始 CQI
        self.s_original = s_data

        # 执行 CQI 归一化（Z-score）
        if self.cqi_type == 'wide':
            self.s_mean = np.mean(s_data)
            self.s_std = np.std(s_data)
        else:  # subband
            self.s_mean = np.mean(s_data, axis=0, keepdims=True)  # shape (1, S)
            self.s_std = np.std(s_data, axis=0, keepdims=True)

        self.s_data = (s_data - self.s_mean) / (self.s_std + 1e-8)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        H_ini = torch.tensor(np.take(self.H_initial, idx, axis=self.sample_axis), dtype=torch.float32)
        H_norm = torch.tensor(np.take(self.H_data, idx, axis=self.sample_axis), dtype=torch.float32)
        s = torch.tensor(self.s_data[idx], dtype=torch.float32)

        if self.cqi_type != 'nan':
            return H_ini, H_norm, s
        else:
            return H_ini, H_norm

    def get_cqi_stats(self):
        """返回归一化参数（可用于反归一化或部署）"""
        return self.cqi_type, self.s_mean, self.s_std
    
    def get_data_info(self):
        """获取数据集信息"""
        return {
            'num_samples': self.num_samples,
            'h_shape': tuple(self.H_data.shape),
            'cqi_shape': tuple(self.s_data.shape),
            'cqi_type': self.cqi_type,
            'h_mean': self.H_data.mean().item(),
            'h_std': self.H_data.std().item(),
            'cqi_mean': self.s_data.mean().item(),
            'cqi_std': self.s_data.std().item(),
        }


class LosAngelesDataLoader(object):
    """Los Angeles数据集的PyTorch数据加载器"""
    
    def __init__(self, root, batch_size, num_workers, pin_memory, cqi_type='wide', device=None, 
                 stratify_by_cqi=False, cqi_bins=None):
        """
        Args:
            root: 数据根目录
            batch_size: 批大小
            num_workers: 工作进程数
            pin_memory: 是否使用固定内存
            cqi_type: CQI类型 ('wide', 'sub', 'nan')
            device: 目标设备
            stratify_by_cqi: 是否按wideband CQI分组后再划分数据集（每组内8:1:1）
            cqi_bins: CQI分组方式，None表示按CQI值分组（0-15），或指定分组数量/区间
        """
        assert os.path.isdir(root), f"数据目录不存在: {root}"
        assert cqi_type in {"wide", "sub", "nan"}, f"不支持的CQI类型: {cqi_type}"
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.cqi_type = cqi_type
        self.device = device if device is not None else torch.device('cpu')
        self.stratify_by_cqi = stratify_by_cqi
        self.cqi_bins = cqi_bins
        
        # 构建数据文件路径
        if cqi_type != 'nan':
            file_path = os.path.join(root, f'losangeles_adCSI_right_{cqi_type}.mat')
        else:
            file_path = os.path.join(root, f'losangeles_adCSI_right_wide.mat')
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"数据文件不存在: {file_path}")
        
        print(f"正在加载数据: {file_path}")
        
        # 加载数据
        H_data, CQI_data, h_sample_axis = self._load_h5_data(file_path, cqi_type)
        # print("查看CQI内容")
        # print(CQI_data)
        print(f"加载的数据形状:")
        print(f"  H矩阵: {H_data.shape}")
        print(f"  CQI: {CQI_data.shape}")
        
        # 如果需要按CQI分组，加载wideband CQI用于分组
        wideband_cqi_for_stratify = None
        if stratify_by_cqi:
            wide_file_path = os.path.join(root, 'losangeles_adCSI_right_wide.mat')
            if os.path.exists(wide_file_path):
                _, wideband_cqi_for_stratify, _ = self._load_h5_data(wide_file_path, 'wide')
                # 确保wideband CQI是一维的
                if len(wideband_cqi_for_stratify.shape) > 1:
                    wideband_cqi_for_stratify = wideband_cqi_for_stratify.flatten()
                print(f"  已加载wideband CQI用于分组: {wideband_cqi_for_stratify.shape}")
            else:
                print(f"  警告: 未找到wideband CQI文件，将使用普通划分方式")
                stratify_by_cqi = False
        
        # 数据分割
        data_splits = self._prepare_data_splits(H_data, CQI_data, 
                                                stratify_by_cqi=stratify_by_cqi,
                                                wideband_cqi=wideband_cqi_for_stratify,
                                                sample_axis=h_sample_axis)
        
        # 创建数据集
        self.train_dataset = CSI_CQIDataset(
            data_splits['train']['H'], 
            data_splits['train']['CQI'], 
            cqi_type=cqi_type,
            sample_axis=h_sample_axis
        )
        
        self.val_dataset = CSI_CQIDataset(
            data_splits['val']['H'], 
            data_splits['val']['CQI'], 
            cqi_type=cqi_type,
            sample_axis=h_sample_axis
        )
        
        self.test_dataset = CSI_CQIDataset(
            data_splits['test']['H'], 
            data_splits['test']['CQI'], 
            cqi_type=cqi_type,
            sample_axis=h_sample_axis
        )
        
        # 打印数据集信息
        print("\n数据集信息:")
        for name, dataset in [('训练', self.train_dataset), 
                             ('验证', self.val_dataset), 
                             ('测试', self.test_dataset)]:
            info = dataset.get_data_info()
            print(f"  {name}集: {info['num_samples']} 样本")
        
        # 输出各划分内CQI分布占比
        self._print_split_cqi_summary(data_splits)
    
    def _load_h5_data(self, file_path, cqi_type):
        """从H5文件加载数据"""
        try:
            with h5py.File(file_path, 'r') as f:
                # 加载信道矩阵节点
                h_node = f['H_final_angle_delay']
                if isinstance(h_node, h5py.Dataset):
                    h_struct = h_node[:]
                    real_part_final = h_struct['real'].astype(np.float32)
                    imag_part_final = h_struct['imag'].astype(np.float32)
                else:
                    # Group形式，包含real/imag子节点
                    real_part_final = h_node['real'][:].astype(np.float32)
                    imag_part_final = h_node['imag'][:].astype(np.float32)
                
                # 根据类型加载CQI
                if cqi_type == 'nan':
                    CQI_results = f['wideband_CQI_results'][:]
                else:
                    CQI_results = f[f'{cqi_type}band_CQI_results'][:]
                
                H_final_combined = np.stack((real_part_final, imag_part_final), axis=1)

                sample_count = self._infer_sample_count(CQI_results)
                sample_axis = self._infer_h_sample_axis(H_final_combined.shape, sample_count)
                H_final_combined = self._standardize_h_shape(H_final_combined, sample_axis)
                sample_axis = 0
                
        except Exception as e:
            raise RuntimeError(f"加载数据时出错 {file_path}: {str(e)}")
        
        return H_final_combined, CQI_results, sample_axis
    
    def _prepare_data_splits(self, H_data, CQI_data, test_size=0.2, val_size=0.5, random_state=42,
                             stratify_by_cqi=False, wideband_cqi=None, sample_axis=0):
        """
        准备数据分割
        
        Args:
            H_data: H矩阵数据
            CQI_data: CQI数据
            test_size: 测试集比例（相对于总数据）
            val_size: 验证集在(验证+测试)中的比例
            random_state: 随机种子
            stratify_by_cqi: 是否按wideband CQI分组
            wideband_cqi: 用于分组的wideband CQI数据（当stratify_by_cqi=True时需要）
        """
        CQI_data = self._reshape_cqi_data(CQI_data)
        num_samples = CQI_data.shape[0]

        if H_data.shape[sample_axis] != num_samples:
            raise ValueError(f"H数据样本维度({H_data.shape[sample_axis]})与CQI样本数({num_samples})不一致")
        
        indices = np.arange(num_samples)
        
        if not stratify_by_cqi or wideband_cqi is None:
            train_idx, temp_idx, CQI_train, CQI_temp = train_test_split(
                indices, CQI_data, test_size=test_size, random_state=random_state
            )
            
            val_idx, test_idx, CQI_val, CQI_test = train_test_split(
                temp_idx, CQI_temp, test_size=val_size, random_state=random_state
            )
        else:
            # 按wideband CQI分组后，每组内按8:1:1划分
            print(f"\n按wideband CQI分组划分数据集...")
            
            if len(wideband_cqi.shape) > 1:
                wideband_cqi = wideband_cqi.flatten()
            if wideband_cqi.shape[0] != num_samples:
                raise ValueError(f"用于分组的wideband CQI数量({wideband_cqi.shape[0]})与样本数({num_samples})不一致")
            
            # 确定CQI分组方式
            if self.cqi_bins is None:
                # 按CQI值分组（0-15，共16组）
                # 将CQI值四舍五入到最近的整数
                cqi_labels = np.round(wideband_cqi).astype(int)
                # 限制在0-15范围内
                cqi_labels = np.clip(cqi_labels, 0, 15)
                unique_cqi_values = np.unique(cqi_labels)
                print(f"  按CQI值分组: {len(unique_cqi_values)} 个CQI类别")
            elif isinstance(self.cqi_bins, int):
                # 按指定数量分组（等频分组）
                # 使用numpy的percentile进行等频分组
                sorted_indices = np.argsort(wideband_cqi)
                sorted_cqi = wideband_cqi[sorted_indices]
                # 计算每个区间的分位数
                percentiles = np.linspace(0, 100, self.cqi_bins + 1)
                bin_edges = np.percentile(sorted_cqi, percentiles)
                # 使用digitize进行分组
                cqi_labels = np.digitize(wideband_cqi, bin_edges[1:])  # 排除最小值
                cqi_labels = np.clip(cqi_labels, 0, self.cqi_bins - 1)  # 确保标签在0到bins-1之间
                unique_cqi_values = np.unique(cqi_labels)
                print(f"  按等频分组: {self.cqi_bins} 个区间")
            else:
                # 自定义区间
                cqi_labels = np.digitize(wideband_cqi, self.cqi_bins) - 1
                unique_cqi_values = np.unique(cqi_labels)
                print(f"  按自定义区间分组: {len(unique_cqi_values)} 个区间")
            
            cqi_counts = {}
            for cqi_val in unique_cqi_values:
                count = np.sum(cqi_labels == cqi_val)
                cqi_counts[cqi_val] = count
                print(f"    CQI类别 {cqi_val}: {count} 个样本")
            
            train_indices = []
            val_indices = []
            test_indices = []
            CQI_train_list = []
            CQI_val_list = []
            CQI_test_list = []
            
            for cqi_val in unique_cqi_values:
                # 获取该CQI类别对应的索引
                mask = (cqi_labels == cqi_val)
                indices = np.where(mask)[0]
                
                if len(indices) < 3:
                    print(f"    警告: CQI类别 {cqi_val} 样本数过少({len(indices)})，全部放入训练集")
                    train_indices.append(indices)
                    CQI_train_list.append(CQI_data[indices])
                    continue

                train_cqi_idx, temp_cqi_idx, CQI_train_cqi, CQI_temp_cqi = train_test_split(
                    indices, CQI_data[indices], test_size=test_size, random_state=random_state
                )
                
                val_cqi_idx, test_cqi_idx, CQI_val_cqi, CQI_test_cqi = train_test_split(
                    temp_cqi_idx, CQI_temp_cqi, test_size=val_size, random_state=random_state
                )
                
                train_indices.append(train_cqi_idx)
                val_indices.append(val_cqi_idx)
                test_indices.append(test_cqi_idx)
                CQI_train_list.append(CQI_train_cqi)
                CQI_val_list.append(CQI_val_cqi)
                CQI_test_list.append(CQI_test_cqi)
            
            train_idx = np.concatenate(train_indices) if train_indices else np.array([], dtype=int)
            val_idx = np.concatenate(val_indices) if val_indices else np.array([], dtype=int)
            test_idx = np.concatenate(test_indices) if test_indices else np.array([], dtype=int)
            CQI_train = np.concatenate(CQI_train_list, axis=0) if CQI_train_list else np.array([])
            CQI_val = np.concatenate(CQI_val_list, axis=0) if CQI_val_list else np.array([])
            CQI_test = np.concatenate(CQI_test_list, axis=0) if CQI_test_list else np.array([])
            
            print(f"\n分组划分完成:")
            print(f"  训练集: {len(train_idx)} 样本 ({len(train_idx)/num_samples*100:.1f}%)")
            print(f"  验证集: {len(val_idx)} 样本 ({len(val_idx)/num_samples*100:.1f}%)")
            print(f"  测试集: {len(test_idx)} 样本 ({len(test_idx)/num_samples*100:.1f}%)")
        
        H_train = np.take(H_data, train_idx, axis=sample_axis)
        H_val = np.take(H_data, val_idx, axis=sample_axis)
        H_test = np.take(H_data, test_idx, axis=sample_axis)
        
        splits = {
            'train': {'H': H_train, 'CQI': CQI_train},
            'val': {'H': H_val, 'CQI': CQI_val},
            'test': {'H': H_test, 'CQI': CQI_test}
        }
        
        return splits

    def _reshape_cqi_data(self, cqi_data):
        """将CQI数据转换为 (N, dim)，并在sub模式下保留子带的空间顺序
        兼容常见形状：
        - (N,)                -> (N, 1)
        - (N, S)              -> (N, S)
        - (S, N)              -> (N, S)  当 S in {8, 13}
        - (1, N)              -> (N, 1)
        - (1, N, S)           -> (N, S)
        - (N, S, 1)           -> (N, S)
        """
        arr = cqi_data
        # 1D: 视为wide单值
        if arr.ndim == 1:
            return arr.reshape(-1, 1)
        # (1, N) -> (N, 1)
        if arr.ndim == 2 and arr.shape[0] == 1:
            return arr.reshape(arr.shape[1], 1)
        # (1, N, S) -> (N, S)
        if arr.ndim == 3 and arr.shape[0] == 1:
            return arr.reshape(arr.shape[1], arr.shape[2])
        # (N, S, 1) -> (N, S)
        if arr.ndim == 3 and arr.shape[-1] == 1:
            return arr.reshape(arr.shape[0], arr.shape[1])
        # (S, N) 且 S 为子带数(8或13)时，转置为 (N, S)
        if arr.ndim == 2 and arr.shape[0] in (8, 13) and arr.shape[1] != 1:
            return arr.T
        # (N, S) 已经是期望形状
        return arr

    def _infer_sample_count(self, cqi_data):
        """根据CQI形状推断样本数量（健壮处理sub/wide形状）"""
        arr = cqi_data
        if arr.ndim == 1:
            # wide: N
            return arr.shape[0]
        # (1, N) wide 情形
        if arr.ndim == 2 and arr.shape[0] == 1:
            return arr.shape[1]
        # (S, N) with S in {8,13}
        if arr.ndim == 2 and arr.shape[0] in (8, 13) and arr.shape[1] != 1:
            return arr.shape[1]
        # (1, N, S)
        if arr.ndim == 3 and arr.shape[0] == 1:
            return arr.shape[1]
        # (N, S, 1)
        if arr.ndim == 3 and arr.shape[-1] == 1:
            return arr.shape[0]
        # 默认：第一维视为样本维
        return arr.shape[0]

    def _infer_h_sample_axis(self, h_shape, sample_count):
        """推断H矩阵中的样本维位置（带稳健回退）"""
        # 首先按精确匹配
        candidates = [idx for idx, dim in enumerate(h_shape) if dim == sample_count]
        if candidates:
            # 优先选择最后匹配的轴（通常是样本维）
            return candidates[-1]
        # 回退策略：选择最大的维度作为样本维（典型为样本数，如10000）
        # 这样可以避免因CQI维度推断失误(如得到1)导致的报错
        max_axis = int(np.argmax(list(h_shape)))
        print(f"警告: 未找到与样本数({sample_count})匹配的维度，在H形状{h_shape}中回退选择最大维axis={max_axis}作为样本维")
        return max_axis

    def _standardize_h_shape(self, h_data, sample_axis):
        """将H矩阵调整为 (N, 2, H, W)，便于模型训练"""
        if h_data.ndim != 4:
            raise ValueError(f"H矩阵应为4维，但得到形状: {h_data.shape}")

        if sample_axis != 0:
            h_data = np.moveaxis(h_data, sample_axis, 0)

        # 查找通道维(=2)并移动到axis=1
        if h_data.shape[1] != 2:
            channel_axis = None
            for idx in range(1, h_data.ndim):
                if h_data.shape[idx] == 2:
                    channel_axis = idx
                    break
            if channel_axis is None:
                raise ValueError(f"无法在H矩阵形状 {h_data.shape} 中找到通道维(=2)")
            h_data = np.moveaxis(h_data, channel_axis, 1)
        
        # 确保空间维度顺序为 (32, 52)
        if h_data.shape[2:] == (52, 32):
            h_data = h_data.swapaxes(2, 3)

        return h_data

    def _print_split_cqi_summary(self, data_splits):
        print("\nCQI分布（各划分占比，按四舍五入到0-15的标签统计）:")
        for split_name in ['train', 'val', 'test']:
            cqi_arr = data_splits[split_name]['CQI']
            if cqi_arr is None or (isinstance(cqi_arr, np.ndarray) and cqi_arr.size == 0):
                print(f"  {split_name} 集: 无CQI数据")
                continue

            cqi_2d = self._reshape_cqi_data(cqi_arr)
            N = cqi_2d.shape[0]
            if N == 0:
                print(f"  {split_name} 集: 无样本")
                continue

            if cqi_2d.shape[1] > 1:
                # 子带CQI：按样本对各子带取平均再统计
                cqi_scalar = np.mean(cqi_2d, axis=1)
                reduce_note = "（子带CQI已对每样本取平均）"
            else:
                cqi_scalar = cqi_2d[:, 0]
                reduce_note = ""

            labels = np.round(cqi_scalar).astype(int)
            labels = np.clip(labels, 0, 15)
            uniq, cnts = np.unique(labels, return_counts=True)
            perc = cnts / N * 100.0
            # 组合成可读字符串
            parts = [f"{int(u)}: {p:.1f}% ({int(c)})" for u, p, c in zip(uniq, perc, cnts)]
            summary_line = " | ".join(parts)
            print(f"  {split_name} 集{reduce_note}: N={N} -> {summary_line}")
    
    def __call__(self):
        """返回数据加载器"""
        # 训练数据加载器
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            drop_last=True
        )
        
        # 验证数据加载器
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=False
        )
        
        # 测试数据加载器
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=False
        )
        
        # 如果使用GPU并且pin_memory为True，添加预取器
        if self.pin_memory and self.device.type == 'cuda':
            train_loader = PreFetcher(train_loader, self.device)
            val_loader = PreFetcher(val_loader, self.device)
            test_loader = PreFetcher(test_loader, self.device)
        
        return train_loader, val_loader, test_loader


def test_data_loader():
    """测试数据加载器"""
    print("🔍 测试Los Angeles数据加载器")
    print("="*50)
    
    # 模拟配置
    class Config:
        data_dir = './losangeles_data'
        batch_size = 32
        num_workers = 0  # 测试时使用0避免多进程问题
        pin_memory = False
    
    config = Config()
    
    # 测试三种CQI类型
    for cqi_type in ['wide', 'sub', 'nan']:
        print(f"\n📡 测试 {cqi_type.upper()} CQI数据加载:")
        print("-" * 30)
        
        try:
            # 创建数据加载器
            data_loader = LosAngelesDataLoader(
                root=config.data_dir,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
                pin_memory=config.pin_memory,
                cqi_type=cqi_type
            )
            
            train_loader, val_loader, test_loader = data_loader()
            
            # 测试加载一个批次
            for batch_data in train_loader:
                if cqi_type == 'nan':
                    h_input, h_target = batch_data
                    print(f"✅ 数据加载成功")
                    print(f"输入形状: {h_input.shape}")
                    print(f"目标形状: {h_target.shape}")
                    print(f"数据类型: {h_input.dtype}")
                    print(f"数据范围: [{h_input.min():.3f}, {h_input.max():.3f}]")
                else:
                    h_target, h_input, cqi = batch_data
                    print(f"✅ 数据加载成功")
                    print(f"输入形状: {h_input.shape}")
                    print(f"目标形状: {h_target.shape}")
                    print(f"CQI形状: {cqi.shape}")
                    print(f"数据类型: {h_input.dtype}, {cqi.dtype}")
                    print(f"H数据范围: [{h_input.min():.3f}, {h_input.max():.3f}]")
                    print(f"CQI数据范围: [{cqi.min():.3f}, {cqi.max():.3f}]")
                break
            
        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
    
    print(f"\n{'='*50}")
    print("🎯 数据加载器测试完成!")


if __name__ == "__main__":
    test_data_loader()