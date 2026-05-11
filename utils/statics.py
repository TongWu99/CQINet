import torch
import numpy as np

__all__ = ['AverageMeter', 'evaluator', 'evaluator_cqi']


class AverageMeter(object):
    """计算和存储平均值和当前值
    从 https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262 导入
    """
    def __init__(self, name):
        self.reset()
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.name = name

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return f"==> For {self.name}: sum={self.sum}; avg={self.avg}"


def evaluator(sparse_pred, sparse_gt, raw_gt):
    """原始TransNet评估器（与COST2100数据集兼容）
    计算规范化均方误差(NMSE)和rho
    """
    with torch.no_grad():
        # 基本参数
        nt = 32
        nc = 32
        nc_expand = 257

        # 去中心化
        sparse_gt = sparse_gt - 0.5
        sparse_pred = sparse_pred - 0.5

        # 计算NMSE
        power_gt = sparse_gt[:, 0, :, :] ** 2 + sparse_gt[:, 1, :, :] ** 2
        difference = sparse_gt - sparse_pred
        mse = difference[:, 0, :, :] ** 2 + difference[:, 1, :, :] ** 2
        nmse = 10 * torch.log10((mse.sum(dim=[1, 2]) / power_gt.sum(dim=[1, 2])).mean())

        # 计算Rho
        n = sparse_pred.size(0)
        sparse_pred = sparse_pred.permute(0, 2, 3, 1)  # 将实/虚部维度移到最后
        zeros = sparse_pred.new_zeros((n, nt, nc_expand - nc, 2))
        sparse_pred = torch.cat((sparse_pred, zeros), dim=2)
        raw_pred = torch.fft(sparse_pred, signal_ndim=1)[:, :, :125, :]

        norm_pred = raw_pred[..., 0] ** 2 + raw_pred[..., 1] ** 2
        norm_pred = torch.sqrt(norm_pred.sum(dim=1))

        norm_gt = raw_gt[..., 0] ** 2 + raw_gt[..., 1] ** 2
        norm_gt = torch.sqrt(norm_gt.sum(dim=1))

        real_cross = raw_pred[..., 0] * raw_gt[..., 0] + raw_pred[..., 1] * raw_gt[..., 1]
        real_cross = real_cross.sum(dim=1)
        imag_cross = raw_pred[..., 0] * raw_gt[..., 1] - raw_pred[..., 1] * raw_gt[..., 0]
        imag_cross = imag_cross.sum(dim=1)
        norm_cross = torch.sqrt(real_cross ** 2 + imag_cross ** 2)

        rho = (norm_cross / (norm_pred * norm_gt)).mean()

        return rho, nmse


# def evaluator_cqi(sparse_pred, sparse_gt):
#     """适用于TransCQA的评估器
#     计算规范化均方误差(NMSE)和相关系数(rho)
#     返回NMSE以dB格式
#     """
#     with torch.no_grad():
#         # 计算MSE
#         difference = sparse_gt - sparse_pred
#         mse = difference[:, 0, :, :] ** 2 + difference[:, 1, :, :] ** 2
        
#         # 计算功率
#         power_gt = sparse_gt[:, 0, :, :] ** 2 + sparse_gt[:, 1, :, :] ** 2
        
#         # 避免除零
#         power_sum = power_gt.sum(dim=[1, 2])
#         mse_sum = mse.sum(dim=[1, 2])
        
#         # 过滤掉功率为零的样本
#         valid_mask = power_sum > 1e-10
#         if valid_mask.sum() > 0:
#             # 计算NMSE并转换为dB
#             nmse_ratio = (mse_sum[valid_mask] / power_sum[valid_mask]).mean()
#             nmse_db = 10 * torch.log10(nmse_ratio.clamp(min=1e-10))
#         else:
#             nmse_db = torch.tensor(float('inf'))

#         # 计算相关系数Rho
#         batch_size = sparse_pred.size(0)
        
#         # 展平为向量
#         pred_flat = sparse_pred.view(batch_size, -1)  # (B, 2048)
#         gt_flat = sparse_gt.view(batch_size, -1)      # (B, 2048)
        
#         # 计算皮尔逊相关系数
#         pred_centered = pred_flat - pred_flat.mean(dim=1, keepdim=True)
#         gt_centered = gt_flat - gt_flat.mean(dim=1, keepdim=True)
        
#         # 计算协方差和标准差
#         covariance = (pred_centered * gt_centered).sum(dim=1)
#         pred_std = torch.sqrt((pred_centered ** 2).sum(dim=1))
#         gt_std = torch.sqrt((gt_centered ** 2).sum(dim=1))
        
#         # 避免除零
#         denominator = pred_std * gt_std
#         valid_mask = denominator > 1e-10
        
#         if valid_mask.sum() > 0:
#             rho = (covariance[valid_mask] / denominator[valid_mask]).mean()
#         else:
#             rho = torch.tensor(0.0)

#         return rho, nmse_db


def evaluator_cqi(input, output):
    """适用于TransCQA的评估器
    计算规范化均方误差(NMSE)和相关系数(rho)
    返回NMSE以dB格式
    """
    with torch.no_grad():
        # 计算NMSE
        input = input.reshape(input.size(0), -1)
        output = output.reshape(output.size(0), -1)
        mse = torch.mean((input - output) ** 2, dim=1)
        #mse = comp_mse(input, output)
        norm_factor = torch.mean(input**2, dim=1)
        nmse_per_sample = mse / norm_factor
        nmse_db = 10 * torch.log10(nmse_per_sample)

        # 计算SGCS
        x_flat = input
        y_flat = output

        dot_product = torch.sum(x_flat * y_flat, dim=1)

        norm_X= torch.norm(x_flat, p=2, dim=1)
        norm_Y= torch.norm(y_flat, p=2, dim=1)

        cosine_similarity = dot_product / (norm_X * norm_Y)

        sgcs = cosine_similarity ** 2

        return sgcs.mean().item(), nmse_db.mean().item()

def advanced_evaluator_cqi(sparse_pred, sparse_gt):
    """高级TransCQA评估器
    提供更多评估指标
    """
    with torch.no_grad():
        batch_size = sparse_pred.size(0)
        
        # 基本MSE
        mse = torch.nn.functional.mse_loss(sparse_pred, sparse_gt)
        
        # 功率计算
        power_pred = sparse_pred[:, 0, :, :] ** 2 + sparse_pred[:, 1, :, :] ** 2
        power_gt = sparse_gt[:, 0, :, :] ** 2 + sparse_gt[:, 1, :, :] ** 2
        
        # NMSE
        power_sum = power_gt.sum(dim=[1, 2])
        diff = sparse_pred - sparse_gt
        mse_per_sample = (diff[:, 0, :, :] ** 2 + diff[:, 1, :, :] ** 2).sum(dim=[1, 2])
        
        valid_mask = power_sum > 1e-10
        if valid_mask.sum() > 0:
            nmse = 10 * torch.log10((mse_per_sample[valid_mask] / power_sum[valid_mask]).mean())
        else:
            nmse = torch.tensor(float('inf'))
        
        # 相关系数
        pred_flat = sparse_pred.view(batch_size, -1)
        gt_flat = sparse_gt.view(batch_size, -1)
        
        # 皮尔逊相关系数
        pred_mean = pred_flat.mean(dim=1, keepdim=True)
        gt_mean = gt_flat.mean(dim=1, keepdim=True)
        
        pred_centered = pred_flat - pred_mean
        gt_centered = gt_flat - gt_mean
        
        numerator = (pred_centered * gt_centered).sum(dim=1)
        pred_var = (pred_centered ** 2).sum(dim=1)
        gt_var = (gt_centered ** 2).sum(dim=1)
        
        denominator = torch.sqrt(pred_var * gt_var)
        valid_mask = denominator > 1e-10
        
        if valid_mask.sum() > 0:
            correlation = (numerator[valid_mask] / denominator[valid_mask]).mean()
        else:
            correlation = torch.tensor(0.0)
        
        # 额外指标
        metrics = {
            'mse': mse,
            'nmse': nmse,
            'correlation': correlation,
            'mae': torch.nn.functional.l1_loss(sparse_pred, sparse_gt),
            'power_ratio': (power_pred.sum() / power_gt.sum()).clamp(min=1e-10)
        }
        
        return correlation, nmse, metrics


def compute_csi_metrics(pred, target):
    """计算CSI特定的评估指标"""
    with torch.no_grad():
        # 转换为复数
        pred_complex = torch.complex(pred[:, 0], pred[:, 1])
        target_complex = torch.complex(target[:, 0], target[:, 1])
        
        # 幅度和相位
        pred_mag = torch.abs(pred_complex)
        target_mag = torch.abs(target_complex)
        
        pred_phase = torch.angle(pred_complex)
        target_phase = torch.angle(target_complex)
        
        # 幅度误差
        mag_error = torch.nn.functional.mse_loss(pred_mag, target_mag)
        
        # 相位误差（考虑周期性）
        phase_diff = torch.remainder(pred_phase - target_phase + np.pi, 2*np.pi) - np.pi
        phase_error = torch.mean(phase_diff ** 2)
        
        # 复数误差
        complex_error = torch.nn.functional.mse_loss(pred_complex.real, target_complex.real) + \
                       torch.nn.functional.mse_loss(pred_complex.imag, target_complex.imag)
        
        return {
            'magnitude_mse': mag_error,
            'phase_mse': phase_error,
            'complex_mse': complex_error
        }


def batch_evaluator(predictions, targets, metrics=['nmse', 'correlation', 'mse']):
    """批量评估多个指标"""
    results = {}
    
    if 'nmse' in metrics or 'correlation' in metrics:
        rho, nmse = evaluator_cqi(predictions, targets)
        if 'correlation' in metrics:
            results['correlation'] = rho.item()
        if 'nmse' in metrics:
            results['nmse'] = nmse.item()
    
    if 'mse' in metrics:
        results['mse'] = torch.nn.functional.mse_loss(predictions, targets).item()
    
    if 'mae' in metrics:
        results['mae'] = torch.nn.functional.l1_loss(predictions, targets).item()
    
    if 'csi_metrics' in metrics:
        csi_metrics = compute_csi_metrics(predictions, targets)
        results.update(csi_metrics)
    
    return results


if __name__ == "__main__":
    # 测试评估器
    print("测试TransCQA评估器")
    
    batch_size = 8
    sparse_pred = torch.randn(batch_size, 2, 32, 32)
    sparse_gt = torch.randn(batch_size, 2, 32, 32)
    
    # 测试基本评估器
    rho, nmse = evaluator_cqi(sparse_pred, sparse_gt)
    print(f"基本评估器 - Rho: {rho:.6f}, NMSE: {nmse:.6f}")
    
    # 测试高级评估器
    rho_adv, nmse_adv, metrics_adv = advanced_evaluator_cqi(sparse_pred, sparse_gt)
    print(f"高级评估器 - Rho: {rho_adv:.6f}, NMSE: {nmse_adv:.6f}")
    print(f"额外指标: {metrics_adv}")
    
    # 测试批量评估器
    batch_results = batch_evaluator(sparse_pred, sparse_gt, 
                                  metrics=['nmse', 'correlation', 'mse', 'mae'])
    print(f"批量评估结果: {batch_results}")
    
    print("评估器测试完成!")