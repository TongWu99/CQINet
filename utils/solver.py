import time
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from collections import namedtuple
from tensorboardX import SummaryWriter
from utils import logger
from utils.statics import AverageMeter, evaluator_cqi

__all__ = ['Trainer', 'Tester']

field = ('nmse', 'rho', 'epoch')
Result = namedtuple('Result', field, defaults=(None,) * len(field))

class Trainer:
    """TransCQA训练管道"""

    def __init__(self, model, device, optimizer, criterion, scheduler, cqi_type='wide',
                 resume=None, save_path='./checkpoints', print_freq=20, val_freq=10, test_freq=10,
                 scaler=None, log_dir='data_vision',
                 use_corr_loss=False, lambda_corr_a=0.0, lambda_corr_b=0.0):

        # 基本参数
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device
        self.cqi_type = cqi_type
        self.scaler = scaler  # 混合精度scaler
        self.use_corr_loss = use_corr_loss
        self.lambda_corr_a = lambda_corr_a
        self.lambda_corr_b = lambda_corr_b

        # 详细参数
        self.resume_file = resume
        self.save_path = save_path
        self.print_freq = print_freq
        self.val_freq = val_freq
        self.test_freq = test_freq
        
        # TensorBoard可视化器（支持自定义目录）
        os.makedirs(f"{log_dir}/test", exist_ok=True)
        os.makedirs(f"{log_dir}/best", exist_ok=True)
        os.makedirs(f"{log_dir}/every", exist_ok=True)
        self.vision_test = SummaryWriter(log_dir=f"{log_dir}/test")
        self.vision_best = SummaryWriter(log_dir=f"{log_dir}/best")
        self.vision_every = SummaryWriter(log_dir=f"{log_dir}/every")

        # 流水线参数
        self.cur_epoch = 1
        self.all_epoch = None
        self.train_loss = None
        self.val_loss = None
        self.test_loss = None
        self.best_rho = Result()
        self.best_nmse = Result()
        # 头1000轮最佳（可配置阈值时再抽参，这里按需求固定1000）
        self.best_nmse_1k = Result()

        # 历史记录 - 用于绘制曲线
        self.history = {
            'epochs': [],
            'train_loss': [],
            'val_loss': [],
            'test_loss': [],
            'test_rho': [],
            'test_nmse': [],
            'lr': [],
            'alpha_weight': [],  # alpha_weight历史（按epoch平均）
            'alpha_bias': []     # alpha_bias历史（按epoch平均）
        }

        self.tester = Tester(model, device, criterion, cqi_type, print_freq)
        # 读取可选的梯度裁剪阈值（最小侵入：从优化器上挂载，不破坏旧接口）
        self.grad_clip = getattr(optimizer, 'grad_clip', 0.0)
        
        # 自适应学习率衰减：每200个epoch自动将学习率乘以0.9
        self.lr_decay_interval = 50000  # 每200个epoch衰减一次
        self.min_lr = 1e-4  # 最小学习率
        self.lr_decay_factor = 0.9  # 学习率衰减系数
        self.initial_lr = optimizer.param_groups[0]['lr']  # 记录初始学习率

    def loop(self, epochs, train_loader, val_loader, test_loader):
        """主训练循环"""
        self.all_epoch = epochs
        self._resume()

        for ep in range(self.cur_epoch, epochs + 1):
            
            # 实时打印CQI-Expert权重矩阵（每10个epoch打印一次，仅对MoE模型）
            if ep % 10 == 0 or ep == 1:
                # 只在MoE模型时打印CQI-Expert矩阵，避免对其他模型产生警告
                if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'layer'):
                    layer = self.model.encoder.layer
                    if hasattr(layer, 'moe_ffn') and hasattr(layer.moe_ffn, 'router'):
                        self._print_cqi_expert_matrix()
            
            self.cur_epoch = ep

            # 训练、验证和测试
            self.train_loss = self.train(train_loader)

            # 记录训练损失和学习率
            self.history['epochs'].append(ep)
            self.history['train_loss'].append(self.train_loss.item() if hasattr(self.train_loss, 'item') else self.train_loss)
            self.history['lr'].append(self.scheduler.get_lr()[0])

            # 记录alpha值（仅对TransNet_QC模型）
            if hasattr(self.model, 'cqi_preprocessor'):
                alpha_history = self.model.cqi_preprocessor.get_alpha_history()
                if alpha_history['alpha_weight']:
                    # 计算该epoch内所有batch的alpha平均值
                    avg_alpha_weight = sum(alpha_history['alpha_weight']) / len(alpha_history['alpha_weight'])
                    avg_alpha_bias = sum(alpha_history['alpha_bias']) / len(alpha_history['alpha_bias'])
                    self.history['alpha_weight'].append(avg_alpha_weight)
                    self.history['alpha_bias'].append(avg_alpha_bias)

                    # 记录到TensorBoard
                    self.vision_every.add_scalar("Alpha/Alpha_Weight", avg_alpha_weight, global_step=ep)
                    self.vision_every.add_scalar("Alpha/Alpha_Bias", avg_alpha_bias, global_step=ep)
                else:
                    self.history['alpha_weight'].append(None)
                    self.history['alpha_bias'].append(None)
            else:
                self.history['alpha_weight'].append(None)
                self.history['alpha_bias'].append(None)

            # 记录到TensorBoard - 训练损失和学习率
            self.vision_every.add_scalar("Loss/Train", self.train_loss, global_step=ep)
            self.vision_every.add_scalar("LR/Learning_Rate", self.scheduler.get_lr()[0], global_step=ep)
            
            if ep % self.val_freq == 0:
                self.val_loss = self.val(val_loader)
                self.history['val_loss'].append(self.val_loss.item() if hasattr(self.val_loss, 'item') else self.val_loss)
                # 记录到TensorBoard - 验证损失
                self.vision_every.add_scalar("Loss/Validation", self.val_loss, global_step=ep)
            else:
                self.history['val_loss'].append(None)

            if ep % self.test_freq == 0:
                self.test_loss, rho, nmse = self.test(test_loader)
                self.history['test_loss'].append(self.test_loss.item() if hasattr(self.test_loss, 'item') else self.test_loss)
                self.history['test_rho'].append(rho.item() if hasattr(rho, 'item') else rho)
                self.history['test_nmse'].append(nmse.item() if hasattr(nmse, 'item') else nmse)
                
                # 记录到TensorBoard - 测试指标
                self.vision_test.add_scalar("Metrics/Test_Loss", self.test_loss, global_step=ep)
                self.vision_test.add_scalar("Metrics/Test_Rho", rho, global_step=ep)
                self.vision_test.add_scalar("Metrics/Test_NMSE", nmse, global_step=ep)
                self.vision_every.add_scalar("Metrics/Rho", rho, global_step=ep)
                self.vision_every.add_scalar("Metrics/NMSE", nmse, global_step=ep)
            else:
                rho, nmse = None, None
                self.history['test_loss'].append(None)
                self.history['test_rho'].append(None)
                self.history['test_nmse'].append(None)
            
            # 自适应学习率调整：仅当使用 FakeLR 时才手动衰减
            if hasattr(self.scheduler, '__class__') and self.scheduler.__class__.__name__ == 'FakeLR':
                if ep > 0 and ep % self.lr_decay_interval == 0:
                    # 对每个参数组分别衰减学习率，保持它们之间的相对比例
                    lr_decay_info = []
                    for param_group in self.optimizer.param_groups:
                        current_lr = param_group['lr']
                        new_lr = max(current_lr * self.lr_decay_factor, self.min_lr)
                        param_group['lr'] = new_lr
                        
                        # 记录衰减信息（用于日志）
                        group_name = param_group.get('name', 'default')
                        lr_decay_info.append(f"{group_name}: {current_lr:.2e}→{new_lr:.2e}")
                    
                    # 打印衰减信息
                    decay_str = ', '.join(lr_decay_info)
                    logger.info(f'📉 学习率定期衰减 (每{self.lr_decay_interval}个epoch): {decay_str}')
                    
                    # 记录到TensorBoard（记录第一个组的学习率作为代表）
                    self.vision_every.add_scalar("LR/Periodic_LR_Decay", self.optimizer.param_groups[0]['lr'], global_step=ep)

            # 保存、可视化和日志打印
            self._loop_postprocessing(rho, nmse)
        
        # 训练完成后绘制曲线
        self.plot_training_curves()

    def train(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        with torch.enable_grad():
            return self._iteration(train_loader)

    def val(self, val_loader):
        """验证"""
        self.model.eval()
        with torch.no_grad():
            return self._iteration(val_loader)

    def test(self, test_loader):
        """测试"""
        self.model.eval()
        with torch.no_grad():
            return self.tester(test_loader, verbose=False)

    def _iteration(self, data_loader):
        """单次迭代"""
        iter_loss = AverageMeter('Iter loss')
        iter_time = AverageMeter('Iter time')
        time_tmp = time.time()

        for batch_idx, batch_data in enumerate(data_loader):
            # 根据CQI类型处理数据
            if self.cqi_type == 'nan':
                # nan情况：只有H矩阵
                sparse_gt, sparse_input = batch_data
                sparse_gt = sparse_gt.to(self.device, non_blocking=True)
                sparse_input = sparse_input.to(self.device, non_blocking=True)
                
                # 前向传播（支持混合精度）
                if self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        sparse_pred = self.model(sparse_input)
                else:
                    sparse_pred = self.model(sparse_input)
                
            else:
                # wide/sub情况：有H矩阵和CQI
                sparse_gt, sparse_input, cqi = batch_data
                sparse_gt = sparse_gt.to(self.device, non_blocking=True)
                sparse_input = sparse_input.to(self.device, non_blocking=True)
                cqi = cqi.to(self.device, non_blocking=True)
                
                
                # 前向传播（支持混合精度）
                if self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        try:
                            sparse_pred = self.model(sparse_input, cqi)
                        except TypeError:
                            if self.cqi_type == 'wide':
                                sparse_pred = self.model(sparse_input, wideband_cqi=cqi)
                            elif self.cqi_type == 'sub':
                                sparse_pred = self.model(sparse_input, subband_cqi=cqi)
                else:
                    try:
                        sparse_pred = self.model(sparse_input, cqi)
                    except TypeError:
                        if self.cqi_type == 'wide':
                            sparse_pred = self.model(sparse_input, wideband_cqi=cqi)
                        elif self.cqi_type == 'sub':
                            sparse_pred = self.model(sparse_input, subband_cqi=cqi)
            
            # 计算损失
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    loss = self.criterion(sparse_pred, sparse_input)
            else:
                loss = self.criterion(sparse_pred, sparse_input)

            # 额外的CQI负相关正则（仅在开启开关且存在corr_cache时）
            if self.use_corr_loss and hasattr(self.model, 'corr_cache') and self.model.corr_cache:
                corr_data = self.model.corr_cache
                a_strength = corr_data.get('a_strength', None)
                b_strength = corr_data.get('b_strength', None)
                c_scalar = corr_data.get('cqi_scalar', None)
                if a_strength is not None and b_strength is not None and c_scalar is not None:
                    # 标准化后计算相关系数，并用ReLU只惩罚正相关
                    def _corr_relu(x, y):
                        x_n = (x - x.mean()) / (x.std() + 1e-6)
                        y_n = (y - y.mean()) / (y.std() + 1e-6)
                        return torch.relu((x_n * y_n).mean())
                    L_corr_a = _corr_relu(a_strength, c_scalar)
                    L_corr_b = _corr_relu(b_strength, c_scalar)
                    loss = loss + self.lambda_corr_a * L_corr_a + self.lambda_corr_b * L_corr_b
            
            # 添加负载平衡损失：鼓励所有专家被均匀使用
            if self.model.training:
                if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'layer'):
                    layer = self.model.encoder.layer
                    if hasattr(layer, 'moe_ffn') and hasattr(layer.moe_ffn, '_current_expert_weights'):
                        expert_weights = layer.moe_ffn._current_expert_weights
                        # 计算专家使用率
                        expert_usage = expert_weights.mean(dim=0)  # (num_experts,)
                        # 使用方差度量不均匀程度
                        usage_variance = torch.var(expert_usage)
                        # 添加负载平衡损失（权重0.01）
                        load_balance_weight = 0.01
                        loss = loss + load_balance_weight * usage_variance

            # 反向传播和优化
            if self.model.training:
                self.optimizer.zero_grad()
                
                # 混合精度训练
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    # 可选梯度裁剪（L2范数）
                    if self.grad_clip and self.grad_clip > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    # 可选梯度裁剪（L2范数）
                    if self.grad_clip and self.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.optimizer.step()
                
                # 根据是否有外部scheduler来更新学习率
                # 如果使用的是FakeLR（恒定学习率），则使用手动的lr_decay_interval
                # 否则由外部scheduler控制学习率
                if not (hasattr(self.scheduler, '__class__') and self.scheduler.__class__.__name__ == 'FakeLR'):
                    # 使用外部scheduler更新学习率
                    self.scheduler.step()

            # 记录和更新
            iter_loss.update(loss)
            iter_time.update(time.time() - time_tmp)
            time_tmp = time.time()

            # 打印进度（显示优化器的lr，这才是真实值）
            if (batch_idx + 1) % self.print_freq == 0:
                current_lr = self.optimizer.param_groups[0]['lr']  # 获取优化器的实际lr
                
                # 检查是否有分组学习率（Router独立学习率）
                router_lr_str = ""
                if len(self.optimizer.param_groups) > 1:
                    # 查找Router组的学习率
                    for group in self.optimizer.param_groups:
                        if 'name' in group and group['name'] == 'router':
                            router_lr = group['lr']
                            router_lr_str = f' | router_lr: {router_lr:.2e}'
                            break
                
                logger.info(f'Epoch: [{self.cur_epoch}/{self.all_epoch}]'
                           f'[{batch_idx + 1}/{len(data_loader)}] '
                           f'lr: {current_lr:.2e}{router_lr_str} | '
                           f'MSE loss: {iter_loss.avg:.3e} | '
                           f'time: {iter_time.avg:.3f}')
                self.vision_every.add_scalar("lr", current_lr, global_step=self.cur_epoch)
                self.vision_every.add_scalar("MSE loss", iter_loss.avg, self.cur_epoch)
                
                # 如果有Router学习率，也记录到TensorBoard
                if router_lr_str:
                    for group in self.optimizer.param_groups:
                        if 'name' in group and group['name'] == 'router':
                            self.vision_every.add_scalar("LR/Router_LR", group['lr'], global_step=self.cur_epoch)
                            break

        mode = 'Train' if self.model.training else 'Val'
        logger.info(f'=> {mode} Loss: {iter_loss.avg:.3e}\n')

        return iter_loss.avg
    
    def _compute_load_balance_loss(self):
        """计算MoE负载平衡损失，防止专家利用不均"""
        try:
            # 获取所有MoE层的expert weights
            load_balance_loss = 0.0
            num_moe_layers = 0
            
            # 遍历encoder的所有层
            if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'layer'):
                layer = self.model.encoder.layer
                if hasattr(layer, 'moe_ffn') and hasattr(layer.moe_ffn, '_current_expert_weights'):
                    weights = layer.moe_ffn._current_expert_weights  # (batch_size, num_experts)
                    # 计算负载平衡损失：鼓励所有专家被均匀使用
                    expert_usage = weights.mean(dim=0)  # (num_experts,) 每个专家的平均使用率
                    # 使用方差度量不均匀程度
                    load_balance_loss += torch.var(expert_usage)
                    num_moe_layers += 1
            
            # 遍历decoder的所有层
            if hasattr(self.model, 'decoder') and hasattr(self.model.decoder, 'layer'):
                layer = self.model.decoder.layer
                if hasattr(layer, 'moe_ffn') and hasattr(layer.moe_ffn, '_current_expert_weights'):
                    weights = layer.moe_ffn._current_expert_weights
                    expert_usage = weights.mean(dim=0)
                    load_balance_loss += torch.var(expert_usage)
                    num_moe_layers += 1
            
            if num_moe_layers > 0:
                return load_balance_loss / num_moe_layers
            else:
                return torch.tensor(0.0, device=self.device)
        except:
            return torch.tensor(0.0, device=self.device)

    def _save(self, state, name):
        """保存检查点"""
        if self.save_path is None:
            logger.warning('没有保存路径')
            return

        os.makedirs(self.save_path, exist_ok=True)
        torch.save(state, os.path.join(self.save_path, name))

    def _resume(self):
        """从检查点恢复"""
        if self.resume_file is None:
            return None
            
        assert os.path.isfile(self.resume_file), f"检查点文件不存在: {self.resume_file}"
        logger.info(f'=> 正在加载检查点 {self.resume_file}')
        
        checkpoint = torch.load(self.resume_file)
        self.cur_epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.best_rho = checkpoint.get('best_rho', Result())
        self.best_nmse = checkpoint.get('best_nmse', Result())
        
        # 恢复自适应学习率相关参数（兼容旧版本checkpoint）
        self.lr_decay_interval = checkpoint.get('lr_decay_interval', 200)
        self.initial_lr = checkpoint.get('initial_lr', self.optimizer.param_groups[0]['lr'])
        
        # 恢复训练历史（如果存在）
        if 'training_history' in checkpoint:
            self.history = checkpoint['training_history']
            logger.info('=> 训练历史已恢复')
        
        logger.info(f'=> 自适应学习率: 每{self.lr_decay_interval}个epoch衰减, 最小lr={self.min_lr:.2e}')
        
        self.cur_epoch += 1  # 从下一个epoch开始

        logger.info(f'=> 成功加载检查点 {self.resume_file} '
                   f'从epoch {checkpoint["epoch"]}开始\n')

    def _print_cqi_expert_matrix(self):
        """打印CQI-Expert权重矩阵（16×专家数）"""
        try:
            import numpy as np
            
            # 获取encoder中的router
            if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'layer'):
                layer = self.model.encoder.layer
                if hasattr(layer, 'moe_ffn') and hasattr(layer.moe_ffn, 'router'):
                    router = layer.moe_ffn.router
                    moe_ffn = layer.moe_ffn
                    
                    # 获取权重矩阵
                    cqi_expert_matrix = router.get_cqi_expert_matrix()
                    
                    # 获取Top-K专家序号
                    top_k = getattr(moe_ffn, 'top_k', moe_ffn.num_experts)
                    cqi_top_k_experts = router.get_cqi_top_k_experts(top_k=top_k)
                    
                    if cqi_expert_matrix is not None:
                        logger.info('\n' + '='*80)
                        logger.info(f'CQI-Expert权重矩阵 (Epoch {self.cur_epoch})')
                        logger.info('='*80)
                        
                        # 如果有Top-K策略，输出激活的专家序号
                        if top_k < moe_ffn.num_experts:
                            logger.info(f'Top-K策略: 每个CQI激活前{top_k}个专家')
                            logger.info('-'*80)
                            logger.info('CQI值 | 激活的专家序号 | ' + ' | '.join([f'专家{i}' for i in range(cqi_expert_matrix.shape[1])]))
                            logger.info('-'*80)
                            
                            for cqi_val, (weights, top_k_experts) in enumerate(zip(cqi_expert_matrix, cqi_top_k_experts)):
                                weight_str = ' | '.join([f'{w:.3f}' for w in weights])
                                active_str = ', '.join([str(e) for e in top_k_experts])
                                max_expert = np.argmax(weights)
                                logger.info(f'  {cqi_val:2d}  | [{active_str}] | {weight_str} [Expert {max_expert}]')
                        else:
                            logger.info('CQI值 | ' + ' | '.join([f'专家{i}' for i in range(cqi_expert_matrix.shape[1])]))
                            logger.info('-'*80)
                            
                            for cqi_val, weights in enumerate(cqi_expert_matrix):
                                weight_str = ' | '.join([f'{w:.3f}' for w in weights])
                                max_expert = np.argmax(weights)
                                logger.info(f'  {cqi_val:2d}  | {weight_str} [Expert {max_expert}]')
                        
                        logger.info('='*80 + '\n')
                        
                        # 统计每个专家被选择的次数
                        if top_k < moe_ffn.num_experts:
                            self._print_expert_selection_count(cqi_top_k_experts, moe_ffn.num_experts)
        except Exception as e:
            logger.warning(f'打印CQI-Expert权重矩阵失败: {e}')
    
    def _print_expert_selection_count(self, cqi_top_k_experts, num_experts):
        """统计并打印每个专家被选择的次数"""
        import numpy as np
        
        # 统计每个专家被选中的次数
        expert_count = np.zeros(num_experts, dtype=int)
        for cqi_experts in cqi_top_k_experts:
            for expert_idx in cqi_experts:
                expert_count[expert_idx] += 1
        
        # 计算总次数（所有CQI × Top-K）
        total_selections = len(cqi_top_k_experts) * len(cqi_top_k_experts[0])
        
        # 计算每个专家被选中的比例
        expert_ratio = expert_count / total_selections
        
        logger.info('='*80)
        logger.info(f'专家被选择统计 (Epoch {self.cur_epoch})')
        logger.info('='*80)
        logger.info(f'总选择次数: {total_selections} (16个CQI × Top-{len(cqi_top_k_experts[0])})')
        logger.info('-'*80)
        logger.info('专家ID | 被选择次数 | 被选择比例 | 状态')
        logger.info('-'*80)
        
        # 按被选次数排序
        sorted_indices = np.argsort(expert_count)[::-1]
        
        for expert_idx in sorted_indices:
            count = expert_count[expert_idx]
            ratio = expert_ratio[expert_idx]
            status = "✅ 活跃" if count > 0 else "⚠️  未激活"
            logger.info(f'   {expert_idx:2d}  |      {count:3d}     |   {ratio:.2%}   | {status}')
        
        logger.info('='*80 + '\n')
        
        # 记录到TensorBoard
        try:
            for expert_idx in range(num_experts):
                # 记录每个专家的被选择次数
                self.vision_every.add_scalar(
                    f'Expert/Expert_{expert_idx}_Selection_Count',
                    expert_count[expert_idx],
                    global_step=self.cur_epoch
                )
                # 记录每个专家的被选择比例
                self.vision_every.add_scalar(
                    f'Expert/Expert_{expert_idx}_Selection_Ratio',
                    expert_ratio[expert_idx],
                    global_step=self.cur_epoch
                )
            
            # 记录专家使用的方差（度量不均匀程度）
            usage_variance = np.var(expert_ratio)
            self.vision_every.add_scalar(
                'Expert/Expert_Usage_Variance',
                usage_variance,
                global_step=self.cur_epoch
            )
            
            # 记录活跃专家数量（被选择次数>0的专家数）
            active_experts = np.sum(expert_count > 0)
            self.vision_every.add_scalar(
                'Expert/Active_Experts_Count',
                active_experts,
                global_step=self.cur_epoch
            )
            
            # 记录未激活专家数量
            inactive_experts = num_experts - active_experts
            self.vision_every.add_scalar(
                'Expert/Inactive_Experts_Count',
                inactive_experts,
                global_step=self.cur_epoch
            )
            
        except Exception as e:
            logger.warning(f'记录专家统计到TensorBoard失败: {e}')
    
    def _loop_postprocessing(self, rho, nmse):
        """后处理"""
        # 生成保存状态
        state = {
            'epoch': self.cur_epoch,
            'lr_decay_interval': self.lr_decay_interval,
            'initial_lr': self.initial_lr,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_rho': self.best_rho,
            'best_nmse': self.best_nmse,
            'best_nmse_1k': self.best_nmse_1k,
            'cqi_type': self.cqi_type,
            'training_history': self.history  # 保存训练历史
        }

        # 保存最佳rho和nmse模型
        if rho is not None:
            if self.best_rho.rho is None or self.best_rho.rho < rho:
                self.best_rho = Result(rho=rho, nmse=nmse, epoch=self.cur_epoch)
                state['best_rho'] = self.best_rho
                self._save(state, name=f"best_rho_{self.cqi_type}.pth")
                
            if self.best_nmse.nmse is None or self.best_nmse.nmse > nmse:
                self.best_nmse = Result(rho=rho, nmse=nmse, epoch=self.cur_epoch)
                state['best_nmse'] = self.best_nmse
                self._save(state, name=f"best_nmse_{self.cqi_type}.pth")

            # 头1000轮最佳NMSE（仅记录，不另存检查点）
            if self.cur_epoch <= 1000:
                if self.best_nmse_1k.nmse is None or self.best_nmse_1k.nmse > nmse:
                    self.best_nmse_1k = Result(rho=rho, nmse=nmse, epoch=self.cur_epoch)
                    state['best_nmse_1k'] = self.best_nmse_1k

        self._save(state, name=f'last_{self.cqi_type}.pth')

        # 打印当前最佳结果
        if self.best_rho.rho is not None:
            print(f'\n=! 最佳 rho: {self.best_rho.rho:.3e} ('
                  f'对应 nmse={self.best_rho.nmse:.3e}; '
                  f'epoch={self.best_rho.epoch})'
                  f'\n   最佳 NMSE: {self.best_nmse.nmse:.3e} ('
                  f'对应 rho={self.best_nmse.rho:.3e}; '
                  f'epoch={self.best_nmse.epoch})\n')
            self.vision_best.add_scalar("best rho", self.best_rho.rho, global_step=self.best_rho.epoch)
            self.vision_best.add_scalar("best nmse", self.best_nmse.nmse, global_step=self.best_nmse.epoch)

    def plot_training_curves(self):
        """绘制训练过程曲线"""
        try:
            # 设置matplotlib参数
            plt.style.use('seaborn-v0_8-darkgrid')
            fig, axes = plt.subplots(3, 2, figsize=(15, 18))
            fig.suptitle(f'TransCQA Training Curves - {self.cqi_type.upper()} CQI', fontsize=16, fontweight='bold')
            #self.history = self.history.to('cpu')

            epochs = self.history['epochs']

            # 1. 损失曲线
            ax1 = axes[0, 0]
            ax1.plot(epochs, self.history['train_loss'], 'b-', label='Training Loss', linewidth=2)

            # 验证损失（只绘制非None值）
            val_epochs = [ep for ep, loss in zip(epochs, self.history['val_loss']) if loss is not None]
            val_losses = [loss for loss in self.history['val_loss'] if loss is not None]
            if val_losses:
                ax1.plot(val_epochs, val_losses, 'r--', label='Validation Loss', linewidth=2)

            # 测试损失（只绘制非None值）
            test_epochs = [ep for ep, loss in zip(epochs, self.history['test_loss']) if loss is not None]
            test_losses = [loss for loss in self.history['test_loss'] if loss is not None]
            if test_losses:
                ax1.plot(test_epochs, test_losses, 'g:', label='Test Loss', linewidth=2, alpha=0.8)

            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('MSE Loss')
            ax1.set_title('Loss Curves')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_yscale('log')  # 使用对数尺度显示损失

            # 2. NMSE曲线
            ax2 = axes[0, 1]
            nmse_epochs = [ep for ep, nmse in zip(epochs, self.history['test_nmse']) if nmse is not None]
            nmse_values = [nmse for nmse in self.history['test_nmse'] if nmse is not None]
            if nmse_values:
                ax2.plot(nmse_epochs, nmse_values, 'purple', marker='o', markersize=3, linewidth=2)
                ax2.axhline(y=self.best_nmse.nmse if self.best_nmse.nmse else min(nmse_values),
                           color='red', linestyle='--', alpha=0.7,
                           label=f'Best NMSE: {self.best_nmse.nmse if self.best_nmse.nmse else min(nmse_values):.2f} dB')
            
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('NMSE (dB)')
            ax2.set_title('NMSE Curve')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # 3. Rho (相关系数) 曲线
            ax3 = axes[1, 0]
            rho_epochs = [ep for ep, rho in zip(epochs, self.history['test_rho']) if rho is not None]
            rho_values = [rho for rho in self.history['test_rho'] if rho is not None]
            if rho_values:
                ax3.plot(rho_epochs, rho_values, 'orange', marker='s', markersize=3, linewidth=2)
                ax3.axhline(y=self.best_rho.rho if self.best_rho.rho else max(rho_values),
                           color='red', linestyle='--', alpha=0.7,
                           label=f'Best Rho: {self.best_rho.rho if self.best_rho.rho else max(rho_values):.4f}')

            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Correlation Coefficient (Rho)')
            ax3.set_title('Rho Curve')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            # 4. 学习率曲线
            ax4 = axes[1, 1]
            ax4.plot(epochs, self.history['lr'], 'brown', linewidth=2)
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Learning Rate')
            ax4.set_title('Learning Rate Schedule')
            ax4.grid(True, alpha=0.3)
            ax4.set_yscale('log')  # 使用对数尺度显示学习率

            # 5. Alpha值曲线（仅TransNet_QC模型有）
            ax5 = axes[2, 0]
            alpha_weight_epochs = [ep for ep, aw in zip(epochs, self.history['alpha_weight']) if aw is not None]
            alpha_weight_values = [aw for aw in self.history['alpha_weight'] if aw is not None]
            if alpha_weight_values:
                ax5.plot(alpha_weight_epochs, alpha_weight_values, 'green', marker='^', markersize=3, linewidth=2, label='Alpha_Weight')
                ax5.set_xlabel('Epoch')
                ax5.set_ylabel('Alpha Weight')
                ax5.set_title('Alpha Weight Curve (TransNet_QC)')
                ax5.legend()
                ax5.grid(True, alpha=0.3)

            # 6. Alpha Bias曲线
            ax6 = axes[2, 1]
            alpha_bias_epochs = [ep for ep, ab in zip(epochs, self.history['alpha_bias']) if ab is not None]
            alpha_bias_values = [ab for ab in self.history['alpha_bias'] if ab is not None]
            if alpha_bias_values:
                ax6.plot(alpha_bias_epochs, alpha_bias_values, 'red', marker='v', markersize=3, linewidth=2, label='Alpha_Bias')
                ax6.set_xlabel('Epoch')
                ax6.set_ylabel('Alpha Bias')
                ax6.set_title('Alpha Bias Curve (TransNet_QC)')
                ax6.legend()
                ax6.grid(True, alpha=0.3)

            # 调整布局
            plt.tight_layout()
            
            # 保存图片
            if self.save_path:
                os.makedirs(self.save_path, exist_ok=True)
                save_path = os.path.join(self.save_path, f'training_curves_{self.cqi_type}.png')
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f'训练曲线已保存到: {save_path}')
                
                # 也保存到项目根目录
                root_save_path = f'training_curves_{self.cqi_type}.png'
                plt.savefig(root_save_path, dpi=300, bbox_inches='tight')
                logger.info(f'训练曲线也保存到: {root_save_path}')
            
            # 显示图片（如果在交互环境中）
            try:
                plt.show()
            except:
                pass  # 在非交互环境中忽略显示错误
            
            plt.close()  # 关闭图形以释放内存
            
            # 保存训练历史数据
            self._save_training_history()
            
        except Exception as e:
            logger.warning(f'绘制训练曲线时出错: {e}')
    
    def _save_training_history(self):
        """保存训练历史数据到文件"""
        try:
            if self.save_path:
                history_file = os.path.join(self.save_path, f'training_history_{self.cqi_type}.txt')
                with open(history_file, 'w') as f:
                    f.write(f'TransCQA Training History - {self.cqi_type.upper()} CQI\n')
                    f.write('=' * 50 + '\n')
                    f.write(f'Total Epochs: {len(self.history["epochs"])}\n')
                    f.write(f'Best NMSE: {self.best_nmse.nmse:.3f} dB (Epoch {self.best_nmse.epoch})\n')
                    f.write(f'Best Rho: {self.best_rho.rho:.6f} (Epoch {self.best_rho.epoch})\n')
                    f.write(f'Final Train Loss: {self.history["train_loss"][-1]:.6e}\n')
                    if any(x is not None for x in self.history["test_nmse"]):
                        final_nmse = [x for x in self.history["test_nmse"] if x is not None][-1]
                        f.write(f'Final NMSE: {final_nmse:.3f} dB\n')
                    if any(x is not None for x in self.history["test_rho"]):
                        final_rho = [x for x in self.history["test_rho"] if x is not None][-1]
                        f.write(f'Final Rho: {final_rho:.6f}\n')
                    f.write('\n')
                    
                    # 详细历史数据
                    f.write('Detailed Training History:\n')
                    f.write('-' * 50 + '\n')
                    f.write('Epoch\tTrain_Loss\tVal_Loss\tTest_Loss\tNMSE(dB)\tRho\t\tLR\t\tAlpha_Weight\tAlpha_Bias\n')
                    for i, epoch in enumerate(self.history['epochs']):
                        f.write(f'{epoch}\t{self.history["train_loss"][i]:.6e}\t')
                        f.write(f'{self.history["val_loss"][i] if self.history["val_loss"][i] is not None else "N/A"}\t')
                        f.write(f'{self.history["test_loss"][i] if self.history["test_loss"][i] is not None else "N/A"}\t')
                        f.write(f'{self.history["test_nmse"][i] if self.history["test_nmse"][i] is not None else "N/A"}\t')
                        f.write(f'{self.history["test_rho"][i] if self.history["test_rho"][i] is not None else "N/A"}\t')
                        f.write(f'{self.history["lr"][i]:.6e}\t')
                        f.write(f'{self.history["alpha_weight"][i] if self.history["alpha_weight"][i] is not None else "N/A"}\t')
                        f.write(f'{self.history["alpha_bias"][i] if self.history["alpha_bias"][i] is not None else "N/A"}\n')
                
                logger.info(f'训练历史数据已保存到: {history_file}')
                
        except Exception as e:
            logger.warning(f'保存训练历史数据时出错: {e}')


class Tester:
    """TransCQA测试接口"""

    def __init__(self, model, device, criterion, cqi_type='wide', print_freq=20):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.cqi_type = cqi_type
        self.print_freq = print_freq

    def __call__(self, test_data, verbose=True):
        """运行测试过程"""
        self.model.eval()
        with torch.no_grad():
            loss, rho, nmse = self._iteration(test_data)
        
        if verbose:
            print(f'\n=> 测试结果: \nloss: {loss:.3e}'
                  f'    rho: {rho:.3e}    NMSE: {nmse:.3e}\n')
        
        return loss, rho, nmse

    def _iteration(self, data_loader):
        """测试迭代"""
        iter_rho = AverageMeter('Iter rho')
        iter_nmse = AverageMeter('Iter nmse')
        iter_loss = AverageMeter('Iter loss')
        iter_time = AverageMeter('Iter time')
        time_tmp = time.time()

        for batch_idx, batch_data in enumerate(data_loader):
            # 根据CQI类型处理数据
            if self.cqi_type == 'nan':
                # nan情况：只有H矩阵
                sparse_gt, sparse_input = batch_data
                sparse_gt = sparse_gt.to(self.device)
                sparse_input = sparse_input.to(self.device)
                
                # 前向传播
                sparse_pred = self.model(sparse_input)
                
            else:
                # wide/sub情况：有H矩阵和CQI
                sparse_gt, sparse_input, cqi = batch_data
                sparse_gt = sparse_gt.to(self.device)
                sparse_input = sparse_input.to(self.device)
                cqi = cqi.to(self.device)
                
                # 前向传播 - try positional argument first (for TransNet_MoE), then keyword arguments
                try:
                    sparse_pred = self.model(sparse_input, cqi)
                except TypeError:
                    # Fallback to keyword arguments for other models
                    if self.cqi_type == 'wide':
                        sparse_pred = self.model(sparse_input, wideband_cqi=cqi)
                    elif self.cqi_type == 'sub':
                        sparse_pred = self.model(sparse_input, subband_cqi=cqi)

            # 计算损失和评估指标
            loss = self.criterion(sparse_pred, sparse_input)
            
            # 使用适配的评估器
            rho, nmse = evaluator_cqi(sparse_input, sparse_pred)

            # 记录和更新
            iter_loss.update(loss)
            iter_rho.update(rho)
            iter_nmse.update(nmse)
            iter_time.update(time.time() - time_tmp)
            time_tmp = time.time()

            # 打印进度
            if (batch_idx + 1) % self.print_freq == 0:
                logger.info(f'[{batch_idx + 1}/{len(data_loader)}] '
                           f'loss: {iter_loss.avg:.3e} | rho: {iter_rho.avg:.3e} | '
                           f'NMSE: {iter_nmse.avg:.3e} | time: {iter_time.avg:.3f}')

        logger.info(f'=> 测试 rho:{iter_rho.avg:.3e}  NMSE: {iter_nmse.avg:.3e}\n')

        return iter_loss.avg, iter_rho.avg, iter_nmse.avg


def analyze_model_performance(model, test_loader, device, cqi_type='wide'):
    """分析模型性能"""
    model.eval()
    fusion_analyses = []
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_loader):
            if batch_idx >= 10:  # 只分析前10个batch
                break
                
            if cqi_type != 'nan':
                sparse_gt, sparse_input, cqi = batch_data
                sparse_input = sparse_input.to(device)
                cqi = cqi.to(device)
                
                # 分析融合权重
                if hasattr(model, 'analyze_fusion_weights'):
                    fusion_analysis = model.analyze_fusion_weights(sparse_input, cqi)
                    if fusion_analysis:
                        fusion_analyses.append(fusion_analysis)
    
    # 汇总分析结果
    if fusion_analyses:
        avg_analysis = {}
        for key in fusion_analyses[0].keys():
            avg_analysis[key] = sum(fa[key] for fa in fusion_analyses) / len(fusion_analyses)
        
        logger.info("融合权重分析结果:")
        for key, value in avg_analysis.items():
            logger.info(f"  {key}: {value:.6f}")
    
    return fusion_analyses


if __name__ == "__main__":
    print("TransCQA Solver模块测试")
    print("请在完整训练环境中测试此模块")