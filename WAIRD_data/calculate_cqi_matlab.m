% 确保已安装必要的工具箱（Statistics and Machine Learning Toolbox用于t-SNE）
clear all
close all
clc
%% ==================== 参数配置 ====================
input_csi_file = 'C:\Users\GreyGoo\Desktop\Sem\data\00390.mat';
% 输出文件名前缀（不含扩展名）
output_prefix = 'losangeles_adCSI_right_00390_test_mapping2';  % 例如: scenario2_00032_BS0

% 系统参数
Nt = [1, 8, 4];  % BS天线配置 [x, y, z]
Nr = [1, 1, 1];  % UE天线配置 [x, y, z]
Nt_total = Nt(1) * Nt(2) * Nt(3);  % 发射天线总数
Nr_total = Nr(1) * Nr(2) * Nr(3);  % 接收天线总数
sampledCarriers = 52;  % 采样子载波数
BWGHz = 0.00936;  % 带宽 (GHz) 0.00936
subcarrier_spacing = 15e3;  % 子载波间隔 (Hz) 15e3

% 噪声和功率参数
k_B = 1.38e-23;  % 玻尔兹曼常数 (J/K)
T = 290;  % 系统温度 (K)
noise_figure = 5;  % 噪声系数 (dB)
total_power_dBm = 20;  % 总发射功率 (dBm)

% 子带参数
subband_size = 4;  % 每个子带包含的子载波数

% 3GPP标准 SINR-to-CQI 映射表 (dB)上为旧映射
%SINR_to_CQI_mapping = [-6.3, -5.8, -1.4, 3.9, 5.3, 8.1, 9.8, ...
%     11.7, 13.6, 15.8, 18.8, 21.4, 23.6, 28.2, 32.0];
SINR_to_CQI_mapping = [-11.2, -6.9, -2.2, 2.7, 4.3, 6.9, 8.5, ...
     10.6, 12.4, 14.4, 17.5, 18.1, 20.2, 22.8, 24.9];
% 可视化选项
enable_plots = true;  % 是否生成图表
enable_tsne = false;  % 是否生成t-SNE图（需要Statistics and Machine Learning Toolbox）
enable_csi_visualization = true;  % 是否可视化CSI矩阵
num_sample_users_for_plot = 0;  % 用于CSI可视化的示例用户数量
enable_dft_compare = false;      % 是否绘制DFT前后H矩阵对比
dft_compare_user_idx = 5000;       % 用于DFT前后对比的用户序号
enable_antenna_only_dft = false;   % 仅对天线维度做DFT（子载波不变）；false为双维DFT
enable_cqi_heatmap_mean = false;       % 按Wideband CQI分组绘制DFT前CSI的平均热力图
enable_dft_cqi_heatmap_mean = false;   % 按Wideband CQI分组绘制DFT后CSI的平均热力图
enable_rb_subcqi_heatmap = false;       % 按RB绘制扇形区域内用户的subband CQI均值热力图
enable_subcqi_distribution = false;    % 分析所有用户的subband CQI分布情况（各RB间差距）
enable_snr_heatmap = false;           % 绘制用户计算CQI时的SNR连续值平面热力图

% ========== 单用户Subband CQI分析 ==========
enable_single_user_subcqi = false;      % 是否启用单用户分析
single_user_idx = 5;                    % 指定用户序号 (1-10000)
sector_center_deg = 90;              % 扇区中心方向（相对于正北+Y，顺时针为正，度）
sector_width_deg = 120;             % 扇区总夹角（度），默认120度

% 统一的CQI颜色映射（16级，从红到绿渐变：CQI=0最红，CQI=15最绿）
cqi_colormap = [linspace(1, 0, 16)', linspace(0, 1, 16)', zeros(16, 1)];

%% ==================== 读取数据 ====================
fprintf('正在读取CSI数据...\n');

% 检查文件是否存在
if ~exist(input_csi_file, 'file')
    error('CSI文件不存在: %s\n请先运行com.py生成合并文件，并使用convert_npy_to_mat.py转换为.mat格式', input_csi_file);
end

% 读取.mat文件
[~, ~, ext] = fileparts(input_csi_file);
if strcmpi(ext, '.mat')
    % 直接读取.mat文件
    fprintf('读取.mat文件: %s\n', input_csi_file);
    loaded_data = load(input_csi_file);
    
    % 查找CSI数据变量（可能是csi_data或其他名称）
    var_names = fieldnames(loaded_data);
    csi_var_found = false;
    for i = 1:length(var_names)
        var_name = var_names{i};
        var_data = loaded_data.(var_name);
        if isnumeric(var_data) && length(size(var_data)) == 4 && size(var_data, 1) == 2
            csi_data = var_data;
            csi_var_name = var_name;
            csi_var_found = true;
            fprintf('找到CSI数据变量: %s\n', var_name);
            break;
        end
    end
    
    if ~csi_var_found
        error('无法在.mat文件中找到CSI数据（期望4维数组，第1维为2）');
    end
    
    if ~isfield(loaded_data, 'scenario_info')
        error('.mat 文件必须包含 scenario_info 结构体：请重新导出数据');
    end
    scenario_info_from_file = loaded_data.scenario_info;
    fprintf('✓ 使用.mat文件中的场景信息\n');
    
else
    % 尝试读取.npy文件（如果安装了npy-matlab工具）
    try
        csi_data = readNPY(input_csi_file);
        fprintf('成功读取.npy文件\n');
    catch
        error('请使用convert_npy_to_mat.py将.npy文件转换为.mat格式');
    end
end

% 检查维度
if length(size(csi_data)) ~= 4 || size(csi_data, 1) ~= 2
    error('CSI数据维度不正确。期望: (2, num_users, Nt*Nr, sampledCarriers)，实际: %s', mat2str(size(csi_data)));
end

num_users = size(csi_data, 2);
antenna_pairs = size(csi_data, 3);
num_subcarriers = size(csi_data, 4);

fprintf('CSI数据维度: %s\n', mat2str(size(csi_data)));
fprintf('用户数量: %d\n', num_users);
fprintf('天线对数量: %d (期望: %d)\n', antenna_pairs, Nt_total * Nr_total);
fprintf('子载波数: %d (期望: %d)\n', num_subcarriers, sampledCarriers);

% 重构复数CSI矩阵
% 形状: (num_users, Nt*Nr, sampledCarriers)
H_complex = csi_data(1, :, :, :) + 1j * csi_data(2, :, :, :);
H_complex = squeeze(H_complex);  % 移除单例维度
% 重新排列为: (Nt*Nr, sampledCarriers, num_users)
H_complex = permute(H_complex, [2, 3, 1]);

fprintf('复数CSI矩阵维度: %s\n', mat2str(size(H_complex)));

%% ==================== 读取位置信息 ====================
fprintf('\n读取场景信息...\n');
scenario_info = scenario_info_from_file;

img_size = scenario_info.img_size(:)';
bs_loc = scenario_info.bs_loc(:)';
ue_locs = scenario_info.ue_locs;
if isfield(scenario_info, 'environment_image')
    env_image_array = scenario_info.environment_image;
else
    env_image_array = [];
end
if isfield(scenario_info, 'env_id')
    env_id = scenario_info.env_id;
else
    env_id = '';
end
if isfield(scenario_info, 'carrier_freq')
    carrier_freq = scenario_info.carrier_freq;
else
    carrier_freq = '';
end
if isfield(scenario_info, 'scenario')
    scenario = scenario_info.scenario;
else
    scenario = 0;
end

fprintf('图像尺寸: [%.0f, %.0f]\n', img_size(1), img_size(2));
fprintf('BS位置: (%.2f, %.2f)\n', bs_loc(1), bs_loc(2));
fprintf('UE数量(位置): %d\n', size(ue_locs, 1));
if ~isempty(env_image_array)
    fprintf('环境图像尺寸: %s\n', mat2str(size(env_image_array)));
else
    fprintf('提示: 场景信息中未包含环境图像\n');
end

%% ==================== 计算噪声功率 ====================
fprintf('\n计算系统参数...\n');

% 计算有效带宽
bandwidth_effective = num_subcarriers * 12 * subcarrier_spacing;  % 假设1个RB=12个子载波

% 计算噪声功率
N0 = k_B * T;  % 热噪声密度 (W/Hz)
N0_dB = 10 * log10(N0) + 30;  % 转换为 dBm/Hz
noise_power_dBm = N0_dB + 10 * log10(bandwidth_effective) + noise_figure;
noise_power = 10^((noise_power_dBm - 30) / 10);  % 转换为线性单位 (W)

% 计算每个子载波的发射功率
total_power_linear = 10^((total_power_dBm - 30) / 10);  % 转换为线性单位 (W)
power_per_subcarrier = total_power_linear / num_subcarriers;

fprintf('有效带宽: %.2f MHz\n', bandwidth_effective / 1e6);
fprintf('噪声功率: %.2e W (%.2f dBm)\n', noise_power, noise_power_dBm);
fprintf('每子载波功率: %.2e W (%.2f dBm)\n', power_per_subcarrier, ...
    10*log10(power_per_subcarrier) + 30);

%% ==================== DFT变换（频域到角度-延迟域）====================
fprintf('\n执行DFT变换（频域 -> 角度-延迟域）...\n');

% 保存原始频域H（用于CQI计算，因为CQI基于频域SNR）
H_complex_freq = H_complex;  % (Nt*Nr, sampledCarriers, num_users)

% 初始化DFT后的H（角度-延迟域，用于保存）
H_complex_angle_delay = zeros(size(H_complex));  % (Nt*Nr, sampledCarriers, num_users)

% 生成DFT矩阵 
% 对天线维度做DFT：dftmtx(Nt*Nr)
% 对子载波维度做DFT：dftmtx(sampledCarriers)
DFT_antenna = dftmtx(Nt_total);  % (Nt*Nr, Nt*Nr)
DFT_subcarrier = dftmtx(sampledCarriers);  % (sampledCarriers, sampledCarriers)

fprintf('DFT矩阵尺寸: 天线维度 %s, 子载波维度 %s\n', ...
    mat2str(size(DFT_antenna)), mat2str(size(DFT_subcarrier)));
if enable_antenna_only_dft
    fprintf('DFT模式: 仅天线维度DFT，子载波维度保持不变。\n');
else
    fprintf('DFT模式: 天线+子载波双维DFT。\n');
end

% 对每个用户进行DFT变换
fprintf('处理用户: ');
for user_idx = 1:num_users
    if mod(user_idx, 1000) == 0
        fprintf('%d/%d ', user_idx, num_users);
    end
    
    % 获取该用户的频域H矩阵，形状 (Nt*Nr, sampledCarriers)
    H_user_freq = H_complex_freq(:, :, user_idx);
    
    if enable_antenna_only_dft
        % 仅对天线维度做DFT，子载波维度保持频域
        H_user_angle_delay = DFT_antenna * H_user_freq;
    else
        % 双维DFT：天线+子载波
        % 注意：MATLAB中矩阵乘法是列优先，所以顺序正确
        H_user_angle_delay = DFT_antenna * H_user_freq * DFT_subcarrier;
    end
    
    % 保存DFT后的H
    H_complex_angle_delay(:, :, user_idx) = H_user_angle_delay;
end
fprintf('\nDFT变换完成！\n');
%H_complex_angle_delay = H_complex_freq;
fprintf('DFT变换后的H矩阵维度: %s\n', mat2str(size(H_complex_angle_delay)));


%% ==================== CQI计算 ====================
fprintf('\n开始计算CQI...\n');

% 计算子带数量
num_subbands = floor(num_subcarriers / subband_size);

% 初始化结果数组
wideband_avg_SNR_dB = zeros(1, num_users);
wideband_CQI_results = zeros(1, num_users);
subband_avg_SNR_dB = zeros(num_subbands, num_users);
subband_CQI_results = zeros(num_subbands, num_users);

% 对每个用户计算CQI
fprintf('处理用户: ');
for user_idx = 1:num_users
    if mod(user_idx, 1000) == 0
        fprintf('%d/%d ', user_idx, num_users);
    end
    
    % 获取该用户的CSI矩阵（使用DFT后的角度-延迟域H进行CQI计算），形状 (Nt*Nr, sampledCarriers)
    H_user = H_complex_freq(:, :, user_idx);
    
    % ========== Wideband CQI 计算 ==========
    sum_SNR_dB = 0;
    for sc_idx = 1:num_subcarriers
        % 提取该子载波的信道向量
        H_sc = H_user(:, sc_idx);  % 形状 (antenna_pairs,)
        
        % MRC波束赋形
        H_norm = norm(H_sc);
        if H_norm < 1e-10
            SNR_dB = -inf;
        else
            v = H_sc / H_norm;  % 波束赋形向量
            received_power = abs(H_sc' * v)^2 * power_per_subcarrier;
            SNR_linear = received_power / noise_power;
            SNR_dB = 10 * log10(SNR_linear);
        end
        sum_SNR_dB = sum_SNR_dB + SNR_dB;
    end
    
    wideband_avg_SNR_dB(user_idx) = sum_SNR_dB / num_subcarriers;
    % 映射到CQI
    wideband_CQI_results(user_idx) = sum(wideband_avg_SNR_dB(user_idx) > SINR_to_CQI_mapping);
    wideband_CQI_results(user_idx) = min(wideband_CQI_results(user_idx), 15);
    
    % ========== Subband CQI 计算 ==========
    for subband_idx = 1:num_subbands
        start_sc = (subband_idx - 1) * subband_size + 1;
        end_sc = subband_idx * subband_size;
        
        sum_SNR_dB_subband = 0;
        for sc_idx = start_sc:end_sc
            H_sc = H_user(:, sc_idx);
            H_norm = norm(H_sc);
            if H_norm < 1e-10
                SNR_dB = -inf;
            else
                v = H_sc / H_norm;
                received_power = abs(H_sc' * v)^2 * power_per_subcarrier;
                SNR_linear = received_power / noise_power;
                SNR_dB = 10 * log10(SNR_linear);
            end
            sum_SNR_dB_subband = sum_SNR_dB_subband + SNR_dB;
        end
        
        subband_avg_SNR_dB(subband_idx, user_idx) = sum_SNR_dB_subband / subband_size;
        % 映射到CQI
        subband_CQI_results(subband_idx, user_idx) = sum(...
            subband_avg_SNR_dB(subband_idx, user_idx) > SINR_to_CQI_mapping);
        subband_CQI_results(subband_idx, user_idx) = min(...
            subband_CQI_results(subband_idx, user_idx), 15);
    end
end
fprintf('\nCQI计算完成！\n');

% 打印统计信息
fprintf('\nCQI统计信息:\n');
fprintf('  Wideband CQI范围: [%d, %d]\n', min(wideband_CQI_results), max(wideband_CQI_results));
fprintf('  Wideband CQI均值: %.2f\n', mean(wideband_CQI_results));
fprintf('  Wideband SNR范围: [%.2f, %.2f] dB\n', ...
    min(wideband_avg_SNR_dB), max(wideband_avg_SNR_dB));
fprintf('  Wideband SNR均值: %.2f dB\n', mean(wideband_avg_SNR_dB));

%% ==================== 按Wideband CQI分组的CSI平均热力图（DFT前）====================
if enable_plots && enable_cqi_heatmap_mean
    fprintf('\n按Wideband CQI分组绘制CSI平均热力图（DFT前）...\n');

    for cqi_val = 0:15
        user_mask = round(wideband_CQI_results) == cqi_val;
        num_cqi_users = sum(user_mask);
        if num_cqi_users == 0
            continue;
        end

        % 取该CQI组所有用户的频域CSI，求幅度均值
        H_group = H_complex_freq(:, :, user_mask);
        avg_amp = mean(abs(H_group), 3);  % (Nt*Nr, sampledCarriers)

        figure('Position', [150, 150, 800, 400]);
        imagesc(avg_amp);
        colorbar;
        xlabel('Subcarrier Index');
        ylabel('Antenna Pair Index');
        title(sprintf('Wideband CQI = %d (n = %d) - Mean |H| Before DFT', cqi_val, num_cqi_users));
        colormap(gca, 'jet');
    end
end

%% ==================== 按Wideband CQI分组的CSI平均热力图（DFT后）====================
if enable_plots && enable_dft_cqi_heatmap_mean
    fprintf('\n按Wideband CQI分组绘制CSI平均热力图（DFT后）...\n');

    % 计算DFT后CSI的全局最大/最小值，用于统一色阶
    all_dft_amp = abs(H_complex_angle_delay);
    global_max = max(all_dft_amp(:));
    global_min = min(all_dft_amp(:));

    for cqi_val = 0:15
        user_mask = round(wideband_CQI_results) == cqi_val;
        num_cqi_users = sum(user_mask);
        if num_cqi_users == 0
            continue;
        end

        % 取该CQI组所有用户的DFT后CSI，求幅度均值
        H_group = H_complex_angle_delay(:, :, user_mask);
        avg_amp = mean(abs(H_group), 3);  % (Nt*Nr, sampledCarriers)

        figure('Position', [150, 150, 800, 400]);
        imagesc(avg_amp, [global_min, global_max]);
        colorbar;
        xlabel('Delay Index (DFT of Subcarriers)');
        ylabel('Angle Index (DFT of Antennas)');
        title(sprintf('Wideband CQI = %d (n = %d) - Mean |H| After DFT', cqi_val, num_cqi_users));
        colormap(gca, 'jet');
    end
end

%% ==================== 单用户DFT前后H矩阵热力图对比 ====================
if enable_plots && enable_dft_compare
    fprintf('\n生成DFT前后H矩阵热力图对比 (用户 %d)...\n', dft_compare_user_idx);
    u = round(dft_compare_user_idx);
    if u < 1 || u > num_users
        warning('dft_compare_user_idx = %d 超出用户范围 [1, %d]，跳过DFT前后对比绘制。', u, num_users);
    else
        H_user_freq = H_complex_freq(:, :, u);
        H_user_angle_delay = H_complex_angle_delay(:, :, u);
        
        figure('Position', [200, 200, 1100, 450]);
        
        subplot(1, 2, 1);
        imagesc(abs(H_user_freq));
        colorbar;
        xlabel('Subcarrier Index');
        ylabel('Antenna Pair Index');
        title(sprintf('User %d: |H| Before DFT (Frequency Domain)', u));
        colormap(gca, 'jet');
        
        subplot(1, 2, 2);
        imagesc(abs(H_user_angle_delay));
        colorbar;
        xlabel('Delay-Doppler / Subcarrier Index');
        ylabel('Angle / Antenna Index');
        title(sprintf('User %d: |H| After DFT (Angle-Delay Domain)', u));
        colormap(gca, 'jet');
        
        sgtitle(sprintf('DFT Effect on Channel Matrix Sparsity - User %d', u), 'FontSize', 14);
    end
end

%% ==================== 基站正方向自定义扇形区域用户筛选 ====================
fprintf('\n计算基站正方向自定义扇形区域的用户...\n');

% 基站正方向是正北（+y方向），角度以正北为0度，顺时针为正
% sector_center_deg: 扇区中心方向（相对正北），sector_width_deg: 总夹角
% 例如：center=0,width=120 即正北左右各60度；center=90,width=90 即正东左右各45度
if ~isempty(ue_locs) && any(ue_locs(:) ~= 0) && ~isempty(bs_loc)
    % 确保坐标是双精度
    ue_locs = double(ue_locs);
    bs_loc = double(bs_loc);
    
    % 计算从基站到每个用户的向量
    vec_to_users = ue_locs - bs_loc;  % (num_users, 2)
    
    % 计算每个用户相对于基站的角度（以正北为0度，顺时针为正）
    % 正北方向是 [0, 1]，角度为90度（在atan2中）
    % 但我们需要以正北为0度，所以需要转换
    angles_to_users = atan2(vec_to_users(:,1), vec_to_users(:,2)) * 180 / pi;  % 转换为度
    
    % 计算扇区内用户
    sector_half_angle = sector_width_deg / 2;
    angle_diff = mod(angles_to_users - sector_center_deg + 180, 360) - 180; % 映射到[-180,180]
    users_in_sector = abs(angle_diff) <= sector_half_angle;
    
    num_users_in_sector = sum(users_in_sector);
    fprintf('  基站位置: (%.2f, %.2f)\n', bs_loc(1), bs_loc(2));
    fprintf('  正方向: 正北（+y方向）\n');
    fprintf('  扇区中心方向: %.2f° (相对正北，顺时针为正)\n', sector_center_deg);
    fprintf('  扇区总夹角: %.2f° (左右各 %.2f°)\n', sector_width_deg, sector_half_angle);
    fprintf('  扇区内用户数量: %d / %d (%.1f%%)\n', ...
        num_users_in_sector, num_users, 100*num_users_in_sector/num_users);
    if num_users_in_sector > 0
        fprintf('  扇区内角度范围: [%.2f°, %.2f°]\n', ...
            min(angle_diff(users_in_sector)), max(angle_diff(users_in_sector)));
        % 计算并输出扇区内用户的wideband CQI均值，便于与全体用户对比
        mean_cqi_all = mean(wideband_CQI_results);
        mean_cqi_sector = mean(wideband_CQI_results(users_in_sector));
        fprintf('  Wideband CQI均值（全体用户）  : %.2f\n', mean_cqi_all);
        fprintf('  Wideband CQI均值（扇区内用户）: %.2f\n', mean_cqi_sector);
        fprintf('  差值（扇区内 - 全体）         : %.2f\n', mean_cqi_sector - mean_cqi_all);
    end
else
    warning('无法计算扇区内用户：位置信息缺失或无效');
    users_in_sector = false(num_users, 1);
    num_users_in_sector = 0;
    angles_to_users = [];
    angle_diff = [];
end

%% ==================== CSI矩阵可视化 ====================
if enable_plots && enable_csi_visualization
    fprintf('\n生成CSI矩阵可视化...\n');
    
    % 选择几个示例用户进行可视化
    sample_users = round(linspace(1, num_users, num_sample_users_for_plot));
    
    for plot_idx = 1:length(sample_users)
        user_idx = sample_users(plot_idx);
        H_user = H_complex(:, :, user_idx);
        
        figure('Position', [100 + plot_idx*50, 100 + plot_idx*50, 1000, 400]);
        
        % 幅度
        subplot(1, 2, 1);
        imagesc(abs(H_user));
        colorbar;
        xlabel('Subcarrier Index');
        ylabel('Antenna Pair Index');
        title(sprintf('CSI Amplitude - User %d (CQI=%d)', user_idx, wideband_CQI_results(user_idx)));
        colormap(gca, 'jet');
        
        % 相位
        subplot(1, 2, 2);
        imagesc(angle(H_user) / pi * 180);
        colorbar;
        xlabel('Subcarrier Index');
        ylabel('Antenna Pair Index');
        title(sprintf('CSI Phase (degrees) - User %d', user_idx));
        colormap(gca, 'hsv');
        
        sgtitle(sprintf('Channel Matrix Visualization - User %d', user_idx), 'FontSize', 14);
    end
end

%% ==================== t-SNE可视化 ====================
if enable_plots && enable_tsne
    fprintf('\n生成t-SNE可视化...\n');
    try
        plot_tsne_for_users(H_complex_angle_delay, true(1, num_users), wideband_CQI_results, ...
            antenna_pairs, num_subcarriers, ...
            't-SNE of Channel Matrix H (Colored by CQI)', false);%%H_complex_angle_delay
    catch ME
        warning('t-SNE可视化失败: %s\n请确保已安装Statistics and Machine Learning Toolbox', ME.message);
    end
end

%% ==================== 扇区内用户的t-SNE可视化 ====================
if enable_plots && enable_tsne && num_users_in_sector > 0
    fprintf('\n生成扇区内用户的t-SNE可视化...\n');
    try
        title_sector = sprintf('t-SNE of Channel Matrix H (Sector Users, n=%d, Colored by CQI)', ...
            num_users_in_sector);
        plot_tsne_for_users(H_complex_angle_delay, users_in_sector, wideband_CQI_results, ...
            antenna_pairs, num_subcarriers, title_sector, true);
    catch ME
        warning('扇区内用户t-SNE可视化失败: %s\n请确保已安装Statistics and Machine Learning Toolbox', ME.message);
    end
end

%% ==================== CDF图 ====================
if enable_plots
    fprintf('\n生成CDF图...\n');
    
    figure('Position', [100, 100, 800, 600]);
    
    % CQI CDF
    subplot(2, 2, 1);
    cdfplot(wideband_CQI_results);
    xlabel('CQI Value');
    ylabel('CDF');
    title('Wideband CQI CDF');
    grid on;
    xlim([0, 16]);
    
    % SNR CDF
    subplot(2, 2, 2);
    cdfplot(wideband_avg_SNR_dB);
    xlabel('SNR (dB)');
    ylabel('CDF');
    title('Wideband SNR CDF');
    grid on;
    
    % Subband CQI分布（箱线图）
    subplot(2, 2, 3);
    boxplot(subband_CQI_results', 'Labels', 1:num_subbands);
    xlabel('Subband Index');
    ylabel('CQI Value');
    title('Subband CQI Distribution');
    grid on;
    
    % CQI直方图
    subplot(2, 2, 4);
    histogram(wideband_CQI_results, 0:16, 'Normalization', 'probability');
    xlabel('CQI Value');
    ylabel('Probability');
    title('Wideband CQI Histogram');
    grid on;
    xlim([0, 16]);
    
    sgtitle('CQI and SNR Statistics', 'FontSize', 14);
end

%% ==================== 单用户 Subband CQI 分析 ====================
if enable_plots && enable_single_user_subcqi
    if single_user_idx < 1 || single_user_idx > num_users
        fprintf('错误: 用户序号 %d 超出范围 [1, %d]\n', single_user_idx, num_users);
    else
        fprintf('\n分析用户 #%d 的 Subband CQI 情况...\n', single_user_idx);

        % 获取该用户的subband CQI数据
        user_subcqi = subband_CQI_results(:, single_user_idx);  % (num_subbands, 1)
        user_wideband_cqi = wideband_CQI_results(single_user_idx);

        % 计算统计量
        user_cqi_mean = mean(user_subcqi);
        user_cqi_std = std(user_subcqi);
        user_cqi_range = max(user_subcqi) - min(user_subcqi);
        user_cqi_max_adj_diff = max(abs(diff(user_subcqi)));  % 相邻RB最大差距
        user_cqi_mean_adj_diff = mean(abs(diff(user_subcqi)));  % 相邻RB平均差距

        % 绘制分析图
        figure('Position', [200, 200, 1200, 700]);

        % 子图1：每个RB的CQI值（折线图）
        subplot(2, 2, 1);
        x_rb = 1:num_subbands;
        plot(x_rb, user_subcqi, 'b-o', 'LineWidth', 2, 'MarkerSize', 6);
        hold on;
        yline(user_wideband_cqi, 'r--', sprintf('Wideband CQI = %d', user_wideband_cqi), ...
            'LineWidth', 1.5, 'LabelHorizontalAlignment', 'left');
        hold off;
        xlabel('RB (Subband) Index');
        ylabel('CQI Value');
        title(sprintf('User #%d: Subband CQI per RB', single_user_idx));
        grid on;
        ylim([0, 16]);
        xticks(1:num_subbands);

        % 子图2：每个RB的CQI值（柱状图）
        subplot(2, 2, 2);
        bar(x_rb, user_subcqi, 'FaceColor', [0.3, 0.5, 0.8], 'FaceAlpha', 0.8);
        hold on;
        yline(user_cqi_mean, 'r--', sprintf('Mean = %.2f', user_cqi_mean), 'LineWidth', 1.5);
        hold off;
        xlabel('RB (Subband) Index');
        ylabel('CQI Value');
        title(sprintf('User #%d: Subband CQI Bar Chart', single_user_idx));
        grid on;
        ylim([0, 16]);
        xticks(1:num_subbands);

        % 子图3：相邻RB差距
        subplot(2, 2, 3);
        adj_diff = diff(user_subcqi);
        bar(1:(num_subbands-1), adj_diff, 'FaceColor', [0.8, 0.4, 0.4], 'FaceAlpha', 0.8);
        hold on;
        yline(0, 'k-', 'LineWidth', 1);
        hold off;
        xlabel('RB Gap (RB i to RB i+1)');
        ylabel('CQI Difference');
        title(sprintf('Adjacent RB CQI Difference (Max=%.0f, Mean=%.2f)', ...
            user_cqi_max_adj_diff, user_cqi_mean_adj_diff));
        grid on;

        % 子图4：统计摘要
        subplot(2, 2, 4);
        axis off;
        summary_text = {
            sprintf('===== User #%d Subband CQI Summary =====', single_user_idx),
            '',
            sprintf('Wideband CQI: %d', user_wideband_cqi),
            '',
            'Subband CQI Statistics:',
            sprintf('  Mean: %.2f', user_cqi_mean),
            sprintf('  Std:  %.2f', user_cqi_std),
            sprintf('  Range: [%.0f, %.0f]', min(user_subcqi), max(user_subcqi)),
            sprintf('  Max-Min Range: %d', user_cqi_range),
            '',
            'Adjacent RB Differences:',
            sprintf('  Max Abs Diff: %.0f', user_cqi_max_adj_diff),
            sprintf('  Mean Abs Diff: %.2f', user_cqi_mean_adj_diff),
            '',
            'Detailed Subband CQI:'
        };
        text(0.05, 0.95, summary_text, 'VerticalAlignment', 'top', ...
            'FontName', 'Consolas', 'FontSize', 11, 'Interpreter', 'none');

        % 右侧添加详细的每个RB的CQI值
        detail_text = {};
        for i = 1:num_subbands
            detail_text{end+1} = sprintf('  RB %2d: CQI = %.0f', i, user_subcqi(i));
        end
        text(0.5, 0.95, detail_text, 'VerticalAlignment', 'top', ...
            'FontName', 'Consolas', 'FontSize', 10, 'Interpreter', 'none');

        sgtitle(sprintf('User #%d Subband CQI Analysis', single_user_idx), 'FontSize', 14);

        % 命令行输出
        fprintf('\n  ===== User #%d Subband CQI =====\n', single_user_idx);
        fprintf('    Wideband CQI: %d\n', user_wideband_cqi);
        fprintf('    Subband CQI Mean: %.2f, Std: %.2f\n', user_cqi_mean, user_cqi_std);
        fprintf('    Range: [%.0f, %.0f], Max-Min = %d\n', ...
            min(user_subcqi), max(user_subcqi), user_cqi_range);
        fprintf('    Adjacent RB Max Diff: %.0f, Mean Diff: %.2f\n', ...
            user_cqi_max_adj_diff, user_cqi_mean_adj_diff);
        fprintf('\n    各RB CQI值:\n      ');
        fprintf('%.0f ', user_subcqi);
        fprintf('\n');
    end
end

%% ==================== 检查原始CSI数据的变化幅度 ====================
% 这个功能用于诊断：为什么很多用户的subband CQI相同
enable_csi_diagnosis = true;  % 设置为true来诊断CSI数据

if enable_plots && enable_csi_diagnosis
    fprintf('\n========== CSI 数据诊断 ==========\n');

    % 随机抽取10个用户
    rng(42);  % 固定随机种子以便复现
    sample_users = sort(randperm(num_users, min(10, num_users)));

    for ui = 1:length(sample_users)
        user_idx = sample_users(ui);

        % 获取该用户的频域CSI
        H_user = H_complex_freq(:, :, user_idx);  % (Nt*Nr, sampledCarriers)

        % 计算每个子载波的接收功率（用于CQI计算）
        user_sc_power = zeros(1, sampledCarriers);
        for sc_idx = 1:sampledCarriers
            H_sc = H_user(:, sc_idx);
            user_sc_power(sc_idx) = norm(H_sc)^2;  % MRC下的等效功率
        end

        % 计算每个RB的平均功率
        user_rb_power = zeros(1, num_subbands);
        for rb_idx = 1:num_subbands
            start_sc = (rb_idx - 1) * subband_size + 1;
            end_sc = rb_idx * subband_size;
            user_rb_power(rb_idx) = mean(user_sc_power(start_sc:end_sc));
        end

        % 转换为dB
        user_rb_power_dB = 10 * log10(user_rb_power + eps);

        fprintf('\n  User #%d (Wideband CQI=%d):\n', user_idx, wideband_CQI_results(user_idx));
        fprintf('    RB功率(dB)范围: [%.2f, %.2f], 变化: %.2f dB\n', ...
            min(user_rb_power_dB), max(user_rb_power_dB), max(user_rb_power_dB) - min(user_rb_power_dB));
        fprintf('    13个RB功率: '); fprintf('%.1f ', user_rb_power_dB); fprintf(' dB\n');
    end

    % 检查所有用户RB功率变化的统计
    fprintf('\n  --- 所有用户RB功率变化统计 ---\n');
    all_rb_power_range = zeros(1, num_users);
    for user_idx = 1:num_users
        H_user = H_complex_freq(:, :, user_idx);
        user_sc_power = zeros(1, sampledCarriers);
        for sc_idx = 1:sampledCarriers
            H_sc = H_user(:, sc_idx);
            user_sc_power(sc_idx) = norm(H_sc)^2;
        end
        user_rb_power = zeros(1, num_subbands);
        for rb_idx = 1:num_subbands
            start_sc = (rb_idx - 1) * subband_size + 1;
            end_sc = rb_idx * subband_size;
            user_rb_power(rb_idx) = mean(user_sc_power(start_sc:end_sc));
        end
        user_rb_power_dB = 10 * log10(user_rb_power + eps);
        all_rb_power_range(user_idx) = max(user_rb_power_dB) - min(user_rb_power_dB);
    end

    fprintf('    用户RB功率变化范围: 均值=%.2f dB, 中位数=%.2f dB\n', ...
        mean(all_rb_power_range), median(all_rb_power_range));
    fprintf('    用户RB功率变化=0的比例: %.1f%%\n', 100*sum(all_rb_power_range==0)/num_users);
    fprintf('    用户RB功率变化<1dB的比例: %.1f%%\n', 100*sum(all_rb_power_range<1)/num_users);
    fprintf('    用户RB功率变化<3dB的比例: %.1f%%\n', 100*sum(all_rb_power_range<3)/num_users);

    % 绘制功率变化分布
    figure('Position', [300, 300, 800, 400]);
    subplot(1, 2, 1);
    histogram(all_rb_power_range, 30, 'Normalization', 'probability');
    xlabel('RB Power Range (dB)');
    ylabel('Probability');
    title('Distribution of RB Power Variation per User');
    grid on;

    subplot(1, 2, 2);
    % 检查有多少用户的subband CQI都相同
    user_subcqi_std = std(subband_CQI_results, 0, 1);  % 每个用户的13个CQI的标准差
    histogram(user_subcqi_std, 0:0.5:8, 'Normalization', 'probability');
    xlabel('User Subband CQI Std Dev');
    ylabel('Probability');
    title('Distribution of Subband CQI Variation per User');
    grid on;

    fprintf('\n  --- Subband CQI变化统计 ---\n');
    fprintf('    用户subband CQI标准差: 均值=%.2f, 中位数=%.2f\n', ...
        mean(user_subcqi_std), median(user_subcqi_std));
    fprintf('    用户subband CQI变化=0的比例: %.1f%%\n', 100*sum(user_subcqi_std==0)/num_users);

    sgtitle('CSI Data Diagnosis', 'FontSize', 14);
end
if enable_plots && enable_subcqi_distribution
    fprintf('\n分析Subband CQI分布情况...\n');

    % subband_CQI_results: (num_subbands, num_users)
    % 计算每个RB上所有用户的CQI统计
    mean_cqi_per_rb = mean(subband_CQI_results, 2);  % (num_subbands, 1)
    std_cqi_per_rb = std(subband_CQI_results, 0, 2);  % (num_subbands, 1)
    min_cqi_per_rb = min(subband_CQI_results, [], 2);  % (num_subbands, 1)
    max_cqi_per_rb = max(subband_CQI_results, [], 2);  % (num_subbands, 1)

    % ========== 每个用户自己13个Subband CQI之间的差距 ==========
    % 计算每个用户的subband CQI最大值和最小值之差
    user_cqi_range = max(subband_CQI_results, [], 1) - min(subband_CQI_results, [], 1);  % (1, num_users)
    % 计算每个用户的subband CQI标准差
    user_cqi_std = std(subband_CQI_results, 0, 1);  % (1, num_users)
    % 计算每个用户的subband CQI均值
    user_cqi_mean = mean(subband_CQI_results, 1);  % (1, num_users)

    % 计算相邻RB之间CQI均值的差距（仅作为参考）
    rb_diff = diff(mean_cqi_per_rb);  % (num_subbands-1, 1)
    max_rb_diff = max(abs(rb_diff));
    mean_rb_diff = mean(abs(rb_diff));

    % 计算每个用户相邻RB之间的最大差距
    user_rb_diff = diff(subband_CQI_results, 1, 1);  % (num_subbands-1, num_users)
    user_max_adj_diff = max(abs(user_rb_diff), [], 1);  % (1, num_users)
    user_mean_adj_diff = mean(abs(user_rb_diff), 1);  % (1, num_users)

    % 绘制分析图
    figure('Position', [100, 100, 1400, 900]);

    % 子图1：每个RB的CQI均值柱状图（带误差棒）
    subplot(2, 3, 1);
    hold on;
    x_rb = 1:num_subbands;
    bar(x_rb, mean_cqi_per_rb, 'FaceColor', [0.4, 0.6, 0.8], 'FaceAlpha', 0.7);
    errorbar(x_rb, mean_cqi_per_rb, std_cqi_per_rb, 'k.', 'LineWidth', 1);
    hold off;
    xlabel('RB (Subband) Index');
    ylabel('Mean CQI');
    title('Mean Subband CQI per RB (with Std Dev)');
    grid on;
    ylim([0, 16]);

    % 子图2：每个RB的CQI范围（min-max阴影）
    subplot(2, 3, 2);
    x_rb = 1:num_subbands;
    fill([x_rb, fliplr(x_rb)], [min_cqi_per_rb', fliplr(max_cqi_per_rb')], ...
        [0.4, 0.6, 0.8], 'FaceAlpha', 0.3, 'EdgeColor', 'b', 'LineWidth', 1);
    hold on;
    plot(x_rb, mean_cqi_per_rb, 'r-', 'LineWidth', 2, 'DisplayName', 'Mean');
    plot(x_rb, min_cqi_per_rb, 'b--', 'LineWidth', 1, 'DisplayName', 'Min');
    plot(x_rb, max_cqi_per_rb, 'g--', 'LineWidth', 1, 'DisplayName', 'Max');
    hold off;
    xlabel('RB (Subband) Index');
    ylabel('CQI Value');
    title('Subband CQI Range per RB');
    legend('Location', 'best');
    grid on;
    ylim([0, 16]);

    % 子图3：相邻RB之间CQI均值的差距（参考图）
    subplot(2, 3, 3);
    x_diff = 1:(num_subbands-1);
    bar(x_diff, abs(rb_diff), 'FaceColor', [0.8, 0.3, 0.3], 'FaceAlpha', 0.7);
    hold on;
    yline(mean_rb_diff, 'r--', sprintf('Mean=%.3f', mean_rb_diff), 'LineWidth', 2);
    hold off;
    xlabel('RB Gap (RB i to RB i+1)');
    ylabel('|CQI Difference|');
    title('Avg Adjacent RB Diff (ALL users avg)');
    grid on;

    % 子图4：每个用户自己的Subband CQI范围分布
    subplot(2, 3, 4);
    histogram(user_cqi_range, 0:1:max(user_cqi_range), 'Normalization', 'probability');
    hold on;
    yline(0, 'w');
    hold off;
    xlabel('CQI Range (max - min across 13 RBs)');
    ylabel('Probability');
    title(sprintf('User-level Subband CQI Range Distribution\n(Each user has 13 RBs, showing range)'));
    grid on;

    % 子图5：每个用户相邻RB之间最大差距的分布
    subplot(2, 3, 5);
    histogram(user_max_adj_diff, 0:1:max(user_max_adj_diff), 'Normalization', 'probability');
    xlabel('Max Adjacent RB Diff (per user)');
    ylabel('Probability');
    title('User-level Max Adjacent RB Difference');
    grid on;

    % 子图6：热力图 - 每个RB上CQI值的分布
    subplot(2, 3, 6);
    cqi_hist = zeros(16, num_subbands);
    for rb_idx = 1:num_subbands
        for cqi_val = 0:15
            cqi_hist(cqi_val+1, rb_idx) = sum(subband_CQI_results(rb_idx, :) == cqi_val);
        end
    end
    cqi_prob = cqi_hist ./ num_users;
    imagesc(1:num_subbands, 0:15, cqi_prob);
    colorbar;
    colormap(gca, 'jet');
    xlabel('RB (Subband) Index');
    ylabel('CQI Value');
    title('Subband CQI Distribution Heatmap');
    set(gca, 'YDir', 'normal', 'YTick', 0:2:15);

    sgtitle(sprintf('Subband CQI Distribution Analysis (n=%d users, %d RBs)', ...
        num_users, num_subbands), 'FontSize', 14);

    % 打印统计信息
    fprintf('\n  ========== Subband CQI 统计信息 ==========\n');
    fprintf('    用户数量: %d\n', num_users);
    fprintf('    RB数量: %d\n', num_subbands);
    fprintf('    每个RB的子载波数: %d\n', subband_size);

    fprintf('\n  --- 各RB平均CQI (所有用户平均) ---\n');
    for rb_idx = 1:num_subbands
        fprintf('      RB %2d: Mean=%.2f, Std=%.2f, Range=[%d, %d]\n', ...
            rb_idx, mean_cqi_per_rb(rb_idx), std_cqi_per_rb(rb_idx), ...
            min_cqi_per_rb(rb_idx), max_cqi_per_rb(rb_idx));
    end

    fprintf('\n  --- 相邻RB间CQI差距 (所有用户平均后) ---\n');
    fprintf('      最大差距: %.4f\n', max_rb_diff);
    fprintf('      平均差距: %.4f\n', mean_rb_diff);

    fprintf('\n  --- 每个用户自己13个RB之间的差距 ---\n');
    fprintf('    【重要】每个用户的CQI范围 (max-min):\n');
    fprintf('      均值: %.2f\n', mean(user_cqi_range));
    fprintf('      中位数: %.2f\n', median(user_cqi_range));
    fprintf('      范围: [%d, %d]\n', min(user_cqi_range), max(user_cqi_range));
    fprintf('      分布: Range=0 占 %.1f%%, Range=1 占 %.1f%%, Range>=2 占 %.1f%%\n', ...
        100*sum(user_cqi_range==0)/num_users, ...
        100*sum(user_cqi_range==1)/num_users, ...
        100*sum(user_cqi_range>=2)/num_users);

    fprintf('\n    每个用户的相邻RB最大差距:\n');
    fprintf('      均值: %.2f\n', mean(user_max_adj_diff));
    fprintf('      中位数: %.2f\n', median(user_max_adj_diff));
    fprintf('      范围: [%d, %d]\n', min(user_max_adj_diff), max(user_max_adj_diff));

    fprintf('\n    每个用户的相邻RB平均差距:\n');
    fprintf('      均值: %.2f\n', mean(user_mean_adj_diff));
    fprintf('      中位数: %.2f\n', median(user_mean_adj_diff));
end

%% ==================== 扇区内用户的Wide CQI分布对比 ====================
if enable_plots && num_users_in_sector > 0
    fprintf('\n生成扇区内用户的Wide CQI分布对比图...\n');
    
    % 提取扇区内用户的CQI
    % 确保都是列向量
    cqi_all = wideband_CQI_results(:);  % 强制转换为列向量
    cqi_in_sector = wideband_CQI_results(users_in_sector);
    cqi_in_sector = cqi_in_sector(:);  % 强制转换为列向量
    
    figure('Position', [100, 100, 1200, 500]);
    
    % 子图1: CDF对比
    subplot(1, 3, 1);
    hold on;
    [f_all, x_all] = ecdf(cqi_all);
    [f_sector, x_sector] = ecdf(cqi_in_sector);
    plot(x_all, f_all, 'b-', 'LineWidth', 2, 'DisplayName', sprintf('全体用户 (n=%d)', num_users));
    plot(x_sector, f_sector, 'r--', 'LineWidth', 2, 'DisplayName', sprintf('扇区内 (n=%d)', num_users_in_sector));
    xlabel('Wideband CQI Value', 'FontSize', 11);
    ylabel('CDF', 'FontSize', 11);
    title('Wideband CQI CDF 对比', 'FontSize', 12);
    legend('Location', 'southeast', 'FontSize', 10);
    grid on;
    xlim([0, 16]);
    hold off;
    
    % 子图2: 直方图对比
    subplot(1, 3, 2);
    hold on;
    histogram(cqi_all, 0:16, 'Normalization', 'probability', ...
        'FaceColor', 'b', 'FaceAlpha', 0.5, 'EdgeColor', 'b', ...
        'DisplayName', sprintf('全体用户 (n=%d)', num_users));
    histogram(cqi_in_sector, 0:16, 'Normalization', 'probability', ...
        'FaceColor', 'r', 'FaceAlpha', 0.7, 'EdgeColor', 'r', ...
        'DisplayName', sprintf('扇区内 (n=%d)', num_users_in_sector));
    xlabel('Wideband CQI Value', 'FontSize', 11);
    ylabel('Probability', 'FontSize', 11);
    title('Wideband CQI 分布对比', 'FontSize', 12);
    legend('Location', 'best', 'FontSize', 10);
    grid on;
    xlim([0, 16]);
    hold off;
    
    % 子图3: 箱线图对比
    subplot(1, 3, 3);
    % 确保都是列向量后再拼接
    cqi_combined = [cqi_all(:); cqi_in_sector(:)];
    group_labels = [ones(length(cqi_all), 1); 2*ones(length(cqi_in_sector), 1)];
    boxplot(cqi_combined, group_labels, 'Labels', ...
        {sprintf('全体用户\n(n=%d)', num_users), ...
         sprintf('扇区内\n(n=%d)', num_users_in_sector)}, ...
        'Colors', 'br');
    ylabel('Wideband CQI Value', 'FontSize', 11);
    title('Wideband CQI 箱线图对比', 'FontSize', 12);
    grid on;
    ylim([0, 16]);
    
    sgtitle('基站正方向扇区内用户 Wide CQI 分布对比', 'FontSize', 14);
    
    % 打印统计信息
    fprintf('  全体用户 CQI统计:\n');
    fprintf('    均值: %.2f, 中位数: %.2f, 标准差: %.2f\n', ...
        mean(cqi_all), median(cqi_all), std(cqi_all));
    fprintf('    范围: [%d, %d]\n', min(cqi_all), max(cqi_all));
    fprintf('  扇区内用户 CQI统计:\n');
    fprintf('    均值: %.2f, 中位数: %.2f, 标准差: %.2f\n', ...
        mean(cqi_in_sector), median(cqi_in_sector), std(cqi_in_sector));
    fprintf('    范围: [%d, %d]\n', min(cqi_in_sector), max(cqi_in_sector));
end

%% ==================== 扇形区域内用户的RB级Subband CQI均值热力图 ====================
if enable_plots && enable_rb_subcqi_heatmap && num_users_in_sector > 0
    fprintf('\n生成扇形区域内用户的RB级Subband CQI均值热力图...\n');

    % 提取扇形区域内用户的subband CQI
    subcqi_sector = subband_CQI_results(:, users_in_sector);  % (num_subbands, num_users_in_sector)

    % 计算每个RB上所有扇形用户subband CQI的均值
    mean_subcqi_per_rb = mean(subcqi_sector, 2);  % (num_subbands, 1)

    % 绘制热力图：X轴为RB(子带)索引，Y轴为CQI值
    % 统计每个RB上每个CQI值的用户数量
    cqi_histogram = zeros(16, num_subbands);
    for rb_idx = 1:num_subbands
        cqi_values_rb = subband_CQI_results(rb_idx, users_in_sector);
        for cqi_val = 0:15
            cqi_histogram(cqi_val+1, rb_idx) = sum(cqi_values_rb == cqi_val);
        end
    end

    figure('Position', [100, 100, 1000, 600]);

    % 子图1：每个RB上CQI分布的直方图/热力图
    subplot(2, 1, 1);
    imagesc(1:num_subbands, 0:15, cqi_histogram);
    colorbar;
    xlabel('RB (Subband) Index');
    ylabel('CQI Value');
    title(sprintf('Subband CQI Distribution per RB (Sector Users, n=%d)', num_users_in_sector));
    colormap(gca, 'jet');
    clim([0, max(cqi_histogram(:))]);
    set(gca, 'YDir', 'normal');

    % 子图2：每个RB上的平均CQI值
    subplot(2, 1, 2);
    bar(1:num_subbands, mean_subcqi_per_rb);
    xlabel('RB (Subband) Index');
    ylabel('Mean Subband CQI');
    title(sprintf('Mean Subband CQI per RB (Sector Users, n=%d)', num_users_in_sector));
    grid on;
    ylim([0, 16]);

    % 添加数值标签
    hold on;
    for rb_idx = 1:num_subbands
        text(rb_idx, mean_subcqi_per_rb(rb_idx) + 0.3, ...
            sprintf('%.2f', mean_subcqi_per_rb(rb_idx)), ...
            'HorizontalAlignment', 'center', 'FontSize', 8);
    end
    hold off;

    sgtitle(sprintf('RB-Level Subband CQI Analysis (Sector: center=%.0f°, width=%.0f°)', ...
        sector_center_deg, sector_width_deg), 'FontSize', 14);

    % 打印统计信息
    fprintf('  扇形区域内用户数量: %d\n', num_users_in_sector);
    fprintf('  子带(RB)数量: %d\n', num_subbands);
    fprintf('  每个RB上的平均CQI:\n');
    for rb_idx = 1:num_subbands
        fprintf('    RB %d: %.2f (范围: [%d, %d])\n', ...
            rb_idx, mean_subcqi_per_rb(rb_idx), ...
            min(subband_CQI_results(rb_idx, users_in_sector)), ...
            max(subband_CQI_results(rb_idx, users_in_sector)));
    end
    fprintf('  整体平均CQI: %.2f\n', mean(mean_subcqi_per_rb));
end

%% ==================== 2D平面图（用户位置按CQI着色）====================
if enable_plots && ~isempty(ue_locs) && any(ue_locs(:) ~= 0)
    fprintf('\n生成2D位置图...\n');
    
    figure('Position', [100, 100, 1000, 800]);
    hold on;
    
    % 如果有环境图像，先显示作为背景
    if ~isempty(env_image_array)
        % 显示环境图像作为背景
        % 注意：ue_locs可能是归一化坐标，需要映射到图像坐标
        
        % 检查图像数据格式
        img_dims = size(env_image_array);
        if length(img_dims) == 2
            % 灰度图像 (m×n)
            img_for_display = env_image_array;
            actual_img_height = img_dims(1);
            actual_img_width = img_dims(2);
        elseif length(img_dims) == 3 && img_dims(3) == 3
            % RGB图像 (m×n×3)
            img_for_display = env_image_array;
            actual_img_height = img_dims(1);
            actual_img_width = img_dims(2);
        elseif length(img_dims) == 3 && img_dims(3) == 4
            % RGBA图像，只取前3个通道
            img_for_display = env_image_array(:, :, 1:3);
            actual_img_height = img_dims(1);
            actual_img_width = img_dims(2);
        else
            warning('环境图像格式不支持，跳过背景显示');
            img_for_display = [];
            actual_img_height = [];
            actual_img_width = [];
        end
        
        if ~isempty(img_for_display)
            % 确定使用哪个尺寸：优先使用环境图像的实际尺寸
            % 同时检查是否需要坐标缩放
            need_coord_scaling = false;
            scale_x = 1.0;
            scale_y = 1.0;
            
            % 确保尺寸变量为双精度
            actual_img_width = double(actual_img_width);
            actual_img_height = double(actual_img_height);
            
            if ~isempty(img_size) && all(img_size > 0)
                % 确保img_size为双精度
                img_size = double(img_size);
                
                % 检查img_size是否与环境图像尺寸匹配
                if abs(img_size(1) - actual_img_width) < 5 && abs(img_size(2) - actual_img_height) < 5
                    % 尺寸匹配，使用img_size
                    display_width = img_size(1);
                    display_height = img_size(2);
                    fprintf('使用Info.npy中的图像尺寸: [%.0f, %.0f]\n', display_width, display_height);
                else
                    % 尺寸不匹配，需要坐标缩放
                    display_width = actual_img_width;
                    display_height = actual_img_height;
                    need_coord_scaling = true;
                    scale_x = double(actual_img_width) / double(img_size(1));
                    scale_y = double(actual_img_height) / double(img_size(2));
                    fprintf('检测到尺寸不匹配:\n');
                    fprintf('  Info.npy尺寸: [%.0f, %.0f]\n', img_size(1), img_size(2));
                    fprintf('  实际图像尺寸: [%.0f, %.0f]\n', actual_img_width, actual_img_height);
                    fprintf('  坐标缩放比例: X=%.4f, Y=%.4f\n', scale_x, scale_y);
                end
            else
                % 没有img_size，使用环境图像的实际尺寸
                display_width = actual_img_width;
                display_height = actual_img_height;
                fprintf('使用环境图像实际尺寸: [%.0f, %.0f]\n', display_width, display_height);
            end
            
            % 判断坐标类型并映射
            % 确保坐标数组为双精度类型
            ue_locs = double(ue_locs);
            bs_loc = double(bs_loc);
            
            if max(ue_locs(:)) <= 1 && min(ue_locs(:)) >= 0
                % 归一化坐标，需要映射到图像尺寸
                if need_coord_scaling
                    % 如果尺寸不匹配，归一化坐标应该映射到实际图像尺寸
                    x_coords = ue_locs(:, 1) * display_width;
                    y_coords = ue_locs(:, 2) * display_height;
                    bs_x = bs_loc(1) * display_width;
                    bs_y = bs_loc(2) * display_height;
                    fprintf('坐标映射: 归一化坐标 -> [0, %.0f] × [0, %.0f] (已缩放)\n', ...
                        display_width, display_height);
                else
                    % 尺寸匹配，直接映射
                    x_coords = ue_locs(:, 1) * display_width;
                    y_coords = ue_locs(:, 2) * display_height;
                    bs_x = bs_loc(1) * display_width;
                    bs_y = bs_loc(2) * display_height;
                    fprintf('坐标映射: 归一化坐标 -> [0, %.0f] × [0, %.0f]\n', ...
                        display_width, display_height);
                end
                
                % 显示图像（注意MATLAB的y轴方向）
                if size(img_for_display, 3) == 3
                    % RGB图像
                    image([0, display_width], [0, display_height], flipud(img_for_display));
                else
                    % 灰度图像
                    image([0, display_width], [0, display_height], flipud(img_for_display), 'CDataMapping', 'scaled');
                    colormap(gca, 'gray');
                end
                axis([0, display_width, 0, display_height]);
            else
                % 已经是实际坐标（基于Info.npy的坐标系）
                if need_coord_scaling
                    % 需要将坐标从Info.npy坐标系缩放到实际图像坐标系
                    x_coords = ue_locs(:, 1) * scale_x;
                    y_coords = ue_locs(:, 2) * scale_y;
                    bs_x = bs_loc(1) * scale_x;
                    bs_y = bs_loc(2) * scale_y;
                    fprintf('坐标映射: Info.npy坐标系 [%.0f, %.0f] -> 实际图像坐标系 [%.0f, %.0f]\n', ...
                        img_size(1), img_size(2), actual_img_width, actual_img_height);
                    fprintf('  BS位置: (%.2f, %.2f) -> (%.2f, %.2f)\n', ...
                        bs_loc(1), bs_loc(2), bs_x, bs_y);
                else
                    % 尺寸匹配，直接使用
                    x_coords = ue_locs(:, 1);
                    y_coords = ue_locs(:, 2);
                    bs_x = bs_loc(1);
                    bs_y = bs_loc(2);
                end
                
                % 计算坐标范围
                x_range = [min(x_coords), max(x_coords)];
                y_range = [min(y_coords), max(y_coords)];
                
                % 如果坐标范围有效，显示图像
                if diff(x_range) > 0 && diff(y_range) > 0
                    % 显示图像（使用实际图像尺寸）
                    if size(img_for_display, 3) == 3
                        % RGB图像
                        image([0, display_width], [0, display_height], flipud(img_for_display));
                    else
                        % 灰度图像
                        image([0, display_width], [0, display_height], flipud(img_for_display), 'CDataMapping', 'scaled');
                        colormap(gca, 'gray');
                    end
                    axis([0, display_width, 0, display_height]);
                    fprintf('坐标范围: X[%.2f, %.2f] Y[%.2f, %.2f]\n', ...
                        x_range(1), x_range(2), y_range(1), y_range(2));
                else
                    warning('坐标范围无效，跳过背景图像显示');
                end
            end
            alpha(0.5);  % 设置透明度，使图像半透明
            fprintf('已添加环境图像作为背景\n');
        end
    end
    
    % 如果没有环境图像或图像显示失败，使用原始坐标
    if ~exist('x_coords', 'var') || isempty(x_coords)
        x_coords = ue_locs(:, 1);
        y_coords = ue_locs(:, 2);
        if ~exist('bs_x', 'var')
            bs_x = bs_loc(1);
            bs_y = bs_loc(2);
        end
    end
    
    % 使用统一的CQI颜色映射（与t-SNE图保持一致）
    % 绘制用户位置（按CQI值着色）
    scatter_handles = gobjects(16, 1);
    for cqi_level = 1:16
        idx = round(wideband_CQI_results) == (cqi_level - 1);
        if any(idx)
            scatter_handles(cqi_level) = scatter(...
                x_coords(idx), y_coords(idx), 30, ...
                'MarkerFaceColor', cqi_colormap(cqi_level,:), ...
                'MarkerEdgeColor', [0.2 0.2 0.2], ...
                'LineWidth', 0.5, ...
                'DisplayName', sprintf('CQI=%d', (cqi_level-1)));
        end
    end
    
    % 标注用户编号，方便调试
    %for user_idx = 1:num_users
        %text(x_coords(user_idx), y_coords(user_idx), ...
            %sprintf(' %d', user_idx), ...
            %'FontSize', 7, 'Color', 'k', 'FontWeight', 'normal');
    %end
    
    % 绘制基站位置（红色五角星）
    scatter(bs_x, bs_y, 400, 'pentagram', 'k', 'filled', ...
        'MarkerFaceColor', [1 0.2 0.2], ...
        'MarkerEdgeColor', 'k', 'LineWidth', 1.5, 'DisplayName', 'Base Station');
    
    % 绘制自定义扇形（如果位置信息有效）
    if num_users_in_sector > 0 && exist('angles_to_users', 'var') && ~isempty(angles_to_users)
        % 计算扇形半径（取用户到基站的最大距离的1.2倍）
        if exist('x_coords', 'var') && exist('y_coords', 'var')
            distances_to_bs = sqrt((x_coords - bs_x).^2 + (y_coords - bs_y).^2);
            max_dist = max(distances_to_bs);
            if max_dist > 0
                sector_radius = max_dist * 1.2;
            else
                sector_radius = 100;  % 默认值
            end
        else
            sector_radius = 100;  % 默认值
        end
        
        % 绘制扇形（中心sector_center_deg，宽度sector_width_deg）
        angle_start = sector_center_deg - sector_half_angle;  % 左边界（度）
        angle_end = sector_center_deg + sector_half_angle;    % 右边界（度）
        
        % 转换为MATLAB坐标系的角度（从x轴正方向逆时针）
        % 映射关系：相对正北顺时针角 angle -> theta = 90 - angle
        theta_start = (90 - angle_start) * pi / 180;
        theta_end = (90 - angle_end) * pi / 180;
        
        % 生成扇形边界
        num_points = 100;
        theta_sector = linspace(theta_start, theta_end, num_points);
        x_sector = bs_x + sector_radius * cos(theta_sector);
        y_sector = bs_y + sector_radius * sin(theta_sector);
        
        % 绘制扇形边界线
        plot([bs_x, x_sector(1)], [bs_y, y_sector(1)], 'k--', 'LineWidth', 1.5, ...
            'DisplayName', 'Sector Boundary');
        plot([bs_x, x_sector(end)], [bs_y, y_sector(end)], 'k--', 'LineWidth', 1.5);
        plot(x_sector, y_sector, 'k--', 'LineWidth', 1.5);
        
        % 添加文本标注
        text(bs_x + sector_radius * 0.5 * cos((90 - angle_start) * pi / 180), ...
             bs_y + sector_radius * 0.5 * sin((90 - angle_start) * pi / 180), ...
             sprintf('%.0f°', angle_start), 'FontSize', 10, 'Color', 'k', 'FontWeight', 'bold');
        text(bs_x + sector_radius * 0.5 * cos((90 - angle_end) * pi / 180), ...
             bs_y + sector_radius * 0.5 * sin((90 - angle_end) * pi / 180), ...
             sprintf('%.0f°', angle_end), 'FontSize', 10, 'Color', 'k', 'FontWeight', 'bold');
        text(bs_x + sector_radius * 0.7 * cos(90 * pi / 180), ...
             bs_y + sector_radius * 0.7 * sin(90 * pi / 180), ...
             '正北', 'FontSize', 11, 'Color', 'k', 'FontWeight', 'bold', ...
             'HorizontalAlignment', 'center');
    end
    
    % 图表修饰
    colormap(cqi_colormap);
    clim([0 15]);
    cbar = colorbar;
    cbar.Label.String = 'CQI Value';
    cbar.Ticks = 0:15;
    cbar.TickLabels = arrayfun(@(x) sprintf('%d',x), 0:15, 'UniformOutput', false);
    
    grid on;
    xlabel('X Coordinate', 'FontSize', 12);
    ylabel('Y Coordinate', 'FontSize', 12);
    title('User Distribution with CQI Coloring', 'FontSize', 14);
    legend('Location', 'eastoutside', 'FontSize', 9);
    axis equal;
    axis tight;
    
    hold off;
end

%% ==================== 2D平面图（用户位置按SNR连续值热力图着色）====================
if enable_plots && enable_snr_heatmap && exist('wideband_avg_SNR_dB', 'var') ...
        && ~isempty(ue_locs) && any(ue_locs(:) ~= 0)
    fprintf('\n生成SNR热力图（2D平面图）...\n');
    
    [img_height, img_width] = size(env_image_array);
    display_height = img_height;
    display_width = img_width;
    
    x_coords = ue_locs(:, 1);
    y_coords = ue_locs(:, 2);
    
    figure('Position', [100, 100, 1000, 800]);
    hold on;
    
    % 绘制用户位置（按SNR连续值着色）
    scatter(x_coords, y_coords, 60, wideband_avg_SNR_dB, 'filled', ...
        'MarkerEdgeColor', [0.2 0.2 0.2], ...
        'LineWidth', 0.5, ...
        'DisplayName', 'User SNR');
    
    % 绘制扇形区域
    if num_users_in_sector > 0 && exist('angles_to_users', 'var') && ~isempty(angles_to_users)
        distances_to_bs = sqrt((x_coords - bs_x).^2 + (y_coords - bs_y).^2);
        max_dist = max(distances_to_bs);
        if max_dist > 0
            sector_radius = max_dist * 1.2;
        else
            sector_radius = 100;
        end
        
        angle_start = sector_center_deg - sector_half_angle;
        angle_end = sector_center_deg + sector_half_angle;
        
        theta_start = (90 - angle_start) * pi / 180;
        theta_end = (90 - angle_end) * pi / 180;
        
        num_points = 100;
        theta_sector = linspace(theta_start, theta_end, num_points);
        x_sector = bs_x + sector_radius * cos(theta_sector);
        y_sector = bs_y + sector_radius * sin(theta_sector);
        
        plot([bs_x, x_sector(1)], [bs_y, y_sector(1)], 'k--', 'LineWidth', 1.5, ...
            'DisplayName', 'Sector Boundary');
        plot([bs_x, x_sector(end)], [bs_y, y_sector(end)], 'k--', 'LineWidth', 1.5);
        plot(x_sector, y_sector, 'k--', 'LineWidth', 1.5);
    end
    
    % 绘制基站位置（红色五角星）
    scatter(bs_x, bs_y, 400, 'pentagram', 'k', 'filled', ...
        'MarkerFaceColor', [1 0.2 0.2], ...
        'MarkerEdgeColor', 'k', 'LineWidth', 1.5, 'DisplayName', 'Base Station');
    
    % 图表修饰
    colormap(jet);
    cbar = colorbar;
    cbar.Label.String = 'SNR (dB)';
    
    grid on;
    xlabel('X Coordinate', 'FontSize', 12);
    ylabel('Y Coordinate', 'FontSize', 12);
    title('User Distribution with SNR Heatmap', 'FontSize', 14);
    legend('Location', 'eastoutside', 'FontSize', 9);
    axis equal;
    axis tight;
    
    hold off;
end

%% ==================== 保存结果 ====================
fprintf('\n保存结果...\n');

% 更新场景结构，补充脚本生成的统计信息
scenario_info.Nt = Nt;
scenario_info.Nr = Nr;
scenario_info.Nt_total = Nt_total;
scenario_info.Nr_total = Nr_total;
scenario_info.sampledCarriers = sampledCarriers;
scenario_info.BWGHz = BWGHz;
scenario_info.subcarrier_spacing = subcarrier_spacing;
scenario_info.num_users = num_users;
scenario_info.num_subbands = num_subbands;
scenario_info.subband_size = subband_size;

fprintf('场景信息:\n');
fprintf('  环境ID: %s\n', env_id);
fprintf('  载波频率: %s\n', carrier_freq);
fprintf('  场景编号: %d\n', scenario_info.scenario);
fprintf('  用户数量: %d\n', num_users);

%% ==================== 准备兼容Python数据加载器的数据格式 ====================
fprintf('\n准备Python兼容的数据格式...\n');


% 重新排列DFT后的H: (num_users, Nt*Nr, sampledCarriers)
H_reshaped = permute(H_complex_angle_delay, [3, 1, 2]);  % 从 (Nt*Nr, sampledCarriers, num_users) 到 (num_users, Nt*Nr, sampledCarriers)

% 分离实部和虚部
H_real = real(H_reshaped);  % (num_users, Nt*Nr, sampledCarriers)
H_imag = imag(H_reshaped);  % (num_users, Nt*Nr, sampledCarriers)

% 将实部和虚部堆叠为4D数组，形状为 (num_users, Nt*Nr, sampledCarriers, 2)
% 最后一个维度: [real, imag]
% 创建一个包含 'real' 和 'imag' 字段的结构体
H_final_angle_delay = struct('real', H_real, 'imag', H_imag);

fprintf('H数据格式转换完成（使用DFT后的角度-延迟域数据）:\n');
fprintf('  DFT后H形状: %s\n', mat2str(size(H_complex_angle_delay)));
fprintf('  转换后类型: struct with fields ''real'' and ''imag''\n');
    fprintf('  H_final_angle_delay.real 形状: %s\n', mat2str(size(H_final_angle_delay.real)));
    fprintf('  H_final_angle_delay.imag 形状: %s\n', mat2str(size(H_final_angle_delay.imag)));


wideband_CQI_results_save = wideband_CQI_results';  % 强制转置为列向量 (num_users, 1)

% subband_CQI_results当前形状: (num_subbands, num_users)
% Python期望: (num_users, num_subbands)，需要转置
if size(subband_CQI_results, 1) == num_subbands && size(subband_CQI_results, 2) == num_users
    subband_CQI_results_save = subband_CQI_results';  % 转置为 (num_users, num_subbands)
else
    subband_CQI_results_save = subband_CQI_results;  % 如果已经是正确形状，直接使用
end

fprintf('CQI数据格式:\n');
fprintf('  Wideband CQI形状: %s (Python期望: (%d,))\n', ...
    mat2str(size(wideband_CQI_results_save)), num_users);
fprintf('  Subband CQI形状: %s (Python期望: (%d, %d))\n', ...
    mat2str(size(subband_CQI_results_save)), num_users, num_subbands);

%% ==================== 保存Python兼容格式的文件 ====================


output_file_wide = sprintf('%s_wide.mat', output_prefix);
output_file_wide_python = sprintf('losangeles_adCSI_right_wide.mat');


wideband_CQI_results = wideband_CQI_results_save;  % 确保是列向量
save(output_file_wide, 'H_final_angle_delay', 'wideband_CQI_results', 'scenario_info', '-v7.3');
fprintf('\n已保存: %s\n', output_file_wide);
fprintf('  包含变量: H_final_angle_delay, wideband_CQI_results, scenario_info\n');


[output_dir, ~, ~] = fileparts(output_file_wide);
if ~isempty(output_dir) && exist(output_dir, 'dir')
    output_file_wide_python_full = fullfile(output_dir, output_file_wide_python);
    save(output_file_wide_python_full, 'H_final_angle_delay', 'wideband_CQI_results', 'scenario_info', '-v7.3');
    fprintf('已保存(Python兼容): %s\n', output_file_wide_python_full);
end


output_file_sub = sprintf('%s_sub.mat', output_prefix);
output_file_sub_python = sprintf('losangeles_adCSI_right_sub.mat');


subband_CQI_results = subband_CQI_results_save;  % 临时赋值以匹配Python期望的变量名
save(output_file_sub, 'H_final_angle_delay', 'subband_CQI_results', 'scenario_info', '-v7.3');
fprintf('已保存: %s\n', output_file_sub);
fprintf('  包含变量: H_final_angle_delay, subband_CQI_results, scenario_info\n');


if ~isempty(output_dir) && exist(output_dir, 'dir')
    output_file_sub_python_full = fullfile(output_dir, output_file_sub_python);
    save(output_file_sub_python_full, 'H_final_angle_delay', 'subband_CQI_results', 'scenario_info', '-v7.3');
    fprintf('已保存(Python兼容): %s\n', output_file_sub_python_full);
end


%% ==================== 保存基站正方向扇区内用户结果 ====================
if num_users_in_sector > 0 && any(users_in_sector)
    fprintf('\n保存扇区内用户的CSI和CQI结果...\n');
    
    % === 重要：为保证加载代码兼容，使用相同变量名保存 ===
    % 1. 先用扇区内用户的数据覆盖内存中的变量
    H_final_angle_delay.real = H_final_angle_delay.real(users_in_sector, :, :);
    H_final_angle_delay.imag = H_final_angle_delay.imag(users_in_sector, :, :);
    wideband_CQI_results = wideband_CQI_results(users_in_sector);
    subband_CQI_results = subband_CQI_results(users_in_sector, :);
    scenario_info.num_users = num_users_in_sector; % 更新场景信息中的用户数
    
    % 2. 定义扇区用户的文件名
    sector_tag = sprintf('sector_w%g_dir%g', sector_width_deg, sector_center_deg);
    sector_tag = strrep(sector_tag, '-', 'm'); % 文件名中替换负号
    output_file_wide_120 = sprintf('%s_wide_%s.mat', output_prefix, sector_tag);
    output_file_sub_120  = sprintf('%s_sub_%s.mat',  output_prefix, sector_tag);
    output_file_wide_120_python = sprintf('losangeles_adCSI_right_wide_%s.mat', sector_tag);
    output_file_sub_120_python  = sprintf('losangeles_adCSI_right_sub_%s.mat',  sector_tag);
    
    % 3. 使用被覆盖后的变量进行保存
    save(output_file_wide_120, 'H_final_angle_delay', 'wideband_CQI_results', 'scenario_info', '-v7.3');
    fprintf('已保存扇区用户(宽带CQI): %s\n', output_file_wide_120);
    fprintf('  包含变量: H_final_angle_delay, wideband_CQI_results, scenario_info\n');
    
    save(output_file_sub_120, 'H_final_angle_delay', 'subband_CQI_results', 'scenario_info', '-v7.3');
    fprintf('已保存扇区用户(子带CQI): %s\n', output_file_sub_120);
    fprintf('  包含变量: H_final_angle_delay, subband_CQI_results, scenario_info\n');
    
    % 4. 保存Python兼容版本
    if ~isempty(output_dir) && exist(output_dir, 'dir')
        output_file_wide_120_python_full = fullfile(output_dir, output_file_wide_120_python);
        save(output_file_wide_120_python_full, 'H_final_angle_delay', 'wideband_CQI_results', 'scenario_info', '-v7.3');
        fprintf('已保存扇区用户(Python兼容, 宽带CQI): %s\n', output_file_wide_120_python_full);
        
        output_file_sub_120_python_full = fullfile(output_dir, output_file_sub_120_python);
        save(output_file_sub_120_python_full, 'H_final_angle_delay', 'subband_CQI_results', 'scenario_info', '-v7.3');
        fprintf('已保存扇区用户(Python兼容, 子带CQI): %s\n', output_file_sub_120_python_full);
    end
else
    fprintf('\n扇区内用户数量为0，跳过针对扇区用户的额外保存。\n');
end

fprintf('\n所有处理完成！\n');
fprintf('\n提示: Python数据加载器期望的文件名为 losangeles_adCSI_right_{wide|sub}.mat\n');
fprintf('      如果文件名不匹配，请将生成的文件重命名或复制到数据目录中。\n');

function plot_tsne_for_users(H_complex, user_mask, cqi_values, antenna_pairs, num_subcarriers, title_str, show_info_box)
if nargin < 7 || isempty(title_str)
    title_str = 't-SNE Visualization';
end
if nargin < 8
    show_info_box = false;
end

mask = logical(user_mask);
subset_idx = find(mask);
if isempty(subset_idx)
    warning('plot_tsne_for_users: 用户筛选结果为空，跳过t-SNE绘制');
    return;
end

num_subset = numel(subset_idx);
num_features = antenna_pairs * num_subcarriers;
H_subset = H_complex(:, :, subset_idx);
H_subset = reshape(H_subset, num_features, num_subset).';
H_real = [real(H_subset), imag(H_subset)];

rng(42);  % 固定随机种子，保证可重复
fprintf('执行t-SNE降维（样本数: %d）...\n', num_subset);
Y = tsne(H_real);

cqi_subset = cqi_values(subset_idx);
% 使用统一的16色颜色映射（与位置分布图保持一致）
colors = hsv(16);

figure('Position', [100, 100, 800, 600]);
gscatter(Y(:,1), Y(:,2), cqi_subset, colors, '.', 10);
title(title_str, 'FontSize', 14);
xlabel('t-SNE Dimension 1', 'FontSize', 12);
ylabel('t-SNE Dimension 2', 'FontSize', 12);
grid on;
legend('Location', 'bestoutside', 'FontSize', 9);

stats_text = sprintf('用户数量: %d\nCQI范围: [%d, %d]\nCQI均值: %.2f', ...
    num_subset, min(cqi_subset), max(cqi_subset), mean(cqi_subset));
if show_info_box
    text(0.02, 0.98, stats_text, 'Units', 'normalized', ...
        'VerticalAlignment', 'top', 'FontSize', 10, ...
        'BackgroundColor', 'white', 'EdgeColor', 'black');
end

fprintf('  %s:\n', title_str);
fprintf('    用户数量: %d\n', num_subset);
fprintf('    CQI范围: [%d, %d]\n', min(cqi_subset), max(cqi_subset));
fprintf('    CQI均值: %.2f\n', mean(cqi_subset));
end

