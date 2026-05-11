import os
import argparse
import datetime as dt
import numpy as np
import h5py


def load_h5_cqi(path, cqi_key):
    assert os.path.isfile(path), f"文件不存在: {path}"
    with h5py.File(path, 'r') as f:
        # H 不直接使用，仅检查样本数一致
        H = f['H_final_angle_delay'][:]
        if cqi_key not in f:
            raise KeyError(f"在 {os.path.basename(path)} 中未找到数据集: {cqi_key}")
        CQI = f[cqi_key][:]
    # 转为 np.float64 便于数值计算
    return H, np.asarray(CQI, dtype=np.float64)


def compute_correlation(x, y):
    # 皮尔逊相关；若方差为0返回nan
    x = np.asarray(x)
    y = np.asarray(y)
    if x.ndim == 1:
        x = x.reshape(-1)
    if y.ndim == 1:
        y = y.reshape(-1)
    if x.size != y.size:
        raise ValueError("x 与 y 长度不一致")
    xv = x - x.mean()
    yv = y - y.mean()
    denom = np.sqrt((xv ** 2).sum()) * np.sqrt((yv ** 2).sum())
    if denom <= 1e-12:
        return np.nan
    return float((xv * yv).sum() / denom)


def main(args):
    data_dir = args.data_dir
    wide_file = os.path.join(data_dir, '/home/wutong/SemCQI/SemCSI_rrn-main/SemCSI_rrn-main/TransCQA/losangeles_data/losangeles_adCSI_right_wide.mat')
    sub_file = os.path.join(data_dir, '/home/wutong/SemCQI/SemCSI_rrn-main/SemCSI_rrn-main/TransCQA/losangeles_data/losangeles_adCSI_right_sub.mat')

    # 读取 raw CQI（未做 Dataset 内部的 Z-score）
    H_wide, cqi_wide = load_h5_cqi(wide_file, 'wideband_CQI_results')
    H_sub, cqi_sub = load_h5_cqi(sub_file, 'subband_CQI_results')

    # 形状检查
    n_wide = cqi_wide.shape[0]
    n_sub, s_sub = cqi_sub.shape[0], (cqi_sub.shape[1] if cqi_sub.ndim > 1 else 1)
    assert n_wide == n_sub, f"样本数不一致: wide={n_wide}, sub={n_sub}"
    assert s_sub == 8, f"预期 sub 维度=8，实际={s_sub}"

    # 如果 H 文件不同，检查维度；一般两文件的 H 相同或等价
    assert H_wide.shape[0] == H_sub.shape[0] == n_wide, "H 与 CQI 样本数不一致"

    # 统计量
    # 将 wide 从 (N,) 或 (N,1) 变为 (N,)
    cqi_wide = cqi_wide.reshape(-1)
    # sub 保持 (N,8)

    # 与 sub 的汇总对比（原始/未归一化）
    sub_mean = np.mean(cqi_sub, axis=1)         # (N,)
    sub_std = np.std(cqi_sub, axis=1)           # (N,)
    sub_range = np.ptp(cqi_sub, axis=1)         # (N,)

    # 相关性（原始）：wide vs sub_mean；wide vs 每个 sub_i
    corr_wide_sub_mean = compute_correlation(cqi_wide, sub_mean)
    corr_wide_sub_each = [compute_correlation(cqi_wide, cqi_sub[:, i]) for i in range(8)]

    # 判定“sub 是否有必要”的一些启发式指标（原始）
    q = [5, 25, 50, 75, 95]
    std_quantiles = np.percentile(sub_std, q)
    range_quantiles = np.percentile(sub_range, q)
    range_thr = args.range_threshold
    std_thr = args.std_threshold
    frac_range = float((sub_range > range_thr).mean())
    frac_std = float((sub_std > std_thr).mean())
    diff = np.abs(cqi_wide - sub_mean)
    diff_quantiles = np.percentile(diff, q)

    # ===== 归一化版本（按 Dataset 的 Z-score 策略）=====
    # wide: 全局标量均值方差；sub: 各维度均值方差
    wide_mu = float(np.mean(cqi_wide))
    wide_sigma = float(np.std(cqi_wide))
    sub_mu = np.mean(cqi_sub, axis=0, keepdims=True)      # (1,8)
    sub_sigma = np.std(cqi_sub, axis=0, keepdims=True)    # (1,8)
    eps = 1e-8

    cqi_wide_norm = (cqi_wide - wide_mu) / (wide_sigma + eps)           # (N,)
    cqi_sub_norm = (cqi_sub - sub_mu) / (sub_sigma + eps)               # (N,8)

    sub_mean_norm = np.mean(cqi_sub_norm, axis=1)         # (N,)
    sub_std_norm = np.std(cqi_sub_norm, axis=1)           # (N,)
    sub_range_norm = np.ptp(cqi_sub_norm, axis=1)         # (N,)

    corr_wide_sub_mean_norm = compute_correlation(cqi_wide_norm, sub_mean_norm)
    corr_wide_sub_each_norm = [compute_correlation(cqi_wide_norm, cqi_sub_norm[:, i]) for i in range(8)]

    std_quantiles_norm = np.percentile(sub_std_norm, q)
    range_quantiles_norm = np.percentile(sub_range_norm, q)
    frac_range_norm = float((sub_range_norm > range_thr).mean())
    frac_std_norm = float((sub_std_norm > std_thr).mean())
    diff_norm = np.abs(cqi_wide_norm - sub_mean_norm)
    diff_quantiles_norm = np.percentile(diff_norm, q)

    # 保存 CSV（可选）
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, 'cqi_wide_vs_sub_stats.csv')
    with open(csv_path, 'w', encoding='utf-8') as f:
        # 原始（未归一化）
        f.write('== RAW (UNNORMALIZED) ==\n')
        f.write('N,wide_mean,wide_std,sub_mean_of_mean,sub_mean_std\n')
        f.write(f"{n_wide},{float(cqi_wide.mean()):.6f},{float(cqi_wide.std()):.6f},"
                f"{float(sub_mean.mean()):.6f},{float(sub_mean.std()):.6f}\n")
        f.write('\n# quantiles (%, sub_std): ' + ','.join(map(str, q)) + '\n')
        f.write(','.join([f"{v:.6f}" for v in std_quantiles]) + '\n')
        f.write('\n# quantiles (%, sub_range): ' + ','.join(map(str, q)) + '\n')
        f.write(','.join([f"{v:.6f}" for v in range_quantiles]) + '\n')
        f.write('\n# corr(wide, sub_mean)\n')
        f.write(f"{corr_wide_sub_mean:.6f}\n")
        f.write('\n# corr(wide, sub_i) i=0..7\n')
        f.write(','.join([f"{c if not np.isnan(c) else 'nan'}" for c in corr_wide_sub_each]) + '\n')
        f.write('\n# fraction(sub_range > thr), thr=' + str(range_thr) + '\n')
        f.write(f"{frac_range:.6f}\n")
        f.write('\n# fraction(sub_std > thr), thr=' + str(std_thr) + '\n')
        f.write(f"{frac_std:.6f}\n")
        f.write('\n# quantiles (%, |wide - mean(sub)|): ' + ','.join(map(str, q)) + '\n')
        f.write(','.join([f"{v:.6f}" for v in diff_quantiles]) + '\n')

        # 归一化（Z-score）
        f.write('\n== NORMALIZED (Z-SCORE) ==\n')
        f.write('wide_mu,wide_sigma\n')
        f.write(f"{wide_mu:.6f},{wide_sigma:.6f}\n")
        f.write('\n# quantiles (%, sub_std_norm): ' + ','.join(map(str, q)) + '\n')
        f.write(','.join([f"{v:.6f}" for v in std_quantiles_norm]) + '\n')
        f.write('\n# quantiles (%, sub_range_norm): ' + ','.join(map(str, q)) + '\n')
        f.write(','.join([f"{v:.6f}" for v in range_quantiles_norm]) + '\n')
        f.write('\n# corr(wide_norm, mean(sub_norm))\n')
        f.write(f"{corr_wide_sub_mean_norm:.6f}\n")
        f.write('\n# corr(wide_norm, sub_norm_i) i=0..7\n')
        f.write(','.join([f"{c if not np.isnan(c) else 'nan'}" for c in corr_wide_sub_each_norm]) + '\n')
        f.write('\n# fraction(sub_range_norm > thr), thr=' + str(range_thr) + '\n')
        f.write(f"{frac_range_norm:.6f}\n")
        f.write('\n# fraction(sub_std_norm > thr), thr=' + str(std_thr) + '\n')
        f.write(f"{frac_std_norm:.6f}\n")
        f.write('\n# quantiles (%, |wide_norm - mean(sub_norm)|): ' + ','.join(map(str, q)) + '\n')
        f.write(','.join([f"{v:.6f}" for v in diff_quantiles_norm]) + '\n')

    # 输出简要报告（md）
    ts = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    md_path = os.path.join(args.output_dir, f'cqi_wide_vs_sub_report_{ts}.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('## CQI wide vs sub 对比报告\n\n')
        f.write(f"- 数据目录: `{data_dir}`\n")
        f.write(f"- 样本数: {n_wide}\n\n")
        f.write('### 相关性（未归一化）\n')
        f.write(f"- corr(wide, mean(sub)): **{corr_wide_sub_mean:.6f}**\n")
        f.write(f"- corr(wide, sub_i): {', '.join([f'{c:.6f}' if not np.isnan(c) else 'nan' for c in corr_wide_sub_each])}\n\n")
        f.write('### 子带内部差异（未归一化 std / range 分位数）\n')
        f.write(f"- std 分位数(5/25/50/75/95): {', '.join([f'{v:.6f}' for v in std_quantiles])}\n")
        f.write(f"- range 分位数(5/25/50/75/95): {', '.join([f'{v:.6f}' for v in range_quantiles])}\n\n")
        f.write('### 显著差异样本占比（未归一化）\n')
        f.write(f"- P(sub_range > {range_thr}) = **{frac_range:.6f}**\n")
        f.write(f"- P(sub_std   > {std_thr})   = **{frac_std:.6f}**\n\n")
        f.write('### 宽带与子带均值差异（未归一化）\n')
        f.write(f"- |wide - mean(sub)| 分位数(5/25/50/75/95): {', '.join([f'{v:.6f}' for v in diff_quantiles])}\n\n")

        # 归一化部分
        f.write('### 相关性（Z-score 归一化）\n')
        f.write(f"- corr(wide_norm, mean(sub_norm)): **{corr_wide_sub_mean_norm:.6f}**\n")
        f.write(f"- corr(wide_norm, sub_norm_i): {', '.join([f'{c:.6f}' if not np.isnan(c) else 'nan' for c in corr_wide_sub_each_norm])}\n\n")
        f.write('### 子带内部差异（归一化 std / range 分位数）\n')
        f.write(f"- std_norm 分位数(5/25/50/75/95): {', '.join([f'{v:.6f}' for v in std_quantiles_norm])}\n")
        f.write(f"- range_norm 分位数(5/25/50/75/95): {', '.join([f'{v:.6f}' for v in range_quantiles_norm])}\n\n")
        f.write('### 显著差异样本占比（归一化）\n')
        f.write(f"- P(sub_range_norm > {range_thr}) = **{frac_range_norm:.6f}**\n")
        f.write(f"- P(sub_std_norm   > {std_thr})   = **{frac_std_norm:.6f}**\n\n")
        f.write('### 宽带与子带均值差异（归一化）\n')
        f.write(f"- |wide_norm - mean(sub_norm)| 分位数(5/25/50/75/95): {', '.join([f'{v:.6f}' for v in diff_quantiles_norm])}\n\n")
        f.write('> 解释建议：\n')
        f.write('- 若 corr(wide, mean(sub)) 逼近 1 且 sub_range/std 很小，说明宽带与子带近似等价，引入 sub 的收益可能有限。\n')
        f.write('- 若 corr 较低且 sub_range/std 有明显质量差异，说明子带信息具有额外价值，使用 sub 融合更有必要。\n')
        f.write('- 归一化结果更贴近训练时 Dataset 的处理口径，建议以归一化结论为主，原始结果为辅。\n')

    print(f"CSV 保存: {csv_path}")
    print(f"报告保存: {md_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze CQI wide vs sub differences')
    parser.add_argument('--data-dir', type=str, default='./losangeles_data', help='数据目录')
    parser.add_argument('--output-dir', type=str, default='./docs', help='输出目录')
    parser.add_argument('--range-threshold', type=float, default=0.5, help='判定子带差异的 range 阈值')
    parser.add_argument('--std-threshold', type=float, default=0.2, help='判定子带差异的 std 阈值')
    args = parser.parse_args()
    main(args)


