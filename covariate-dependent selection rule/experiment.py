# File: decision_permutation/experiment.py
import numpy as np
import random
from tqdm import trange
from typing import Tuple
from config import M, alpha, method
from data import generate_data
from selection import selection_rule_covariate_dependent
from interval import construct_prediction_interval
import matplotlib.pyplot as plt

def is_covered(y: float, interval: Tuple[float, float]) -> bool:
    lower, upper = interval
    return lower <= y <=upper

def run_randomized_quantile_interval_experiment(
    n_offline: int,
    n_online: int,
    M: int = M,
    alpha: float = alpha,
    quantile_method: str = "upper",
    seed: int = 0
) -> Tuple[list, int]:
    X_on, Y_on, mu_on = generate_data(n_online, seed)
    t_records = []  # (t, is_covered, length, is_inf, cal_size)
    selected = 0
    for t in range(n_online):
        sel = selection_rule_covariate_dependent(mu_on[t], mu_on[:t].tolist(),method=method)
        if not sel:
            continue
        selected += 1
        interval = construct_prediction_interval(t, X_on, Y_on, mu_on, M, alpha, quantile_method=quantile_method)
        iscov = is_covered(Y_on[t], interval[:2])
        lower, upper = interval[:2] #interval这个tuple的前两个元素是lower和upper
        length = upper - lower
        is_inf = not np.isfinite(length)
        cal_size = interval[2] #interval这个tuple的第三个元素是cal_size
        t_records.append((t, iscov, length, is_inf, cal_size))
    return t_records, selected

def main():
    import matplotlib.pyplot as plt
    import datetime
    import os
    
    # 要测试的不同行数列表
    n_online_list = [200]#[1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 75,100,125,150,175,200]
    # Monte Carlo 次数不变
    n_runs = 1000

    # 生成带时间戳的文件名
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # 在当前脚本所在目录下创建 experiment_results 文件夹
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "experiment_results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    filename = os.path.join(results_dir, f"results_{timestamp}.txt")

    # 用来存放各 n_online 的汇总结果
    summary_upper = []
    summary_randomize = []
    all_t_records_upper = []
    all_t_records_randomize = []

    # 创建文件并写入实验参数
    with open(filename, 'w') as f:
        f.write(f"Experiment Results - {timestamp}\n")
        f.write("="*50 + "\n")
        f.write(f"n_online_list: {n_online_list}\n")
        f.write(f"n_runs: {n_runs}\n")
        f.write(f"alpha: {alpha}\n")
        f.write(f"M: {M}\n")
        f.write("="*50 + "\n\n")

    # 分别运行两种方法
    for quantile_method in ["upper", "randomize"]:
        print(f"\nRunning experiments for {quantile_method} quantile method...")
        #summary = summary_upper if method == "upper" else summary_randomize
        #all_t_records_method = []
        
        for n_online in n_online_list:
            all_t_records = []  # 存储所有 simulation 的 (t, is_covered, length, is_inf, cal_size)
            selected_counts = []
            for seed in trange(n_runs, desc=f"n_online={n_online}"):
                np.random.seed(seed + 2)
                random.seed(seed + 2)
                t_records, selected = run_randomized_quantile_interval_experiment(
                    n_offline=0,
                    n_online=n_online,
                    quantile_method=quantile_method,
                    seed=seed
                )
                all_t_records.extend(t_records)
                selected_counts.append(selected)
            #all_t_records_method.extend(all_t_records)
            
            # 统计每个 t 的 selection conditional 指标
            t_to_cov = {}
            t_to_cov_count = {}
            t_to_len = {}
            t_to_len_count = {}
            t_to_inf = {}
            t_to_inf_count = {}
            t_to_cal = {}
            t_to_cal_count = {}
            for t, iscov, length, is_inf, cal_size in all_t_records:
                # coverage
                t_to_cov.setdefault(t, 0)
                t_to_cov_count.setdefault(t, 0)
                t_to_cov[t] += iscov
                t_to_cov_count[t] += 1
                # length
                t_to_len.setdefault(t, 0.0)
                t_to_len_count.setdefault(t, 0)
                if not np.isnan(length):
                    t_to_len[t] += length
                    t_to_len_count[t] += 1
                # inf
                t_to_inf.setdefault(t, 0)
                t_to_inf_count.setdefault(t, 0)
                t_to_inf[t] += is_inf
                t_to_inf_count[t] += 1
                # cal_size
                t_to_cal.setdefault(t, 0.0)
                t_to_cal_count.setdefault(t, 0)
                t_to_cal[t] += cal_size
                t_to_cal_count[t] += 1
            t_vals = sorted(t_to_cov.keys())
            t_covs = [t_to_cov[t]/t_to_cov_count[t] for t in t_vals]
            t_lens = [t_to_len[t]/t_to_len_count[t] if t_to_len_count[t]>0 else np.nan for t in t_vals]
            t_inf_fracs = [t_to_inf[t]/t_to_inf_count[t] for t in t_vals]
            t_cals = [t_to_cal[t]/t_to_cal_count[t] for t in t_vals]
            avg_sel = np.nanmean(selected_counts)
            # 记录 t-coverage 曲线和 summary
            summary_upper.append((n_online, t_vals, t_covs, t_lens, t_inf_fracs, t_cals, avg_sel)) if quantile_method == "upper" else summary_randomize.append((n_online, t_vals, t_covs, t_lens, t_inf_fracs, t_cals, avg_sel))
        
        if quantile_method == "upper":
            all_t_records_upper = all_t_records
        else:
            all_t_records_randomize = all_t_records

    # 针对每个 t 输出详细表格（mean/std 按 simulation 聚合）
    print("\nPer-t selection conditional statistics (mean/std across simulations):")
    header = ("Method", "t", "Count", "CovMean", "CovStd", "LenMedian", "LenStd", "InfFracMean", "InfFracStd", "CalMean", "CalStd")
    header_str = "{:>10} | {:>4} | {:>6} | {:>7} | {:>7} | {:>7} | {:>7} | {:>11} | {:>11} | {:>8} | {:>8}".format(*header)
    print(header_str)
    print("-"*102)
    
    # 将表格结果写入文件
    with open(filename, 'a') as f:
        f.write("Per-t selection conditional statistics (mean/std across simulations):\n")
        f.write(header_str + "\n")
        f.write("-"*102 + "\n")
    
    # 分别处理两种方法的数据
    for method_name, all_t_records_method in [("Upper", all_t_records_upper), ("Randomize", all_t_records_randomize)]:
        # 只输出第一个 n_online 的详细 t 表格
        n_online, t_vals, *_ = summary_upper[0] if method_name == "Upper" else summary_randomize[0]
        # 需要聚合每个 t 的所有 simulation 的原始数据
        t_cov_list = {t: [] for t in t_vals}
        t_len_list = {t: [] for t in t_vals}
        t_inf_list = {t: [] for t in t_vals}
        t_cal_list = {t: [] for t in t_vals}
        for t, iscov, length, is_inf, cal_size in all_t_records_method:
            if t in t_cov_list:
                t_cov_list[t].append(iscov)
                t_len_list[t].append(length)
                t_inf_list[t].append(is_inf)
                t_cal_list[t].append(cal_size)

        for t in t_vals:
            cov_arr = np.array(t_cov_list[t])
            len_arr = np.array(t_len_list[t])
            inf_arr = np.array(t_inf_list[t])
            cal_arr = np.array(t_cal_list[t])
            count = len(cov_arr)
            if count == 0:
                row_str = f"{method_name:>10} | {t:4d} | {count:6d} | {'-':>7} | {'-':>7} | {'-':>7} | {'-':>7} | {'-':>11} | {'-':>11} | {'-':>8} | {'-':>8}"
            else:
                with np.errstate(invalid='ignore'):
                    row_str = f"{method_name:>10} | {t:4d} | {count:6d} | {np.nanmean(cov_arr):7.3f} | {np.nanstd(cov_arr):7.3f} | {np.nanmedian(len_arr):7.3f} | {np.nanstd(len_arr):7.3f} | {np.nanmean(inf_arr):11.3f} | {np.nanstd(inf_arr):11.3f} | {np.nanmean(cal_arr):8.3f} | {np.nanstd(cal_arr):8.3f}"
            print(row_str)
            with open(filename, 'a') as f:
                f.write(row_str + "\n")

    print(f"\nResults saved to: {filename}")

    # 绘制 selection conditional coverage/length/inf/cal_size 曲线在同一张图的四个子图中
    fig, axes = plt.subplots(2, 2, figsize=(15, 8))
    metrics = [
        (2, "Coverage", axes[0, 0]),
        (3, "Median Length", axes[0, 1]),
        (4, "Inf Fraction", axes[1, 0]),
        (5, "Mean Cal Size", axes[1, 1]),
    ]
    
    # 为每种方法准备画图数据
    method_data = []
    for method_name, all_t_records_method in [("Upper", all_t_records_upper), ("Randomize", all_t_records_randomize)]:
        n_online, t_vals, *_ = summary_upper[0] if method_name == "Upper" else summary_randomize[0]
        t_cov_list = {t: [] for t in t_vals}
        t_len_list = {t: [] for t in t_vals}
        t_inf_list = {t: [] for t in t_vals}
        t_cal_list = {t: [] for t in t_vals}
        for t, iscov, length, is_inf, cal_size in all_t_records_method:
            if t in t_cov_list:
                t_cov_list[t].append(iscov)
                t_len_list[t].append(length)
                t_inf_list[t].append(is_inf)
                t_cal_list[t].append(cal_size)
        method_data.append((method_name, t_vals, t_cov_list, t_len_list, t_inf_list, t_cal_list))
    
    for metric, label, ax in metrics:
        for method_name, t_vals, t_cov_list, t_len_list, t_inf_list, t_cal_list in method_data:
            if metric == 2:
                vals = [np.nanmean(t_cov_list[t]) for t in t_vals]
            elif metric == 3:
                vals = [np.nanmedian(t_len_list[t]) for t in t_vals]
            elif metric == 4:
                vals = [np.nanmean(t_inf_list[t]) for t in t_vals]
            elif metric == 5:
                vals = [np.nanmean(t_cal_list[t]) for t in t_vals]
            
            color = 'tab:blue' if method_name == "Upper" else 'tab:orange'
            marker = 'o' 
            linestyle = '-'if method_name == "Upper" else '--'
            ax.plot(
                t_vals, vals,
                marker=marker,
                linestyle=linestyle,
                linewidth=2,        # 加粗线条
                markersize=4,       # 放大标记
                alpha=0.8,          # 半透明，重叠时也能看见下层
                label=f"{method_name} (T={n_online})",
                color=color)
            #ax.plot(t_vals, vals, marker=marker, label=f"{method_name} (T={n_online})", color=color)
        
        ax.set_xlabel("t")
        ax.set_ylabel(label)
        ax.set_title(f"Selection Conditional {label} vs t")
        ax.legend()
        ax.grid(True)
    plt.tight_layout()
    
    # 保存图片
    plot_filename = os.path.join(results_dir, f"plot_{timestamp}.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_filename}")
    # plt.show()

if __name__ == "__main__":
    main()



# def main():
#     n_runs = 1000
#     summary = []
#     n_online_list = [150]#[1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 75,100,125,150,175,200]
#     for n_online in n_online_list:
#         all_t_records = []  # 存储所有 simulation 的 (t, is_covered, length, is_inf, cal_size)
#         selected_counts = []
#         for seed in trange(n_runs, desc=f"n_online={n_online}"):
#             np.random.seed(seed + 2)
#             random.seed(seed + 2)
#             t_records, selected = run_randomized_quantile_interval_experiment(0, n_online, seed=seed)
#             all_t_records.extend(t_records)
#             selected_counts.append(selected)
#         # 统计每个 t 的 selection conditional 指标
#         t_to_cov = {}
#         t_to_cov_count = {}
#         t_to_len = {}
#         t_to_len_count = {}
#         t_to_inf = {}
#         t_to_inf_count = {}
#         t_to_cal = {}
#         t_to_cal_count = {}
#         for t, iscov, length, is_inf, cal_size in all_t_records:
#             # coverage
#             t_to_cov.setdefault(t, 0)
#             t_to_cov_count.setdefault(t, 0)
#             t_to_cov[t] += iscov
#             t_to_cov_count[t] += 1
#             # length
#             t_to_len.setdefault(t, 0.0)
#             t_to_len_count.setdefault(t, 0)
#             if not np.isnan(length):
#                 t_to_len[t] += length
#                 t_to_len_count[t] += 1
#             # inf
#             t_to_inf.setdefault(t, 0)
#             t_to_inf_count.setdefault(t, 0)
#             t_to_inf[t] += is_inf
#             t_to_inf_count[t] += 1
#             # cal_size
#             t_to_cal.setdefault(t, 0.0)
#             t_to_cal_count.setdefault(t, 0)
#             t_to_cal[t] += cal_size
#             t_to_cal_count[t] += 1
#         t_vals = sorted(t_to_cov.keys()) #t_vals是所有t的list，按从小到大排序
#         t_covs = [t_to_cov[t]/t_to_cov_count[t] for t in t_vals]
#         t_lens = [t_to_len[t]/t_to_len_count[t] if t_to_len_count[t]>0 else np.nan for t in t_vals]
#         t_inf_fracs = [t_to_inf[t]/t_to_inf_count[t] for t in t_vals]
#         t_cals = [t_to_cal[t]/t_to_cal_count[t] for t in t_vals]
#         avg_sel = np.nanmean(selected_counts)
#         summary.append((n_online, t_vals, t_covs, t_lens, t_inf_fracs, t_cals, avg_sel))

#     # 打印表格
#     # 针对每个 t 输出详细表格（mean/std 按 simulation 聚合）
#     print("\nPer-t selection conditional statistics (mean/std across simulations):")
#     header = ("t", "Count", "CovMean", "CovStd", "LenMedian", "LenStd", "InfFracMean", "InfFracStd", "CalMean", "CalStd")
#     print("{:>4} | {:>6} | {:>7} | {:>7} | {:>7} | {:>7} | {:>11} | {:>11} | {:>8} | {:>8}".format(*header))
#     print("-"*90)
#     # 只输出第一个 n_online 的详细 t 表格（一般就跑一个 n_online，大的n_online可以完全覆盖小的n_online的结果）
#     n_online, t_vals, *_ = summary[0]
#     # 需要聚合每个 t 的所有 simulation 的原始数据
#     # 先构建 t -> list 的映射
#     t_cov_list = {t: [] for t in t_vals}
#     t_len_list = {t: [] for t in t_vals}
#     t_inf_list = {t: [] for t in t_vals}
#     t_cal_list = {t: [] for t in t_vals}
#     for t, iscov, length, is_inf, cal_size in all_t_records:
#         if t in t_cov_list:
#             t_cov_list[t].append(iscov)
#             t_len_list[t].append(length)
#             t_inf_list[t].append(is_inf)
#             t_cal_list[t].append(cal_size)

#     for t in t_vals:
#         cov_arr = np.array(t_cov_list[t])
#         len_arr = np.array(t_len_list[t])
#         inf_arr = np.array(t_inf_list[t])
#         cal_arr = np.array(t_cal_list[t])
#         count = len(cov_arr) #数一下每个t被select的数量
#         with np.errstate(invalid='ignore'): #忽略inf的warning
#             print(f"{t:4d} | {count:6d} | {np.nanmean(cov_arr):7.3f} | {np.nanstd(cov_arr):7.3f} | {np.nanmedian(len_arr):7.3f} | {np.nanstd(len_arr):7.3f} | {np.nanmean(inf_arr):11.3f} | {np.nanstd(inf_arr):11.3f} | {np.nanmean(cal_arr):8.3f} | {np.nanstd(cal_arr):8.3f}")

#     # 绘制 selection conditional coverage/length/inf/cal_size 曲线在同一张图的四个子图中
#     fig, axes = plt.subplots(2, 2, figsize=(15, 8))
#     metrics = [
#         (2, "Coverage", axes[0, 0]),
#         (3, "Median Length", axes[0, 1]),
#         (4, "Inf Fraction", axes[1, 0]),
#         (5, "Mean Cal Size", axes[1, 1]),
#     ]
#     for metric, label, ax in metrics:
#         if metric == 2:
#             vals = [np.nanmean(t_cov_list[t]) for t in t_vals]
#         elif metric == 3:
#             vals = [np.nanmedian(t_len_list[t]) for t in t_vals]
#         elif metric == 4:
#             vals = [np.nanmean(t_inf_list[t]) for t in t_vals]
#         elif metric == 5:
#             vals = [np.nanmean(t_cal_list[t]) for t in t_vals]
#         ax.plot(t_vals, vals, marker='o', label=f"T={n_online}")
#         ax.set_xlabel("t")
#         ax.set_ylabel(label)
#         ax.set_title(f"Selection Conditional {label} vs t")
#         ax.legend()
#         ax.grid(True)
#     plt.tight_layout()
#     plt.show()

# if __name__ == "__main__":
#     main()