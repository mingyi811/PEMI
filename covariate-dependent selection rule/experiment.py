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
        interval = construct_prediction_interval(t, X_on, Y_on, mu_on, M, alpha)
        iscov = is_covered(Y_on[t], interval[:2])
        lower, upper = interval[:2] #interval这个tuple的前两个元素是lower和upper
        length = upper - lower
        is_inf = not np.isfinite(length)
        cal_size = interval[2] #interval这个tuple的第三个元素是cal_size
        t_records.append((t, iscov, length, is_inf, cal_size))
    return t_records, selected

def main():
    n_runs = 1000
    summary = []
    n_online_list = [200]#[1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 75,100,125,150,175,200]
    for n_online in n_online_list:
        all_t_records = []  # 存储所有 simulation 的 (t, is_covered, length, is_inf, cal_size)
        selected_counts = []
        for seed in trange(n_runs, desc=f"n_online={n_online}"):
            np.random.seed(seed + 2)
            random.seed(seed + 2)
            t_records, selected = run_randomized_quantile_interval_experiment(0, n_online, seed=seed)
            all_t_records.extend(t_records)
            selected_counts.append(selected)
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
        t_vals = sorted(t_to_cov.keys()) #t_vals是所有t的list，按从小到大排序
        t_covs = [t_to_cov[t]/t_to_cov_count[t] for t in t_vals]
        t_lens = [t_to_len[t]/t_to_len_count[t] if t_to_len_count[t]>0 else np.nan for t in t_vals]
        t_inf_fracs = [t_to_inf[t]/t_to_inf_count[t] for t in t_vals]
        t_cals = [t_to_cal[t]/t_to_cal_count[t] for t in t_vals]
        avg_sel = np.nanmean(selected_counts)
        summary.append((n_online, t_vals, t_covs, t_lens, t_inf_fracs, t_cals, avg_sel))

    # 打印表格
    # 针对每个 t 输出详细表格（mean/std 按 simulation 聚合）
    print("\nPer-t selection conditional statistics (mean/std across simulations):")
    header = ("t", "Count", "CovMean", "CovStd", "LenMedian", "LenStd", "InfFracMean", "InfFracStd", "CalMean", "CalStd")
    print("{:>4} | {:>6} | {:>7} | {:>7} | {:>7} | {:>7} | {:>11} | {:>11} | {:>8} | {:>8}".format(*header))
    print("-"*90)
    # 只输出第一个 n_online 的详细 t 表格（一般就跑一个 n_online，大的n_online可以完全覆盖小的n_online的结果）
    n_online, t_vals, *_ = summary[0]
    # 需要聚合每个 t 的所有 simulation 的原始数据
    # 先构建 t -> list 的映射
    t_cov_list = {t: [] for t in t_vals}
    t_len_list = {t: [] for t in t_vals}
    t_inf_list = {t: [] for t in t_vals}
    t_cal_list = {t: [] for t in t_vals}
    for t, iscov, length, is_inf, cal_size in all_t_records:
        if t in t_cov_list:
            t_cov_list[t].append(iscov)
            t_len_list[t].append(length)
            t_inf_list[t].append(is_inf)
            t_cal_list[t].append(cal_size)

        count = len(t_cov_list) #数一下每个t被select的数量
        with np.errstate(invalid='ignore'): #忽略inf的warning
            print(f"{t:4d} | {count:6d} | {np.nanmean(t_cov_list[t]):7.3f} | {np.nanstd(t_cov_list[t]):7.3f} | {np.nanmedian(t_len_list[t]):7.3f} | {np.nanstd(t_len_list[t]):7.3f} | {np.nanmean(t_inf_list[t]):11.3f} | {np.nanstd(t_inf_list[t]):11.3f} | {np.nanmean(t_cal_list[t]):8.3f} | {np.nanstd(t_cal_list[t]):8.3f}")

    # 绘制 selection conditional coverage/length/inf/cal_size 曲线在同一张图的四个子图中
    fig, axes = plt.subplots(2, 2, figsize=(15, 8))
    metrics = [
        (2, "Coverage", axes[0, 0]),
        (3, "Median Length", axes[0, 1]),
        (4, "Inf Fraction", axes[1, 0]),
        (5, "Mean Cal Size", axes[1, 1]),
    ]
    for metric, label, ax in metrics:
        if metric == 2:
            vals = [np.nanmean(t_cov_list[t]) for t in t_vals]
        elif metric == 3:
            vals = [np.nanmedian(t_len_list[t]) for t in t_vals]
        elif metric == 4:
            vals = [np.nanmean(t_inf_list[t]) for t in t_vals]
        elif metric == 5:
            vals = [np.nanmean(t_cal_list[t]) for t in t_vals]
        ax.plot(t_vals, vals, marker='o', label=f"T={n_online}")
        ax.set_xlabel("t")
        ax.set_ylabel(label)
        ax.set_title(f"Selection Conditional {label} vs t")
        ax.legend()
        ax.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()