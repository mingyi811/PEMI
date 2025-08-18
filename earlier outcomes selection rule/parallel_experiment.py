# File: decision_permutation/experiment.py
# ===== 放在最顶部，且在 import numpy 之前，避免线程竞争 =====
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# ===== 正式导入 =====
import numpy as np
import random
from typing import Tuple, List
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import csv

from config import M, alpha, method, q
from data import generate_data
from selection import selection_rule_earlier_outcomes
from interval import construct_prediction_interval
import matplotlib.pyplot as plt


def is_covered(y: float, interval: List[Tuple[float, float]]) -> bool:
    for interval_tuple in interval:
        lower, upper = interval_tuple
        if lower <= y <=upper:
            return True
    return False


def run_randomized_quantile_interval_experiment(
    n_offline: int,
    n_online: int,
    M: int = M,
    alpha: float = alpha,
    quantile_method: str = "upper",
    seed: int = 0
) -> Tuple[List[Tuple[int, bool, float, bool, float]], int]:
    """
    返回：
      - t_records: List[(t, is_covered, length, is_inf, cal_size)]
      - selected: 该 run 被选中的 t 的个数
    """
    X_on, Y_on, mu_on = generate_data(n_online, seed)
    X_off, Y_off, mu_off = generate_data(n_offline, seed)
    t_records = []  # (t, is_covered, length, is_inf, cal_size)
    selected = 0
    for t in range(n_online):
        sel = selection_rule_earlier_outcomes(mu_on[t], Y_on[:t].tolist(),method=method,q=q)
        if not sel:
            continue
        selected += 1
        interval, cal_size = construct_prediction_interval(t, X_on, Y_on, mu_on, X_off, Y_off, mu_off, M, alpha, quantile_method=quantile_method,q=q,method=method)
        iscov = is_covered(Y_on[t], interval)
        #计算length
        length = 0
        for interval_tuple in interval:
            lower, upper = interval_tuple
            length += upper - lower
        is_inf = not np.isfinite(length)
        t_records.append((t, iscov, length, is_inf, cal_size))
    return t_records, selected


# ---------------- 并行 worker：跑单个 seed 的一次实验 ----------------
def _worker_one_run(run_id: int, n_online: int, n_offline: int, quantile_method: str, base_seed: int):
    """
    返回 (run_id, selected, t_records)
    t_records: List[(t, iscov, length, is_inf, cal_size)]
    """
    # 与你原代码一致的 RNG 设定（可复现）
    np.random.seed(base_seed + 2)
    random.seed(base_seed + 2)

    t_records, selected = run_randomized_quantile_interval_experiment(
        n_offline=n_offline,
        n_online=n_online,
        quantile_method=quantile_method,
        seed=base_seed
    )
    return run_id, selected, t_records


def main():
    import datetime
    import pandas as pd

    # 要测试的不同行数列表（保持一致）
    n_online_list = [50]
    n_offline = 50
    # Monte Carlo 次数
    n_runs = 100

    # 生成带时间戳的文件名
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "experiment_results")
    os.makedirs(results_dir, exist_ok=True)

    # 汇总 TXT（保留）
    filename = os.path.join(results_dir, f"results_{timestamp}.txt")
    # 原始记录 CSV（新增）
    csv_path = os.path.join(results_dir, f"records_{timestamp}.csv")

    # 写入实验参数到 TXT
    with open(filename, 'w') as f:
        f.write(f"Experiment Results - {timestamp}\n")
        f.write("="*50 + "\n")
        f.write(f"n_online_list: {n_online_list}\n")
        f.write(f"n_offline: {n_offline}\n")
        f.write(f"n_runs: {n_runs}\n")
        f.write(f"alpha: {alpha}\n")
        f.write(f"M: {M}\n")
        f.write("="*50 + "\n\n")

    # 初始化 CSV 表头
    with open(csv_path, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow([
            "method", "n_online", "run_id", "t",
            "is_covered", "length", "is_inf", "cal_size",
            "selected_total"
        ])

    # 并行进程数：M2 建议先 6（避免过度并行 + BLAS 多线程）
    max_workers = min(6, os.cpu_count() or 4)

    # 外层总体进度条：方法×n_online 组合
    total_groups = len(["upper", "randomize"]) * len(n_online_list)
    group_bar = tqdm(total=total_groups, desc="All groups", position=0)

    for quantile_method in ["upper", "randomize"]:
        print(f"\nRunning experiments for {quantile_method} quantile method...")

        for n_online in n_online_list:
            futures = []
            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                for run_id in range(n_runs):
                    futures.append(ex.submit(
                        _worker_one_run,
                        run_id=run_id,
                        n_online=n_online,
                        n_offline=n_offline,
                        quantile_method=quantile_method,
                        base_seed=run_id
                    ))

                # 该组的进度条（内层）
                with open(csv_path, "a", newline="") as fcsv:
                    writer = csv.writer(fcsv)
                    inner_bar = tqdm(
                        total=len(futures),
                        desc=f"{quantile_method} | T={n_online}",
                        position=1,
                        leave=False
                    )
                    for fut in as_completed(futures):
                        run_id, selected, t_records = fut.result()
                        if len(t_records) == 0:
                            # 没有被选中的 t，也写一行占位（便于统计 selected_total）
                            writer.writerow([
                                quantile_method, n_online, run_id, -1,
                                "", "", "", "",
                                selected
                            ])
                        else:
                            for (t, iscov, length, is_inf, cal_size) in t_records:
                                writer.writerow([
                                    quantile_method, n_online, run_id, t,
                                    int(iscov),
                                    length,
                                    int(is_inf),
                                    float(cal_size),
                                    selected
                                ])
                        inner_bar.update(1)
                    inner_bar.close()
            group_bar.update(1)

    group_bar.close()
    print(f"\nRaw per-t records saved to CSV: {csv_path}")

    # ---------------- 从 CSV 读回并复用你原有的统计与画图逻辑 ----------------
    df = pd.read_csv(csv_path)

    # 保持与原来一致：只输出第一个 n_online 的详细表格/曲线
    n_online = n_online_list[0]

    summary_upper = []
    summary_randomize = []
    all_t_records_upper = []
    all_t_records_randomize = []

    for mth in ["upper", "randomize"]:
        df_m = df[(df["method"] == mth) & (df["n_online"] == n_online)]

        # selected_counts：按 run_id 取 selected_total 的首个值
        sel_by_run = (
            df_m[["run_id", "selected_total"]]
            .drop_duplicates(subset=["run_id"])
            .set_index("run_id")["selected_total"]
            .fillna(0)
        )
        selected_counts = sel_by_run.to_list()

        # 过滤掉 t<0 的占位行
        df_m_t = df_m[df_m["t"] >= 0].copy()

        # 组装回 all_t_records_* 结构
        all_t_records = [
            (int(row.t),
             bool(row.is_covered),
             row.length,
             bool(row.is_inf),
             float(row.cal_size))
            for _, row in df_m_t.iterrows()
        ]

        # 统计每个 t 的 selection conditional 指标（与你原版一致）
        t_to_cov = {}
        t_to_cov_count = {}
        t_to_len = {}
        t_to_len_count = {}
        t_to_inf = {}
        t_to_inf_count = {}
        t_to_cal = {}
        t_to_cal_count = {}

        for t, iscov, length, is_inf, cal_size in all_t_records:
            t_to_cov.setdefault(t, 0)
            t_to_cov_count.setdefault(t, 0)
            t_to_len.setdefault(t, 0.0)
            t_to_len_count.setdefault(t, 0)
            t_to_inf.setdefault(t, 0)
            t_to_inf_count.setdefault(t, 0)
            t_to_cal.setdefault(t, 0.0)
            t_to_cal_count.setdefault(t, 0)

            t_to_cov[t] += int(iscov)
            t_to_cov_count[t] += 1

            if not np.isnan(length):
                t_to_len[t] += float(length)
                t_to_len_count[t] += 1

            t_to_inf[t] += int(is_inf)
            t_to_inf_count[t] += 1

            t_to_cal[t] += float(cal_size)
            t_to_cal_count[t] += 1

        t_vals = sorted(t_to_cov.keys())
        t_covs = [t_to_cov[t] / t_to_cov_count[t] for t in t_vals]
        t_lens = [
            t_to_len[t] / t_to_len_count[t] if t_to_len_count[t] > 0 else np.nan
            for t in t_vals
        ]
        t_inf_fracs = [t_to_inf[t] / t_to_inf_count[t] for t in t_vals]
        t_cals = [t_to_cal[t] / t_to_cal_count[t] for t in t_vals]
        avg_sel = float(np.nanmean(selected_counts)) if len(selected_counts) > 0 else np.nan

        if mth == "upper":
            summary_upper.append((n_online, t_vals, t_covs, t_lens, t_inf_fracs, t_cals, avg_sel))
            all_t_records_upper = all_t_records
        else:
            summary_randomize.append((n_online, t_vals, t_covs, t_lens, t_inf_fracs, t_cals, avg_sel))
            all_t_records_randomize = all_t_records

    # ----- 以下打印与画图：保持一致 -----
    print("\nPer-t selection conditional statistics (mean/std across simulations):")
    header = ("Method", "t", "Count", "CovMean", "CovStd", "LenMedian", "LenStd", "InfFracMean", "InfFracStd", "CalMean", "CalStd")
    header_str = "{:>10} | {:>4} | {:>6} | {:>7} | {:>7} | {:>7} | {:>7} | {:>11} | {:>11} | {:>8} | {:>8}".format(*header)
    print(header_str)
    print("-"*102)

    with open(filename, 'a') as f:
        f.write("Per-t selection conditional statistics (mean/std across simulations):\n")
        f.write(header_str + "\n")
        f.write("-"*102 + "\n")

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

    # 绘制四子图
    plot_filename = os.path.join(results_dir, f"plot_{timestamp}.png")
    fig, axes = plt.subplots(2, 2, figsize=(15, 8))
    metrics = [
        (2, "Coverage", axes[0, 0]),
        (3, "Median Length", axes[0, 1]),
        (4, "Inf Fraction", axes[1, 0]),
        (5, "Mean Cal Size", axes[1, 1]),
    ]

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
            linestyle = '-' if method_name == "Upper" else '--'
            ax.plot(
                t_vals, vals,
                marker=marker,
                linestyle=linestyle,
                linewidth=2,
                markersize=4,
                alpha=0.8,
                label=f"{method_name} (T={n_online})",
                color=color
            )
        ax.set_xlabel("t")
        ax.set_ylabel(label)
        ax.set_title(f"Selection Conditional {label} vs t")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_filename}")


if __name__ == "__main__":
    main() 