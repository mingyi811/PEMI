import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import random
from tqdm import tqdm
from typing import Tuple, List
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv

from config import M, alpha, method, reference_set_method
from data import generate_data
from selection import selection_rule_conformal_p_value
from interval import construct_prediction_interval
import matplotlib.pyplot as plt


def is_covered(y: float, interval: Tuple[float, float, float, float]) -> bool:
    bound1, bound2, bound3, bound4 = interval
    return (bound1 <= y <= bound2) or (bound3 <= y <= bound4)


def run_randomized_quantile_interval_experiment(
    n_offline: int,
    n_online: int,
    M: int = M,
    alpha: float = alpha,
    quantile_method: str = "upper",
    seed: int = 0,
    reference_set_method: str = "ours"
) -> Tuple[List[Tuple[int, bool, float, bool, float]], int]:
    """
    Returns:
      - t_records: List[(t, iscov, length, is_inf, cal_size)]
      - selected: number of selected t in this run
    """
    np.random.seed(seed + 2)
    random.seed(seed + 2)
    rng_u = np.random.default_rng(seed + 2)
    Y_all, mu_all, c_all, w_i_all = generate_data(n_online+n_offline, seed)
    Y_on = Y_all[:n_online]
    mu_on = mu_all[:n_online]
    Y_off = Y_all[n_online:]
    mu_off = mu_all[n_online:]
    c_on = c_all[:n_online]
    w_i_on = w_i_all[:n_online]
    c_off = c_all[n_online:]
    w_i_off = w_i_all[n_online:]
    t_records = []  # (t, is_covered, length, is_inf, cal_size)
    selected = 0
    for t in range(n_online):
        indices = list(range(t + 1))
        base_perm = tuple(indices)
        u=rng_u.uniform(0,1)
        sel = selection_rule_conformal_p_value(w_i_on[:t+1], c_on[:t+1], Y_on[:t+1], mu_on[:t+1], Y_off, mu_off, c_off, w_i_off, base_perm, t, 1, u, method=method)
        if not sel:
            continue
        selected += 1
        interval = construct_prediction_interval(t, Y_on, mu_on, Y_off, mu_off, c_on, w_i_on, c_off, w_i_off, M, alpha, u, quantile_method=quantile_method, reference_set_method=reference_set_method)
        iscov = is_covered(Y_on[t], interval[:4])
        bound_1, bound_2, bound_3, bound_4 = interval[:4]
        length = (bound_2 - bound_1) + (bound_4 - bound_3)
        is_inf = not np.isfinite(length)
        cal_size = interval[4]
        t_records.append((t, iscov, length, is_inf, cal_size))
    return t_records, selected


# Parallel worker: run single experiment for one seed
def _worker_one_run(run_id: int, n_online: int, n_offline: int, quantile_method: str, base_seed: int, reference_set_method: str):
    """
    Returns (run_id, selected, t_records)
    t_records: List[(t, iscov, length, is_inf, cal_size)]
    """
    # Reproducible RNG setup (consistent with original code)
    np.random.seed(base_seed + 2)
    random.seed(base_seed + 2)

    t_records, selected = run_randomized_quantile_interval_experiment(
        n_offline=n_offline,
        n_online=n_online,
        quantile_method=quantile_method,
        seed=base_seed,
        reference_set_method=reference_set_method
    )
    return run_id, selected, t_records


def main():
    import datetime

    # List of n_online values to test
    n_online_list = [50]
    n_offline = 50
    # Number of runs
    n_runs = 100000

    # Timestamp and directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "experiment_results")
    os.makedirs(results_dir, exist_ok=True)

    # Result files: CSV and TXT (keep original txt summary)
    csv_path = os.path.join(results_dir, f"records_{timestamp}.csv")
    txt_path = os.path.join(results_dir, f"results_{timestamp}.txt")

    # Write experiment parameters to txt
    with open(txt_path, 'w') as f:
        f.write(f"Experiment Results - {timestamp}\n")
        f.write("="*50 + "\n")
        f.write(f"n_online_list: {n_online_list}\n")
        f.write(f"n_offline: {n_offline}\n")
        f.write(f"n_runs: {n_runs}\n")
        f.write(f"alpha: {alpha}\n")
        f.write(f"M: {M}\n")
        f.write("="*50 + "\n\n")

    # Initialize CSV
    with open(csv_path, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow([
            "method", "n_online", "run_id", "t",
            "is_covered", "length", "is_inf", "cal_size",
            "selected_total"
        ])

    # Run all experiments in parallel, write to CSV as results come in
    max_workers = min(6, os.cpu_count() or 4)

    for quantile_method in ["upper", "randomize"]:
        print(f"\nRunning experiments for {quantile_method} quantile method...")

        for n_online in n_online_list:
            # Submit all run tasks
            futures = []
            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                for run_id in range(n_runs):
                    # base_seed = run_id (consistent with original, internally uses seed+2 for RNG)
                    futures.append(ex.submit(
                        _worker_one_run,
                        run_id=run_id,
                        n_online=n_online,
                        n_offline=n_offline,
                        quantile_method=quantile_method,
                        base_seed=run_id,
                        reference_set_method=reference_set_method
                    ))

                # Wait for completion and write rows to CSV
                with open(csv_path, "a", newline="") as fcsv:
                    writer = csv.writer(fcsv)
                    #for fut in as_completed(futures):
                    for fut in tqdm(as_completed(futures), total=len(futures), desc=f"{quantile_method} T={n_online}"):
 
                        run_id, selected, t_records = fut.result()
                        if len(t_records) == 0:
                            # Placeholder row for runs with no selected t (for selected statistics)
                            writer.writerow([
                                quantile_method, n_online, run_id, -1,
                                "", "", "", "",
                                selected
                            ])
                        else:
                            for (t, iscov, length, is_inf, cal_size) in t_records:
                                writer.writerow([
                                    quantile_method, n_online, run_id, t,
                                    int(iscov), length,
                                    int(is_inf), float(cal_size),
                                    selected
                                ])

    print(f"\nRaw per-t records saved to CSV: {csv_path}")

    # Read CSV and reuse original statistics and plotting logic
    # Strictly follow original process for aggregation and plotting to maintain consistency

    import pandas as pd
    df = pd.read_csv(csv_path)

    # Keep only current n_online (original code only plots detailed t table for first n_online)
    n_online = n_online_list[0]

    summary_upper = []
    summary_randomize = []
    all_t_records_upper = []
    all_t_records_randomize = []

    # Aggregate by method (consistent with original code's 4 metrics)
    for quantile_method in ["upper", "randomize"]:
        df_m = df[(df["method"] == quantile_method) & (df["n_online"] == n_online)]
        # Get selected_total first value by run_id
        sel_by_run = (
            df_m[["run_id", "selected_total"]]
            .drop_duplicates(subset=["run_id"])
            .set_index("run_id")["selected_total"]
            .fillna(0)
        )
        selected_counts = sel_by_run.to_list()

        # Filter out placeholder rows with t<0
        df_m_t = df_m[df_m["t"] >= 0].copy()

        # Assemble all_t_records_* list structure to reuse subsequent code
        all_t_records = [
            (int(row.t),
             bool(row.is_covered),
             row.length ,#if not np.isnan(row.length) else np.nan,
             bool(row.is_inf),
             float(row.cal_size))
            for _, row in df_m_t.iterrows()
        ]

        # Compute selection conditional metrics for each t
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
            t_to_cov[t] += int(iscov)
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
            t_to_inf[t] += int(is_inf)
            t_to_inf_count[t] += 1
            # cal_size
            t_to_cal.setdefault(t, 0.0)
            t_to_cal_count.setdefault(t, 0)
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

        if quantile_method == "upper":
            summary_upper.append((n_online, t_vals, t_covs, t_lens, t_inf_fracs, t_cals, avg_sel))
            all_t_records_upper = all_t_records
        else:
            summary_randomize.append((n_online, t_vals, t_covs, t_lens, t_inf_fracs, t_cals, avg_sel))
            all_t_records_randomize = all_t_records

    # Print and plot (reusing original logic)
    print("\nPer-t selection conditional statistics (mean/std across simulations):")
    header = ("Method", "t", "Count", "CovMean", "CovStd", "LenMedian", "LenStd", "InfFracMean", "InfFracStd", "CalMean", "CalStd")
    header_str = "{:>10} | {:>4} | {:>6} | {:>7} | {:>7} | {:>7} | {:>7} | {:>11} | {:>11} | {:>8} | {:>8}".format(*header)
    print(header_str)
    print("-"*102)

    with open(txt_path, 'a') as f:
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
            with open(txt_path, 'a') as f:
                f.write(row_str + "\n")

    print(f"\nResults saved to: {txt_path}")

    # Plot (consistent with original code)
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
