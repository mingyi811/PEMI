import pandas as pd
import numpy as np
from pathlib import Path

# === 修改这里为你的 CSV 路径 ===
CSV_PATH = Path("/Users/zhengmingyi/JOMI-Online code/conformal p-value selection rule/experiment_results/records_20250810_211425.csv")

def compute_fcr_for_method(df: pd.DataFrame, method_name: str) -> float:
    """
    返回该 method 的 FCR(T) = E[ (#未覆盖) / (1 ∨ #被选中) ]
    """
    df_m = df[df["method"] == method_name].copy()

    # 仅保留 t>=0 的有效选择
    df_sel = df_m[df_m["t"].notna() & (df_m["t"] >= 0)].copy()
    df_sel["uncovered"] = np.where(df_sel["is_covered"] == 0, 1, 0)

    # 每个 run 的分子（未覆盖次数）
    num_by_run = df_sel.groupby("run_id")["uncovered"].sum()

    # 每个 run 的分母（1 ∨ selected_total）
    denom_by_run = df_m.groupby("run_id")["selected_total"].first().fillna(0).astype(int)
    denom_by_run = denom_by_run.clip(lower=1)

    # 对齐并计算比例
    runs = denom_by_run.index
    numerator = num_by_run.reindex(runs).fillna(0).astype(int)
    denominator = denom_by_run
    ratio = numerator / denominator

    return float(ratio.mean())

def main():
    df = pd.read_csv(CSV_PATH)

    methods = sorted(df["method"].dropna().unique().tolist())
    print(f"Loaded {len(df)} rows from: {CSV_PATH}")
    print(f"Methods found: {methods}\n")

    for m in methods:
        fcr_mean = compute_fcr_for_method(df, m)
        print(f"[{m}] FCR(T) = {fcr_mean:.6f}")

if __name__ == "__main__":
    main()
