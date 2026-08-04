#!/usr/bin/env python3

from pathlib import Path
import argparse
import numpy as np
import pandas as pd


ROOT = Path("/home/hatef/Desktop/Final-thesis-plots/experiments")

DEFAULT_DELTA_ROOT = ROOT / "paper-runs" / "delta_kg2_m6_q035_theta016_dosON"
DEFAULT_DIRECT_ROOT_CANDIDATES = [
    ROOT / "paper-runs" / "direct_hse",
    ROOT / "paper-runs" / "direct_hse_fullpool",
    ROOT / "paper-runs" / "direct_hse_al",
    ROOT / "random-sets",
]


def existing_direct_root():
    for p in DEFAULT_DIRECT_ROOT_CANDIDATES:
        if p.exists():
            return p
    return DEFAULT_DIRECT_ROOT_CANDIDATES[-1]


def find_metric_file(run_dir):
    candidates = [
        run_dir / "metrics" / "metrics.csv",
        run_dir / "run_v3" / "metrics" / "metrics.csv",
    ]

    for p in candidates:
        if p.exists():
            return p

    hits = sorted(run_dir.rglob("metrics.csv"))
    if hits:
        return hits[0]

    return None


def pick_col(df, candidates, required=True):
    norm = {c.lower().replace("-", "_"): c for c in df.columns}

    for c in candidates:
        key = c.lower().replace("-", "_")
        if key in norm:
            return norm[key]

    # loose contains search
    for c in candidates:
        key = c.lower().replace("-", "_")
        for actual in df.columns:
            a = actual.lower().replace("-", "_")
            if key in a:
                return actual

    if required:
        raise KeyError(
            f"Could not find any of {candidates}\nAvailable columns:\n{list(df.columns)}"
        )

    return None


def read_runs(root, run_prefixes):
    rows = []

    for prefix in run_prefixes:
        for run_dir in sorted(root.glob(f"{prefix}*")):
            if not run_dir.is_dir():
                continue

            metric_file = find_metric_file(run_dir)
            if metric_file is None:
                print(f"[WARN] no metrics.csv for {run_dir}")
                continue

            df = pd.read_csv(metric_file)

            cycle_col = pick_col(df, ["cycle"], required=True)

            end2end_col = pick_col(
                df,
                [
                    "end2end_hse_MAE",
                    "end2end_hse_mae",
                    "val_hse_mae",
                    "hse_val_mae",
                    "MAE",
                ],
                required=True,
            )

            # This column name differs between Δ and direct workflows.
            nonmetal_col = pick_col(
                df,
                [
                    "delta_val_MAE_nonmetal",
                    "delta_val_mae_nonmetal",
                    "mae_nonmetal",
                    "val_mae_nonmetal",
                    "hse_val_MAE_nonmetal",
                    "hse_val_mae_nonmetal",
                    "direct_val_MAE_nonmetal",
                    "direct_val_mae_nonmetal",
                    "nonmetal_MAE",
                    "nonmetal_mae",
                ],
                required=False,
            )

            for _, r in df.iterrows():
                rows.append({
                    "run": run_dir.name,
                    "metric_file": str(metric_file),
                    "cycle": pd.to_numeric(r[cycle_col], errors="coerce"),
                    "end2end_MAE": pd.to_numeric(r[end2end_col], errors="coerce"),
                    "nonmetal_MAE": (
                        pd.to_numeric(r[nonmetal_col], errors="coerce")
                        if nonmetal_col is not None
                        else np.nan
                    ),
                    "end2end_col": end2end_col,
                    "nonmetal_col": nonmetal_col if nonmetal_col else "",
                })

    out = pd.DataFrame(rows)
    out = out[out["cycle"].notna()].copy()
    out["cycle"] = out["cycle"].astype(int)
    return out


def summarize_variance(delta_df, direct_df, max_cycle):
    delta_df = delta_df[delta_df["cycle"].between(1, max_cycle)].copy()
    direct_df = direct_df[direct_df["cycle"].between(1, max_cycle)].copy()

    rows = []

    for metric in ["end2end_MAE", "nonmetal_MAE"]:
        for cycle in range(1, max_cycle + 1):
            dvals = delta_df.loc[delta_df["cycle"].eq(cycle), metric].dropna().to_numpy(float)
            rvals = direct_df.loc[direct_df["cycle"].eq(cycle), metric].dropna().to_numpy(float)

            d_var = np.var(dvals, ddof=1) if len(dvals) > 1 else np.nan
            r_var = np.var(rvals, ddof=1) if len(rvals) > 1 else np.nan

            rows.append({
                "metric": metric,
                "cycle": cycle,
                "n_delta": len(dvals),
                "n_direct": len(rvals),
                "delta_mean": np.mean(dvals) if len(dvals) else np.nan,
                "direct_mean": np.mean(rvals) if len(rvals) else np.nan,
                "delta_std": np.std(dvals, ddof=1) if len(dvals) > 1 else np.nan,
                "direct_std": np.std(rvals, ddof=1) if len(rvals) > 1 else np.nan,
                "delta_variance": d_var,
                "direct_variance": r_var,
                "variance_ratio_direct_over_delta": (
                    r_var / d_var
                    if np.isfinite(r_var) and np.isfinite(d_var) and d_var > 0
                    else np.nan
                ),
            })

    return pd.DataFrame(rows)


def ratio_summary(by_cycle):
    rows = []

    for metric, sub in by_cycle.groupby("metric"):
        ratios = sub["variance_ratio_direct_over_delta"].replace([np.inf, -np.inf], np.nan).dropna()

        rows.append({
            "metric": metric,
            "n_cycles_with_ratio": len(ratios),
            "mean_ratio_direct_over_delta": ratios.mean() if len(ratios) else np.nan,
            "median_ratio_direct_over_delta": ratios.median() if len(ratios) else np.nan,
            "min_ratio_direct_over_delta": ratios.min() if len(ratios) else np.nan,
            "max_ratio_direct_over_delta": ratios.max() if len(ratios) else np.nan,
            "cycle24_ratio_direct_over_delta": (
                sub.loc[sub["cycle"].eq(24), "variance_ratio_direct_over_delta"].iloc[0]
                if (sub["cycle"].eq(24)).any()
                else np.nan
            ),
            "cycle24_delta_variance": (
                sub.loc[sub["cycle"].eq(24), "delta_variance"].iloc[0]
                if (sub["cycle"].eq(24)).any()
                else np.nan
            ),
            "cycle24_direct_variance": (
                sub.loc[sub["cycle"].eq(24), "direct_variance"].iloc[0]
                if (sub["cycle"].eq(24)).any()
                else np.nan
            ),
        })

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delta-root", default=str(DEFAULT_DELTA_ROOT))
    ap.add_argument("--direct-root", default=str(existing_direct_root()))
    ap.add_argument("--max-cycle", type=int, default=24)
    ap.add_argument("--out-dir", default=str(ROOT / "artifacts" / "diagnostics" / "delta_vs_direct_variance"))
    args = ap.parse_args()

    delta_root = Path(args.delta_root)
    direct_root = Path(args.direct_root)
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("[delta root]", delta_root)
    print("[direct root]", direct_root)

    delta = read_runs(delta_root, ["al"])
    direct = read_runs(direct_root, ["direct"])

    if delta.empty:
        raise RuntimeError(f"No Δ-ML runs found under {delta_root}")
    if direct.empty:
        raise RuntimeError(f"No direct-HSE runs found under {direct_root}")

    print("\nΔ-ML runs found:")
    print(delta[["run", "metric_file", "end2end_col", "nonmetal_col"]].drop_duplicates().to_string(index=False))

    print("\nDirect-HSE runs found:")
    print(direct[["run", "metric_file", "end2end_col", "nonmetal_col"]].drop_duplicates().to_string(index=False))

    delta.to_csv(outdir / "delta_metrics_long.csv", index=False)
    direct.to_csv(outdir / "direct_metrics_long.csv", index=False)

    by_cycle = summarize_variance(delta, direct, args.max_cycle)
    summary = ratio_summary(by_cycle)

    by_cycle_path = outdir / "variance_by_cycle_delta_vs_direct.csv"
    summary_path = outdir / "variance_ratio_summary_delta_vs_direct.csv"

    by_cycle.to_csv(by_cycle_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("\nVariance by cycle:")
    print(by_cycle.to_string(index=False))

    print("\nRatio summary:")
    print(summary.to_string(index=False))

    print("\nWrote:")
    print(by_cycle_path)
    print(summary_path)


if __name__ == "__main__":
    main()
