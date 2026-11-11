#!/usr/bin/env python3
"""
Gate-rescue evidence figure
---------------------------------
Claim-oriented figure showing that the gate can rescue recoverable
baseline zero-gap nonmetals, even if quantitative HSE-gap prediction
for that subset is not perfect.

Panels
------
A) Recovery of baseline zero-gap nonmetals
   stacked bars: mean rescued + mean missed per replicate

B) Rescue-oriented gate rates
   bars: mean recall / precision across replicates
   error bars: SD across replicates
   colored dots: individual replicate values

C) Quantitative HSE-gap error after rescue
   bars: mean HSE MAE across replicates for recoverable zero-gap nonmetals
   error bars: SD across replicates
   colored dots: individual replicate values
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


# -----------------------------
# Colors
# -----------------------------
COLORS = {
    "delta_bar": "#003566",
    "direct_bar": "#7b2cbf",
    "rescued": "#0a9396",
    "missed": "#ee9b00",
    "recall": "#0a9396",
    "precision": "#2d6a4f",
    "ink": "#111111",
    "grid": "#cccccc",
    "legend_bg": "#ffffff",
}

# replicate colors (consistent across all panels)
REPL_COLORS = ["#0a9396", "#ee9b00", "#003566", "#ff4d6d", "#2d6a4f"]

METHOD_LABELS = {
    "delta": r"$\Delta$-ML AL",
    "direct": "Direct-HSE AL",
}


# -----------------------------
# Helpers
# -----------------------------
def load_gate_threshold(run_dir: Path, cycle: int, default_thr: float):
    metrics_file = run_dir / "metrics" / "metrics.csv"
    if not metrics_file.exists():
        return default_thr
    try:
        dfm = pd.read_csv(metrics_file)
        row = dfm.loc[dfm["cycle"] == cycle]
        if len(row) == 0:
            return default_thr
        if "gate_decision_thr" in row.columns:
            v = float(row["gate_decision_thr"].iloc[0])
            if np.isfinite(v):
                return v
    except Exception:
        pass
    return default_thr


def load_pred_file(run_dir: Path, cycle: int):
    p = run_dir / "metrics" / f"val_pred_cycle{cycle}.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing prediction file: {p}")
    return pd.read_csv(p)


def add_masks(df: pd.DataFrame, pbe_zero_thr: float, metal_thr: float, gate_thr: float):
    out = df.copy()

    if "is_pbe0" in out.columns:
        out["is_baseline_zbg"] = out["is_pbe0"].astype(float) >= 0.5
    else:
        if "pbe_gap_effective" not in out.columns:
            raise KeyError("Need either 'is_pbe0' or 'pbe_gap_effective'.")
        out["is_baseline_zbg"] = pd.to_numeric(out["pbe_gap_effective"], errors="coerce") <= pbe_zero_thr

    out["is_recoverable_nonmetal"] = (
        out["is_baseline_zbg"] &
        (pd.to_numeric(out["hse_true"], errors="coerce") > metal_thr)
    )

    if "p_gate" not in out.columns:
        raise KeyError("Need 'p_gate' in val_pred file.")
    out["is_rescued"] = (
        out["is_baseline_zbg"] &
        (pd.to_numeric(out["p_gate"], errors="coerce") >= gate_thr)
    )

    out["hse_abs_err"] = (
        pd.to_numeric(out["hse_pred"], errors="coerce")
        - pd.to_numeric(out["hse_true"], errors="coerce")
    ).abs()

    if {"delta_true", "delta_pred"}.issubset(out.columns):
        out["delta_abs_err"] = (
            pd.to_numeric(out["delta_pred"], errors="coerce")
            - pd.to_numeric(out["delta_true"], errors="coerce")
        ).abs()
    else:
        out["delta_abs_err"] = np.nan

    return out


def summarize_run(run_dir: Path, method_key: str, cycle: int, pbe_zero_thr: float, metal_thr: float, gate_thr_default: float):
    gate_thr = load_gate_threshold(run_dir, cycle, gate_thr_default)
    df = load_pred_file(run_dir, cycle)
    df = add_masks(df, pbe_zero_thr=pbe_zero_thr, metal_thr=metal_thr, gate_thr=gate_thr)

    zbg = df["is_baseline_zbg"]
    rec = df["is_recoverable_nonmetal"]
    rescued = df["is_rescued"]

    rescued_true = rec & rescued
    missed = rec & (~rescued)
    false_rescue = zbg & (~rec) & rescued

    n_zbg = int(zbg.sum())
    n_recoverable = int(rec.sum())
    n_rescued_true = int(rescued_true.sum())
    n_missed = int(missed.sum())
    n_all_rescued = int(rescued.sum())
    n_false_rescue = int(false_rescue.sum())

    rescue_recall = n_rescued_true / n_recoverable if n_recoverable > 0 else np.nan
    rescue_precision = n_rescued_true / n_all_rescued if n_all_rescued > 0 else np.nan
    false_rescue_rate = n_false_rescue / n_all_rescued if n_all_rescued > 0 else np.nan

    rec_df = df.loc[rec].copy()
    hse_mae_recoverable = rec_df["hse_abs_err"].mean() if len(rec_df) > 0 else np.nan
    delta_mae_recoverable = rec_df["delta_abs_err"].mean() if len(rec_df) > 0 else np.nan

    row = {
        "method": method_key,
        "run": run_dir.name,
        "cycle": cycle,
        "gate_decision_thr": gate_thr,
        "pbe_zero_thr": pbe_zero_thr,
        "metal_thr": metal_thr,
        "baseline_zbg_n": n_zbg,
        "recoverable_n": n_recoverable,
        "rescued_n": n_rescued_true,
        "missed_n": n_missed,
        "all_rescued_n": n_all_rescued,
        "false_rescue_n": n_false_rescue,
        "rescue_recall": rescue_recall,
        "rescue_precision": rescue_precision,
        "false_rescue_rate": false_rescue_rate,
        "hse_mae_recoverable": hse_mae_recoverable,
        "delta_mae_recoverable": delta_mae_recoverable,
    }

    return row


def collect_runs(base_dir: Path, run_glob: str):
    runs = sorted([p for p in base_dir.glob(run_glob) if p.is_dir()])
    if not runs:
        raise FileNotFoundError(f"No run directories matched: {base_dir / run_glob}")
    return runs


def grouped_summary(df: pd.DataFrame, value_cols):
    rows = []
    for method_key, g in df.groupby("method"):
        row = {
            "method": method_key,
            "method_label": METHOD_LABELS.get(method_key, method_key),
            "n_runs": len(g),
        }
        for c in value_cols:
            vals = pd.to_numeric(g[c], errors="coerce").to_numpy(float)
            row[f"{c}_mean"] = np.nanmean(vals)
            row[f"{c}_sd"] = np.nanstd(vals, ddof=1) if len(vals) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def jitter_points(x_center, values, width=0.05):
    n = len(values)
    if n == 1:
        return np.array([x_center])
    return np.linspace(x_center - width, x_center + width, n)


# -----------------------------
# Plot
# -----------------------------
def make_figure(per_run: pd.DataFrame, summary: pd.DataFrame, out_path_base: Path):
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 15,
        "axes.labelsize": 15,
        "axes.linewidth": 1.4,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "xtick.major.width": 1.4,
        "ytick.major.width": 1.4,
        "font.family": "DejaVu Sans",
    })

    fig, axes = plt.subplots(1, 3, figsize=(16.8, 5.3), dpi=300)
    axA, axB, axC = axes

    methods = ["delta", "direct"]
    x = np.arange(len(methods))
    summ_idx = summary.set_index("method")

    legend_kw = dict(
        frameon=True,
        framealpha=0.92,
        facecolor=COLORS["legend_bg"],
        edgecolor="#bbbbbb",
        fontsize=12
    )

    # -------------------------
    # Panel A
    # -------------------------
    rescued_mean = np.array([summ_idx.loc[m, "rescued_n_mean"] for m in methods], dtype=float)
    missed_mean = np.array([summ_idx.loc[m, "missed_n_mean"] for m in methods], dtype=float)
    recoverable_mean = rescued_mean + missed_mean
    width = 0.55

    axA.bar(
        x, rescued_mean, width=width,
        color=COLORS["rescued"], edgecolor="white", linewidth=1.2, label="rescued"
    )
    axA.bar(
        x, missed_mean, width=width, bottom=rescued_mean,
        color=COLORS["missed"], edgecolor="white", linewidth=1.2, label="missed"
    )

    # mean total labels with more padding
    for i, total in enumerate(recoverable_mean):
        axA.text(
            x[i], total + 0.10, f"total {total:.1f}",
            ha="center", va="bottom", fontsize=13
        )

    max_total = max(recoverable_mean) if len(recoverable_mean) else 1.0
    axA.set_ylim(0, max_total + 0.45)

    axA.set_title("Recovery of baseline zero-gap nonmetals", pad=12)
    axA.set_ylabel("Count per replicate")
    axA.set_xticks(x)
    axA.set_xticklabels([METHOD_LABELS[m] for m in methods])
    axA.grid(axis="y", color=COLORS["grid"], linewidth=0.8, alpha=0.75)
    axA.set_axisbelow(True)
    axA.legend(loc="lower center", bbox_to_anchor=(0.50, 0.82), bbox_transform=axA.transAxes, **legend_kw)
    axA.text(0.02, 0.98, "A", transform=axA.transAxes, ha="left", va="top",
             fontsize=18, fontweight="bold")

    # -------------------------
    # Panel B
    # -------------------------
    metric_names = ["rescue_recall", "rescue_precision"]
    metric_labels = ["recall", "precision"]
    metric_colors = [COLORS["recall"], COLORS["precision"]]
    bar_w = 0.30

    for j, (metric, mlabel, mcolor) in enumerate(zip(metric_names, metric_labels, metric_colors)):
        centers = x + (j - 0.5) * bar_w
        means = np.array([summ_idx.loc[m, f"{metric}_mean"] for m in methods], dtype=float)
        sds   = np.array([summ_idx.loc[m, f"{metric}_sd"] for m in methods], dtype=float)

        axB.bar(
            centers, means, width=bar_w,
            color=mcolor, edgecolor="white", linewidth=1.2,
            yerr=sds, capsize=5, label=mlabel
        )

        # colored replicate dots
        for i, method_key in enumerate(methods):
            g = per_run.loc[per_run["method"] == method_key].sort_values("run")
            vals = pd.to_numeric(g[metric], errors="coerce").to_numpy(float)
            xs = jitter_points(centers[i], vals, width=0.055)
            for k, (xx, yy) in enumerate(zip(xs, vals)):
                axB.scatter(
                    xx, yy,
                    s=44,
                    facecolor=REPL_COLORS[k % len(REPL_COLORS)],
                    edgecolor=COLORS["ink"],
                    linewidth=0.9,
                    zorder=5
                )

    axB.set_title("Rescue-oriented gate rates", pad=12)
    axB.set_ylabel("Rate")
    axB.set_ylim(0, 1.18)
    axB.set_yticks(np.arange(0, 1.01, 0.2))
    axB.set_xticks(x)
    axB.set_xticklabels([METHOD_LABELS[m] for m in methods])
    axB.grid(axis="y", color=COLORS["grid"], linewidth=0.8, alpha=0.75)
    axB.set_axisbelow(True)
    axB.legend(loc="lower center", bbox_to_anchor=(0.50, 0.82), bbox_transform=axB.transAxes, **legend_kw)
    axB.text(0.02, 0.98, "B", transform=axB.transAxes, ha="left", va="top",
             fontsize=18, fontweight="bold")

    # -------------------------
    # Panel C
    # -------------------------
    err_col = "hse_mae_recoverable"
    means = np.array([summ_idx.loc[m, f"{err_col}_mean"] for m in methods], dtype=float)
    sds   = np.array([summ_idx.loc[m, f"{err_col}_sd"] for m in methods], dtype=float)
    bar_colors = [COLORS["delta_bar"], COLORS["direct_bar"]]

    axC.bar(
        x, means, width=0.55, color=bar_colors,
        edgecolor=COLORS["ink"], linewidth=1.0,
        yerr=sds, capsize=6
    )

    all_vals = []
    for i, method_key in enumerate(methods):
        g = per_run.loc[per_run["method"] == method_key].sort_values("run")
        vals = pd.to_numeric(g[err_col], errors="coerce").dropna().to_numpy(float)
        xs = jitter_points(x[i], vals, width=0.12)
        for k, (xx, yy) in enumerate(zip(xs, vals)):
            axC.scatter(
                xx, yy,
                s=48,
                facecolor=REPL_COLORS[k % len(REPL_COLORS)],
                edgecolor=COLORS["ink"],
                linewidth=1.0,
                zorder=5
            )
            all_vals.append(yy)

        axC.text(
            x[i], means[i] + sds[i] + 0.035, f"{means[i]:.3f}",
            ha="center", va="bottom", fontsize=13
        )

    ymax = max(max(all_vals) if all_vals else 0.5, np.max(means + sds)) + 0.06
    axC.set_ylim(0, ymax)

    axC.set_title("Quantitative HSE-gap error after rescue", pad=12)
    axC.set_ylabel("HSE MAE (eV)")
    axC.set_xticks(x)
    axC.set_xticklabels([METHOD_LABELS[m] for m in methods])
    axC.grid(axis="y", color=COLORS["grid"], linewidth=0.8, alpha=0.75)
    axC.set_axisbelow(True)
    axC.text(0.02, 0.98, "C", transform=axC.transAxes, ha="left", va="top",
             fontsize=18, fontweight="bold")

    # panel C legend
    handles_C = [
        Patch(facecolor="#999999", edgecolor=COLORS["ink"], label="bar = mean ± SD"),
        Line2D([0], [0], marker="o", linestyle="None", markersize=7,
               markerfacecolor=REPL_COLORS[0], markeredgecolor=COLORS["ink"],
               label="colored dots = replicates")
    ]
    axC.legend(handles=handles_C, loc="lower center", bbox_to_anchor=(0.50, 0.82), bbox_transform=axC.transAxes, **legend_kw)

    fig.suptitle("Gate-rescue evidence in the baseline zero-gap regime",
                 y=1.03, fontsize=18, fontweight="bold")
    fig.tight_layout(w_pad=1.3)

    for ext in [".png", ".pdf", ".svg"]:
        fig.savefig(out_path_base.with_suffix(ext), bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Gate-rescue evidence figure")
    parser.add_argument("--root", default=".")
    parser.add_argument("--delta-dir", required=True)
    parser.add_argument("--direct-dir", required=True)
    parser.add_argument("--delta-glob", default="al*")
    parser.add_argument("--direct-glob", default="direct*")
    parser.add_argument("--cycle", type=int, default=24)
    parser.add_argument("--pbe-zero-thr", type=float, default=0.05)
    parser.add_argument("--metal-thr", type=float, default=0.05)
    parser.add_argument("--gate-decision-thr", type=float, default=0.16)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--out-prefix", default="fig_gate_rescue_evidence")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    delta_base = (root / args.delta_dir).resolve()
    direct_base = (root / args.direct_dir).resolve()
    out_dir = (root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    delta_runs = collect_runs(delta_base, args.delta_glob)
    direct_runs = collect_runs(direct_base, args.direct_glob)

    per_run_rows = []

    for run_dir in delta_runs:
        per_run_rows.append(
            summarize_run(
                run_dir=run_dir,
                method_key="delta",
                cycle=args.cycle,
                pbe_zero_thr=args.pbe_zero_thr,
                metal_thr=args.metal_thr,
                gate_thr_default=args.gate_decision_thr,
            )
        )

    for run_dir in direct_runs:
        per_run_rows.append(
            summarize_run(
                run_dir=run_dir,
                method_key="direct",
                cycle=args.cycle,
                pbe_zero_thr=args.pbe_zero_thr,
                metal_thr=args.metal_thr,
                gate_thr_default=args.gate_decision_thr,
            )
        )

    per_run = pd.DataFrame(per_run_rows)

    value_cols = [
        "baseline_zbg_n",
        "recoverable_n",
        "rescued_n",
        "missed_n",
        "all_rescued_n",
        "false_rescue_n",
        "rescue_recall",
        "rescue_precision",
        "false_rescue_rate",
        "hse_mae_recoverable",
        "delta_mae_recoverable",
    ]
    summary = grouped_summary(per_run, value_cols=value_cols)

    out_base = out_dir / args.out_prefix

    per_run.to_csv(out_base.with_name(out_base.name + "_per_run.csv"), index=False)
    summary.to_csv(out_base.with_name(out_base.name + "_summary.csv"), index=False)

    make_figure(per_run=per_run, summary=summary, out_path_base=out_base)

    print("[OK] wrote figure:")
    print("   ", out_base.with_suffix(".png"))
    print("   ", out_base.with_suffix(".pdf"))
    print("   ", out_base.with_suffix(".svg"))
    print("[OK] wrote tables:")
    print("   ", out_base.with_name(out_base.name + "_per_run.csv"))
    print("   ", out_base.with_name(out_base.name + "_summary.csv"))

    print("\nPer-run overview:")
    print(per_run.to_string(index=False))

    print("\nGroup summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()