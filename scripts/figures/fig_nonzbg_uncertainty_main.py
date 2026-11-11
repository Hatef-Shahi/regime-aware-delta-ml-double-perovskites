#!/usr/bin/env python3
"""
Main-paper figure:
Non-ZBG uncertainty diagnostics.

Purpose
-------
This figure tests whether the uncertainty estimate used in the non-ZBG
Delta-regression branch is informative.

Panel A:
    Pooled uncertainty-error relation across AL validation replicates.

Panel B:
    Absolute Delta error grouped by uncertainty tertile.

This avoids the circular plot showing that selected candidates had high
acquisition scores.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PALETTE = {
    "navy": "#003566",
    "teal": "#0a9396",
    "gold": "#ee9b00",
    "pink": "#ff4d6d",
    "green": "#2d6a4f",
    "purple": "#7b2cbf",
    "gray": "#888888",
    "light": "#dddddd",
    "ink": "#111111",
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Main-paper non-ZBG uncertainty diagnostic figure"
    )
    p.add_argument("--root", default=".")
    p.add_argument("--runs-dir", required=True)
    p.add_argument("--run-glob", default="al*")
    p.add_argument("--cycle", type=int, default=24)
    p.add_argument("--baseline-zero-thr", type=float, default=0.05)
    p.add_argument("--hse-metal-thr", type=float, default=0.05)
    p.add_argument("--sigma-val", default="sigma_gpr")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--out-prefix", default="fig_nonzbg_uncertainty_main")
    p.add_argument("--title", default="Non-ZBG uncertainty diagnostics")
    p.add_argument("--dpi", type=int, default=600)
    p.add_argument("--figwidth", type=float, default=7.0)
    p.add_argument("--figheight", type=float, default=3.2)
    return p.parse_args()


def to_num(df, col, default=np.nan):
    if col not in df.columns:
        return np.full(len(df), default, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").to_numpy(float)


def nonzbg_val_mask(df_val, thr, hse_thr):
    pbe = to_num(df_val, "pbe_gap_effective", default=np.nan)

    if "is_pbe0" in df_val.columns:
        is_zbg = to_num(df_val, "is_pbe0", default=0.0) > 0.5
    else:
        is_zbg = np.isfinite(pbe) & (pbe <= thr)

    hse_true = to_num(df_val, "hse_true", default=np.nan)

    label_ok = np.ones(len(df_val), dtype=bool)
    if "label_status" in df_val.columns:
        st = df_val["label_status"].astype(str).str.upper().str.strip().to_numpy()
        label_ok = st != "METAL_ASSUMED"

    return (
        np.isfinite(pbe)
        & (pbe > thr)
        & (~is_zbg)
        & label_ok
        & np.isfinite(hse_true)
        & (hse_true > hse_thr)
    )


def spearman_np(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]

    if len(x) < 3:
        return np.nan

    rx = pd.Series(x).rank(method="average").to_numpy()
    ry = pd.Series(y).rank(method="average").to_numpy()

    rx = rx - rx.mean()
    ry = ry - ry.mean()

    denom = np.sqrt(np.sum(rx**2) * np.sum(ry**2))
    if denom == 0:
        return np.nan

    return float(np.sum(rx * ry) / denom)


def binned_mean(x, y, nbins=6):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]

    if len(x) < nbins:
        return np.array([]), np.array([])

    edges = np.unique(np.quantile(x, np.linspace(0, 1, nbins + 1)))
    if len(edges) < 4:
        return np.array([]), np.array([])

    bx, by = [], []

    for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        if i == len(edges) - 2:
            mm = (x >= a) & (x <= b)
        else:
            mm = (x >= a) & (x < b)

        if mm.sum() >= 2:
            bx.append(np.mean(x[mm]))
            by.append(np.mean(y[mm]))

    return np.asarray(bx), np.asarray(by)


def collect_uncertainty_data(args):
    root = Path(args.root).resolve()
    base = root / args.runs_dir

    run_dirs = sorted([p for p in base.glob(args.run_glob) if p.is_dir()])
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found: {base / args.run_glob}")

    rows = []
    per_run_rows = []

    for run_dir in run_dirs:
        val_path = run_dir / "metrics" / f"val_pred_cycle{args.cycle}.csv"

        if not val_path.exists():
            print(f"[SKIP] missing {val_path}")
            continue

        val = pd.read_csv(val_path)

        vm = nonzbg_val_mask(
            val,
            thr=args.baseline_zero_thr,
            hse_thr=args.hse_metal_thr,
        )
        v = val.loc[vm].copy()

        delta_true = to_num(v, "delta_true", default=np.nan)

        if "delta_pred_stageB" in v.columns:
            delta_pred = to_num(v, "delta_pred_stageB", default=np.nan)
        elif "delta_pred" in v.columns:
            delta_pred = to_num(v, "delta_pred", default=np.nan)
        else:
            raise KeyError(f"{val_path} missing delta_pred or delta_pred_stageB")

        if args.sigma_val in v.columns:
            sigma = to_num(v, args.sigma_val, default=np.nan)
            sigma_col_used = args.sigma_val
        else:
            sigma_col_used = None
            for c in ["sigma_gpr", "sigma_mix", "sigma_eff"]:
                if c in v.columns:
                    sigma_col_used = c
                    sigma = to_num(v, c, default=np.nan)
                    break
            if sigma_col_used is None:
                raise KeyError(f"No uncertainty column found in {val_path}")

        err = np.abs(delta_pred - delta_true)

        m = np.isfinite(sigma) & np.isfinite(err)

        for s, e in zip(sigma[m], err[m]):
            rows.append({
                "run": run_dir.name,
                "cycle": args.cycle,
                "sigma_col": sigma_col_used,
                "sigma_delta": float(s),
                "abs_delta_error": float(e),
            })

        per_run_rows.append({
            "run": run_dir.name,
            "cycle": args.cycle,
            "n_nonzbg_val": int(m.sum()),
            "spearman_uncertainty_error": spearman_np(sigma[m], err[m]),
            "coverage_1sigma": float(np.mean(err[m] <= sigma[m])) if m.sum() else np.nan,
            "coverage_2sigma": float(np.mean(err[m] <= 2 * sigma[m])) if m.sum() else np.nan,
            "mean_abs_delta_error": float(np.mean(err[m])) if m.sum() else np.nan,
            "median_abs_delta_error": float(np.median(err[m])) if m.sum() else np.nan,
        })

    data = pd.DataFrame(rows)
    per_run = pd.DataFrame(per_run_rows)

    if data.empty:
        raise RuntimeError("No valid non-ZBG validation uncertainty-error data collected.")

    return data, per_run


def assign_uncertainty_tertiles(data):
    df = data.copy()

    labels = ["low", "medium", "high"]

    try:
        df["uncertainty_tertile"] = pd.qcut(
            df["sigma_delta"],
            q=3,
            labels=labels,
            duplicates="drop",
        )
    except ValueError:
        df["uncertainty_tertile"] = pd.cut(
            df["sigma_delta"],
            bins=3,
            labels=labels,
            include_lowest=True,
        )

    return df


def make_plot(data, per_run, args):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "axes.linewidth": 1.0,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 7.8,
        "figure.titlesize": 12,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    sigma = data["sigma_delta"].to_numpy(float)
    err = data["abs_delta_error"].to_numpy(float)

    rho = spearman_np(sigma, err)
    cov1 = float(np.mean(err <= sigma))
    cov2 = float(np.mean(err <= 2 * sigma))

    data_t = assign_uncertainty_tertiles(data)

    tertile_order = ["low", "medium", "high"]
    tertile_colors = [PALETTE["teal"], PALETTE["gold"], PALETTE["pink"]]

    fig, (axA, axB) = plt.subplots(
        1,
        2,
        figsize=(args.figwidth, args.figheight),
        dpi=300,
    )

    # -------------------------
    # Panel A: scatter + binned mean
    # -------------------------
    axA.scatter(
        sigma,
        err,
        s=22,
        color=PALETTE["purple"],
        alpha=0.60,
        edgecolors="none",
        label="validation candidates",
    )

    bx, by = binned_mean(sigma, err, nbins=6)
    if len(bx):
        axA.plot(
            bx,
            by,
            color=PALETTE["ink"],
            marker="o",
            markersize=3.6,
            linewidth=1.4,
            label="pooled binned mean",
        )

    xmax = np.nanmax(sigma) * 1.05 if np.isfinite(sigma).any() else 1.0
    xx = np.linspace(0, xmax, 100)

    axA.plot(
        xx,
        xx,
        color=PALETTE["gray"],
        linestyle="--",
        linewidth=1.0,
        label=r"$|e|=\sigma$",
    )
    axA.plot(
        xx,
        2 * xx,
        color=PALETTE["gray"],
        linestyle=":",
        linewidth=1.0,
        label=r"$|e|=2\sigma$",
    )

    axA.text(
        0.05,
        0.9,
        f"Spearman ρ = {rho:.2f}\n|e|≤σ = {cov1:.2f}\n|e|≤2σ = {cov2:.2f}",
        transform=axA.transAxes,
        ha="left",
        va="top",
        fontsize=6.5,
        bbox=dict(facecolor="white", edgecolor=PALETTE["light"], alpha=0.92),
    )

    axA.set_title("Uncertainty-error relation")
    axA.set_xlabel(r"Predicted uncertainty $\sigma_\Delta$ (eV)")
    axA.set_ylabel(r"Absolute $\Delta$ error (eV)")
    axA.grid(True, color=PALETTE["light"], alpha=0.30)
    axA.legend(
        loc="upper right",
        frameon=True,
        fontsize=6,
        handlelength=2.0,
        borderpad=0.35,
    )
    axA.text(
        0.02,
        0.98,
        "A",
        transform=axA.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
    )

    # -------------------------
    # Panel B: error by uncertainty tertile
    # -------------------------
    box_data = []
    counts = []
    means = []

    for lab in tertile_order:
        vals = data_t.loc[
            data_t["uncertainty_tertile"].astype(str) == lab,
            "abs_delta_error",
        ].to_numpy(float)
        vals = vals[np.isfinite(vals)]
        box_data.append(vals)
        counts.append(len(vals))
        means.append(np.mean(vals) if len(vals) else np.nan)

    bp = axB.boxplot(
        box_data,
        patch_artist=True,
        widths=0.55,
        showfliers=False,
        medianprops=dict(color=PALETTE["ink"], linewidth=1.2),
        boxprops=dict(color=PALETTE["ink"], linewidth=1.0),
        whiskerprops=dict(color=PALETTE["ink"], linewidth=1.0),
        capprops=dict(color=PALETTE["ink"], linewidth=1.0),
    )

    for patch, color in zip(bp["boxes"], tertile_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)

    # jittered points
    rng = np.random.default_rng(42)
    for i, vals in enumerate(box_data, start=1):
        if len(vals) == 0:
            continue
        x = i + rng.normal(0, 0.045, size=len(vals))
        axB.scatter(
            x,
            vals,
            s=15,
            color=tertile_colors[i - 1],
            alpha=0.55,
            edgecolors="none",
            zorder=3,
        )
        axB.text(
            i,
            max(vals) + 0.03,
            f"n={counts[i-1]}\nmean={means[i-1]:.2f}",
            ha="center",
            va="bottom",
            fontsize=7.5,
        )

    axB.set_title("Error increases with uncertainty group")
    axB.set_xlabel(r"Uncertainty group from $\sigma_\Delta$")
    axB.set_ylabel(r"Absolute $\Delta$ error (eV)")
    axB.set_xticks([1, 2, 3])
    axB.set_xticklabels(["low", "medium", "high"])
    axB.grid(axis="y", color=PALETTE["light"], alpha=0.35)
    axB.text(
        0.02,
        0.98,
        "B",
        transform=axB.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
    )

    y_max = max(np.nanmax(err) + 0.10, 0.5)
    axA.set_ylim(0, y_max)
    axB.set_ylim(0, y_max)

    fig.suptitle(args.title, fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout(w_pad=1.2)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_base = out_dir / args.out_prefix

    for ext in [".png", ".pdf", ".svg"]:
        fig.savefig(out_base.with_suffix(ext), dpi=args.dpi, bbox_inches="tight")

    plt.close(fig)

    overall = pd.DataFrame([{
        "cycle": args.cycle,
        "n_validation_points": len(data),
        "spearman_uncertainty_error": rho,
        "coverage_1sigma": cov1,
        "coverage_2sigma": cov2,
        "mean_abs_delta_error": float(np.mean(err)),
        "median_abs_delta_error": float(np.median(err)),
    }])

    tertile_summary = (
        data_t.groupby("uncertainty_tertile", observed=False)
        .agg(
            n=("abs_delta_error", "size"),
            sigma_mean=("sigma_delta", "mean"),
            sigma_median=("sigma_delta", "median"),
            abs_delta_error_mean=("abs_delta_error", "mean"),
            abs_delta_error_median=("abs_delta_error", "median"),
            abs_delta_error_std=("abs_delta_error", "std"),
        )
        .reset_index()
    )

    data_path = out_base.with_name(out_base.name + "_pooled_points.csv")
    per_run_path = out_base.with_name(out_base.name + "_per_run.csv")
    overall_path = out_base.with_name(out_base.name + "_overall.csv")
    tertile_path = out_base.with_name(out_base.name + "_tertiles.csv")

    data_t.to_csv(data_path, index=False)
    per_run.to_csv(per_run_path, index=False)
    overall.to_csv(overall_path, index=False)
    tertile_summary.to_csv(tertile_path, index=False)

    print("[OK] wrote:")
    print(" ", out_base.with_suffix(".png"))
    print(" ", out_base.with_suffix(".pdf"))
    print(" ", out_base.with_suffix(".svg"))
    print(" ", data_path)
    print(" ", per_run_path)
    print(" ", overall_path)
    print(" ", tertile_path)
    print()
    print("Overall:")
    print(overall.to_string(index=False))
    print()
    print("Uncertainty tertiles:")
    print(tertile_summary.to_string(index=False))


def main():
    args = parse_args()
    data, per_run = collect_uncertainty_data(args)
    make_plot(data, per_run, args)


if __name__ == "__main__":
    main()
