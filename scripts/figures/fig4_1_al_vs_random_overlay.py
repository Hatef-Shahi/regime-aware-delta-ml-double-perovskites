#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Publication figure: AL vs quota-matched random baseline.

This script compares two groups of completed AL-cycle runs at a fixed budget.
It is intended for the final paper workflow where both AL and random use the
same quota structure, e.g. K_g=2 and K_r=5.

Panels:
  A) End-to-end HSE MAE vs cycle
  B) Stage-B/non-PBE-ZBG Delta MAE vs cycle

The plotted lines are the across-replicate mean at each cycle. Shaded bands are
across-replicate standard deviations at each cycle. This explanation is not
printed inside the figure; put it in the caption.

Outputs:
  <out-dir>/<out-prefix>.pdf/.svg/.png
  <out-dir>/<out-prefix>_fixed_cycle_summary.csv/.md
  <out-dir>/<out-prefix>_cyclewise_stability.csv/.md
  <out-dir>/<out-prefix>_metadata.json

Example:
  python3 plots/fig4_1_al_vs_random_overlay.py \
    --root . \
    --al-dir paper-runs/delta_kg2_m6_q035_theta016_dosOFF \
    --al-glob 'al*' \
    --random-dir paper-runs/delta_kg2_m6_q035_theta016_dosOFF \
    --random-glob 'random*' \
    --match-cycle 24 \
    --seed-labels 100 \
    --batch-size 7 \
    --al-label 'AL' \
    --random-label 'Quota-random' \
    --title 'AL vs quota-matched random selection' \
    --out-dir artifacts/pub_final_figures/fig2_al_vs_quota_random_kg2_m6_q035_dosOFF \
    --out-prefix fig2_al_vs_quota_random \
    --dpi 600
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PALETTE: Dict[str, str] = {
    "navy": "#0D1164",
    "purple": "#8C00FF",
    "pink": "#FF3F7F",
    "green": "#638C6D",
    "ink": "#111111",
    "light": "#DDDDDD",
    "midgray": "#A8A8A8",
    "bg": "#FFFFFF",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Publication AL vs quota-matched random plot."
    )
    p.add_argument("--root", type=str, default=".", help="Experiments root")
    p.add_argument("--al-dir", type=str, required=True, help="Directory containing AL replicate folders")
    p.add_argument("--random-dir", type=str, required=True, help="Directory containing random replicate folders")
    p.add_argument("--al-glob", type=str, default="al*", help="Glob for AL run folders")
    p.add_argument("--random-glob", type=str, default="random*", help="Glob for random run folders")
    p.add_argument("--metrics-file", type=str, default="metrics/metrics.csv")

    p.add_argument("--col-end", type=str, default="end2end_hse_MAE")
    p.add_argument("--col-del", type=str, default="delta_val_MAE_nonmetal")

    p.add_argument("--al-label", type=str, default="AL")
    p.add_argument("--random-label", type=str, default="Quota-random")
    p.add_argument("--title", type=str, default="AL vs quota-matched random selection")

    p.add_argument("--out-dir", type=str, default="artifacts/fig_al_vs_quota_random")
    p.add_argument("--out-prefix", type=str, default="fig_al_vs_quota_random")

    p.add_argument("--match-cycle", type=int, default=24)
    p.add_argument("--seed-labels", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=7)

    p.add_argument("--dpi", type=int, default=600)
    p.add_argument("--figwidth", type=float, default=7.2)
    p.add_argument("--figheight", type=float, default=5.8)
    p.add_argument("--font", type=str, default="")

    p.add_argument("--grid", action="store_true", default=True)
    p.add_argument("--no-grid", action="store_true")
    p.add_argument("--no-table", action="store_true")
    p.add_argument("--no-inset", action="store_true", help="Do not show cycle-24 summary inset")
    p.add_argument("--legend-loc-a", type=str, default="upper center")
    p.add_argument("--legend-loc-b", type=str, default="upper center")
    p.add_argument("--inset-decimals", type=int, default=3)
    return p.parse_args()


def style_matplotlib(font: str = "") -> None:
    rc = {
        "figure.facecolor": PALETTE["bg"],
        "axes.facecolor": PALETTE["bg"],
        "axes.edgecolor": PALETTE["ink"],
        "axes.labelcolor": PALETTE["ink"],
        "xtick.color": PALETTE["ink"],
        "ytick.color": PALETTE["ink"],
        "text.color": PALETTE["ink"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.linewidth": 1.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
    }
    if font.strip():
        rc["font.family"] = font.strip()
    plt.rcParams.update(rc)


def full_frame(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(PALETTE["ink"])
        spine.set_linewidth(1.0)


def _norm_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def find_column(df: pd.DataFrame, expected: str) -> Optional[str]:
    exp = _norm_name(expected)
    mapping = {_norm_name(c): c for c in df.columns}
    if exp in mapping:
        return mapping[exp]

    tokens = [t for t in re.split(r"[_\W]+", expected.lower()) if t]
    for c in df.columns:
        cc = c.lower()
        if all(t in cc for t in tokens):
            return c
    return None


def discover_run_dirs(base: Path, pattern: str) -> List[Path]:
    if not base.exists():
        raise FileNotFoundError(f"Directory does not exist: {base}")
    runs = [p for p in sorted(base.glob(pattern)) if p.is_dir()]
    if not runs:
        raise RuntimeError(f"No run directories found under {base} matching {pattern!r}")
    return runs


def resolve_metrics_path(run_dir: Path, rel_metrics_file: str) -> Path:
    path = run_dir / rel_metrics_file
    if path.exists():
        return path
    # Last-resort search, but only inside the requested run directory.
    candidates = sorted(run_dir.glob("**/metrics/metrics.csv"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"No metrics file found for run {run_dir}; tried {path}")


def load_metrics(run_dir: Path, rel_metrics_file: str, col_end: str, col_del: str) -> pd.DataFrame:
    path = resolve_metrics_path(run_dir, rel_metrics_file)
    df = pd.read_csv(path)

    cyc_col = find_column(df, "cycle") or find_column(df, "al_cycle") or find_column(df, "iteration")
    if cyc_col is None:
        df["cycle"] = np.arange(len(df), dtype=float)
        cyc_col = "cycle"

    end_col = find_column(df, col_end) or (col_end if col_end in df.columns else None)
    del_col = find_column(df, col_del) or (col_del if col_del in df.columns else None)

    if end_col is None or del_col is None:
        raise KeyError(
            f"Missing required metric columns in {path}\n"
            f"Need: {col_end}, {col_del}\n"
            f"Have: {list(df.columns)}"
        )

    out = pd.DataFrame(
        {
            "cycle": pd.to_numeric(df[cyc_col], errors="coerce"),
            "end2end": pd.to_numeric(df[end_col], errors="coerce"),
            "delta": pd.to_numeric(df[del_col], errors="coerce"),
        }
    )
    out = out[np.isfinite(out["cycle"].to_numpy(float))].copy()
    out = out.sort_values("cycle")
    return out


def load_group(run_dirs: List[Path], rel_metrics_file: str, col_end: str, col_del: str) -> Dict[str, pd.DataFrame]:
    return {rd.name: load_metrics(rd, rel_metrics_file, col_end, col_del) for rd in run_dirs}


def to_wide(dfs: Dict[str, pd.DataFrame], value_col: str) -> pd.DataFrame:
    series = []
    for name, df in dfs.items():
        s = pd.Series(
            df[value_col].to_numpy(float),
            index=pd.Index(df["cycle"].to_numpy(float), name="cycle"),
            name=name,
        )
        series.append(s)
    wide = pd.concat(series, axis=1).sort_index()
    wide.index.name = "cycle"
    return wide


def truncate_wide_at_cycle(wide: pd.DataFrame, cycle: int) -> pd.DataFrame:
    idx = np.asarray(wide.index, dtype=float)
    return wide.loc[idx <= float(cycle)].copy()


def mean_std_n(wide: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = wide.index.to_numpy(float)
    n = wide.notna().sum(axis=1).to_numpy(int)
    mean = wide.mean(axis=1, skipna=True).to_numpy(float)
    std = wide.std(axis=1, ddof=1, skipna=True).to_numpy(float)
    return x, mean, std, n


def values_at_cycle(wide: pd.DataFrame, cycle: int) -> np.ndarray:
    idx = np.asarray(wide.index, dtype=float)
    matches = np.where(np.isclose(idx, float(cycle)))[0]
    if matches.size == 0:
        raise ValueError(f"Cycle {cycle} not found. Available cycles: {list(wide.index.to_numpy())}")
    return wide.iloc[matches[0]].dropna().to_numpy(float)


def mean_std_values(vals: np.ndarray) -> Tuple[float, float, int]:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan"), float("nan"), 0
    sd = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
    return float(np.mean(vals)), sd, int(vals.size)


def ylims_from_mean_std(mean: np.ndarray, std: np.ndarray) -> Tuple[float, float]:
    mean = np.asarray(mean, dtype=float)
    std = np.asarray(std, dtype=float)
    mask = np.isfinite(mean)
    if not np.any(mask):
        return 0.0, 1.0
    lo = float(np.nanmin(mean[mask]))
    hi = float(np.nanmax(mean[mask]))
    band = np.isfinite(std)
    if np.any(band):
        lo = min(lo, float(np.nanmin((mean - std)[band])))
        hi = max(hi, float(np.nanmax((mean + std)[band])))
    pad = 0.07 * (hi - lo) if hi > lo else 0.1
    return max(0.0, lo - pad), hi + pad


def plot_mean_std(
    ax: plt.Axes,
    x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    line_color: str,
    band_color: str,
    label: str,
    linestyle: str,
    band_alpha: float,
    lw: float = 2.3,
    ms: float = 3.0,
) -> None:
    valid = np.isfinite(mean) & np.isfinite(std)
    if np.any(valid):
        ax.fill_between(
            x,
            np.where(valid, mean - std, np.nan),
            np.where(valid, mean + std, np.nan),
            color=band_color,
            alpha=band_alpha,
            linewidth=0,
            zorder=1,
        )
    ax.plot(
        x,
        mean,
        color=line_color,
        linewidth=lw,
        marker="o",
        markersize=ms,
        linestyle=linestyle,
        label=label,
        zorder=3,
    )


def add_cycle_marker(ax: plt.Axes, match_cycle: int) -> None:
    ax.axvline(
        match_cycle,
        color=PALETTE["ink"],
        linestyle=":",
        linewidth=0.9,
        alpha=0.95,
        zorder=2,
    )
    ax.text(
        match_cycle,
        0.965,
        f"cycle {match_cycle}",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=8,
        color=PALETTE["ink"],
        bbox=dict(facecolor=PALETTE["bg"], edgecolor="none", alpha=0.78, pad=1.2),
        zorder=4,
    )


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.012,
        0.955,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
    )


def add_cycle_inset(
    ax: plt.Axes,
    *,
    match_cycle: int,
    al_label: str,
    random_label: str,
    al_mu: float,
    al_sd: float,
    rnd_mu: float,
    rnd_sd: float,
    decimals: int,
    loc: str,
) -> None:
    fmt = f"{{:.{decimals}f}}"
    text = (
        f"cycle {match_cycle}:\n"
        f"{al_label}: {fmt.format(al_mu)} ± {fmt.format(al_sd)}\n"
        f"{random_label}: {fmt.format(rnd_mu)} ± {fmt.format(rnd_sd)}"
    )

    if loc == "lower left":
        x, y, ha, va = 0.025, 0.08, "left", "bottom"
    elif loc == "lower right":
        x, y, ha, va = 0.975, 0.08, "right", "bottom"
    elif loc == "upper left":
        x, y, ha, va = 0.025, 0.88, "left", "top"
    else:
        x, y, ha, va = 0.975, 0.88, "right", "top"

    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=8,
        linespacing=1.15,
        bbox=dict(facecolor=PALETTE["bg"], edgecolor=PALETTE["light"], alpha=0.82, pad=2.0),
        zorder=5,
    )


def df_to_md_table(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for _, row in df.iterrows():
        vals = []
        for c in headers:
            v = row[c]
            if isinstance(v, float):
                vals.append(f"{v:.6g}" if np.isfinite(v) else "—")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def fixed_cycle_summary(
    al_end: pd.DataFrame,
    al_del: pd.DataFrame,
    rnd_end: pd.DataFrame,
    rnd_del: pd.DataFrame,
    match_cycle: int,
    seed_labels: int,
    batch_size: int,
    al_label: str,
    random_label: str,
) -> pd.DataFrame:
    acquired = match_cycle * batch_size
    total = seed_labels + acquired

    rows = []
    for method, end_wide, del_wide in [
        (al_label, al_end, al_del),
        (random_label, rnd_end, rnd_del),
    ]:
        e_vals = values_at_cycle(end_wide, match_cycle)
        d_vals = values_at_cycle(del_wide, match_cycle)
        e_mu, e_sd, e_n = mean_std_values(e_vals)
        d_mu, d_sd, d_n = mean_std_values(d_vals)
        rows.append(
            {
                "method": method,
                "n_runs": min(e_n, d_n),
                "fixed_cycle": match_cycle,
                "seed_labels": seed_labels,
                "acquired_labels": acquired,
                "total_labels": total,
                "end2end_hse_MAE_mean": e_mu,
                "end2end_hse_MAE_sigma": e_sd,
                "delta_MAE_non_PBE_ZBG_mean": d_mu,
                "delta_MAE_non_PBE_ZBG_sigma": d_sd,
            }
        )

    # Add absolute and relative improvements, Random - AL.
    if len(rows) == 2:
        al = rows[0]
        rnd = rows[1]
        rows.append(
            {
                "method": f"{random_label} - {al_label}",
                "n_runs": "paired groups",
                "fixed_cycle": match_cycle,
                "seed_labels": seed_labels,
                "acquired_labels": acquired,
                "total_labels": total,
                "end2end_hse_MAE_mean": rnd["end2end_hse_MAE_mean"] - al["end2end_hse_MAE_mean"],
                "end2end_hse_MAE_sigma": np.nan,
                "delta_MAE_non_PBE_ZBG_mean": rnd["delta_MAE_non_PBE_ZBG_mean"] - al["delta_MAE_non_PBE_ZBG_mean"],
                "delta_MAE_non_PBE_ZBG_sigma": np.nan,
            }
        )
    return pd.DataFrame(rows)


def cyclewise_stability_summary(
    al_end_plot: pd.DataFrame,
    rnd_end_plot: pd.DataFrame,
    al_del_plot: pd.DataFrame,
    rnd_del_plot: pd.DataFrame,
    al_label: str,
    random_label: str,
) -> pd.DataFrame:
    _, _, sAE, _ = mean_std_n(al_end_plot)
    _, _, sRE, _ = mean_std_n(rnd_end_plot)
    _, _, sAD, _ = mean_std_n(al_del_plot)
    _, _, sRD, _ = mean_std_n(rnd_del_plot)

    def _mean_std_of_std(x: np.ndarray) -> Tuple[float, float]:
        vals = np.asarray(x, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return np.nan, np.nan
        return float(np.mean(vals)), float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0

    ae_mu, ae_sd = _mean_std_of_std(sAE)
    re_mu, re_sd = _mean_std_of_std(sRE)
    ad_mu, ad_sd = _mean_std_of_std(sAD)
    rd_mu, rd_sd = _mean_std_of_std(sRD)

    rows = [
        {"metric": "end2end_hse_MAE", "method": al_label, "mean_cyclewise_sigma": ae_mu, "sigma_of_cyclewise_sigma": ae_sd},
        {"metric": "end2end_hse_MAE", "method": random_label, "mean_cyclewise_sigma": re_mu, "sigma_of_cyclewise_sigma": re_sd},
        {"metric": "delta_MAE_non_PBE_ZBG", "method": al_label, "mean_cyclewise_sigma": ad_mu, "sigma_of_cyclewise_sigma": ad_sd},
        {"metric": "delta_MAE_non_PBE_ZBG", "method": random_label, "mean_cyclewise_sigma": rd_mu, "sigma_of_cyclewise_sigma": rd_sd},
        {
            "metric": "end2end_hse_MAE",
            "method": f"stability_gain_{al_label}_vs_{random_label}",
            "mean_cyclewise_sigma": 1.0 - ae_mu / re_mu if np.isfinite(ae_mu) and np.isfinite(re_mu) and re_mu > 0 else np.nan,
            "sigma_of_cyclewise_sigma": np.nan,
        },
        {
            "metric": "delta_MAE_non_PBE_ZBG",
            "method": f"stability_gain_{al_label}_vs_{random_label}",
            "mean_cyclewise_sigma": 1.0 - ad_mu / rd_mu if np.isfinite(ad_mu) and np.isfinite(rd_mu) and rd_mu > 0 else np.nan,
            "sigma_of_cyclewise_sigma": np.nan,
        },
    ]
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    out_dir = root / args.out_dir
    show_grid = bool(args.grid) and not bool(args.no_grid)

    style_matplotlib(args.font)

    al_base = root / args.al_dir
    rnd_base = root / args.random_dir
    al_run_dirs = discover_run_dirs(al_base, args.al_glob)
    rnd_run_dirs = discover_run_dirs(rnd_base, args.random_glob)

    al_dfs = load_group(al_run_dirs, args.metrics_file, args.col_end, args.col_del)
    rnd_dfs = load_group(rnd_run_dirs, args.metrics_file, args.col_end, args.col_del)

    al_end = to_wide(al_dfs, "end2end")
    al_del = to_wide(al_dfs, "delta")
    rnd_end = to_wide(rnd_dfs, "end2end")
    rnd_del = to_wide(rnd_dfs, "delta")

    match_cycle = int(args.match_cycle)

    # Fixed-cycle summary from full wide data.
    summary_df = fixed_cycle_summary(
        al_end,
        al_del,
        rnd_end,
        rnd_del,
        match_cycle,
        int(args.seed_labels),
        int(args.batch_size),
        args.al_label,
        args.random_label,
    )

    # Plot only through matched cycle.
    al_end_plot = truncate_wide_at_cycle(al_end, match_cycle)
    al_del_plot = truncate_wide_at_cycle(al_del, match_cycle)
    rnd_end_plot = truncate_wide_at_cycle(rnd_end, match_cycle)
    rnd_del_plot = truncate_wide_at_cycle(rnd_del, match_cycle)

    xAE, mAE, sAE, _ = mean_std_n(al_end_plot)
    xAD, mAD, sAD, _ = mean_std_n(al_del_plot)
    xRE, mRE, sRE, _ = mean_std_n(rnd_end_plot)
    xRD, mRD, sRD, _ = mean_std_n(rnd_del_plot)

    # Cycle-24 values for insets.
    al_end_mu, al_end_sd, _ = mean_std_values(values_at_cycle(al_end, match_cycle))
    rnd_end_mu, rnd_end_sd, _ = mean_std_values(values_at_cycle(rnd_end, match_cycle))
    al_del_mu, al_del_sd, _ = mean_std_values(values_at_cycle(al_del, match_cycle))
    rnd_del_mu, rnd_del_sd, _ = mean_std_values(values_at_cycle(rnd_del, match_cycle))

    stability_df = cyclewise_stability_summary(
        al_end_plot,
        rnd_end_plot,
        al_del_plot,
        rnd_del_plot,
        args.al_label,
        args.random_label,
    )

    fig, (axA, axB) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(args.figwidth, args.figheight),
        sharex=True,
        constrained_layout=False,
    )

    # Panel A: end-to-end HSE MAE.
    plot_mean_std(
        axA,
        xAE,
        mAE,
        sAE,
        line_color=PALETTE["navy"],
        band_color=PALETTE["navy"],
        label=f"{args.al_label} (n={len(al_dfs)})",
        linestyle="-",
        band_alpha=0.18,
        lw=2.4,
        ms=3.2,
    )
    plot_mean_std(
        axA,
        xRE,
        mRE,
        sRE,
        line_color=PALETTE["purple"],
        band_color=PALETTE["light"],
        label=f"{args.random_label} (n={len(rnd_dfs)})",
        linestyle="--",
        band_alpha=0.38,
        lw=2.1,
        ms=3.0,
    )
    axA.set_ylabel("End-to-end HSE MAE (eV)")
    add_panel_label(axA, "A")

    # Panel B: non-PBE-ZBG Delta MAE.
    plot_mean_std(
        axB,
        xAD,
        mAD,
        sAD,
        line_color=PALETTE["pink"],
        band_color=PALETTE["pink"],
        label=f"{args.al_label} (n={len(al_dfs)})",
        linestyle="-",
        band_alpha=0.18,
        lw=2.4,
        ms=3.2,
    )
    plot_mean_std(
        axB,
        xRD,
        mRD,
        sRD,
        line_color=PALETTE["green"],
        band_color=PALETTE["light"],
        label=f"{args.random_label} (n={len(rnd_dfs)})",
        linestyle="--",
        band_alpha=0.38,
        lw=2.1,
        ms=3.0,
    )
    axB.set_ylabel("Stage-B Δ-MAE\n(non-PBE-ZBG, eV)")
    axB.set_xlabel("Cycle")
    add_panel_label(axB, "B")

    for ax in (axA, axB):
        add_cycle_marker(ax, match_cycle)
        full_frame(ax)
        if show_grid:
            ax.grid(True, which="major", axis="both", color=PALETTE["light"], linewidth=0.7, alpha=0.6)
        ax.tick_params(length=3.5, width=1.0)

    axA.legend(frameon=True, facecolor=PALETTE["bg"], edgecolor=PALETTE["light"], loc=args.legend_loc_a)
    axB.legend(frameon=True, facecolor=PALETTE["bg"], edgecolor=PALETTE["light"], loc=args.legend_loc_b)

    if not args.no_inset:
        add_cycle_inset(
            axA,
            match_cycle=match_cycle,
            al_label=args.al_label,
            random_label=args.random_label,
            al_mu=al_end_mu,
            al_sd=al_end_sd,
            rnd_mu=rnd_end_mu,
            rnd_sd=rnd_end_sd,
            decimals=args.inset_decimals,
            loc="lower left",
        )
        add_cycle_inset(
            axB,
            match_cycle=match_cycle,
            al_label=args.al_label,
            random_label=args.random_label,
            al_mu=al_del_mu,
            al_sd=al_del_sd,
            rnd_mu=rnd_del_mu,
            rnd_sd=rnd_del_sd,
            decimals=args.inset_decimals,
            loc="lower left",
        )

    axA.set_ylim(*ylims_from_mean_std(np.r_[mAE, mRE], np.r_[sAE, sRE]))
    axB.set_ylim(*ylims_from_mean_std(np.r_[mAD, mRD], np.r_[sAD, sRD]))

    fig.suptitle(args.title, y=0.995, fontsize=10.5, fontweight="bold")
    fig.subplots_adjust(left=0.12, right=0.985, top=0.93, bottom=0.10, hspace=0.19)

    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / args.out_prefix

    fig.savefig(str(base) + ".pdf", bbox_inches="tight")
    fig.savefig(str(base) + ".svg", bbox_inches="tight")
    fig.savefig(str(base) + ".png", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "root": str(root),
        "al_dir": str(al_base),
        "random_dir": str(rnd_base),
        "al_glob": args.al_glob,
        "random_glob": args.random_glob,
        "n_al_runs": len(al_run_dirs),
        "n_random_runs": len(rnd_run_dirs),
        "match_cycle": match_cycle,
        "seed_labels": int(args.seed_labels),
        "batch_size": int(args.batch_size),
        "acquired_labels": int(match_cycle * args.batch_size),
        "total_labels": int(args.seed_labels + match_cycle * args.batch_size),
        "caption_note": (
            "Lines show across-replicate means; shaded regions show across-replicate "
            "standard deviations at each cycle. Cycle-24 insets report mean ± standard deviation."
        ),
    }

    if not args.no_table:
        summary_csv = out_dir / f"{args.out_prefix}_fixed_cycle_summary.csv"
        summary_md = out_dir / f"{args.out_prefix}_fixed_cycle_summary.md"
        stability_csv = out_dir / f"{args.out_prefix}_cyclewise_stability.csv"
        stability_md = out_dir / f"{args.out_prefix}_cyclewise_stability.md"

        summary_df.to_csv(summary_csv, index=False)
        stability_df.to_csv(stability_csv, index=False)
        summary_md.write_text(df_to_md_table(summary_df), encoding="utf-8")
        stability_md.write_text(df_to_md_table(stability_df), encoding="utf-8")

    (out_dir / f"{args.out_prefix}_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"[OK] Figure: {base}.pdf/.svg/.png")
    if not args.no_table:
        print(f"[OK] Fixed-cycle table: {out_dir / (args.out_prefix + '_fixed_cycle_summary.csv')}")
        print(f"[OK] Stability table:   {out_dir / (args.out_prefix + '_cyclewise_stability.csv')}")

    print("\n[cycle-24 summary]")
    print(summary_df.to_string(index=False))
    print("\n[cycle-wise replicate disagreement]")
    print(stability_df.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
