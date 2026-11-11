#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Publication figure: Δ-ML AL vs direct-HSE AL target ablation.

Compares two active-learning target formulations under the same fixed-cycle budget:
  1) Δ-ML AL:       Stage-B regression target = HSE - PBE
  2) Direct-HSE AL: Stage-B regression target = HSE

Expected final-paper layout from experiments root:
  paper-runs/delta_kg2_m6_q035_theta016_dosOFF/al1 ... al5/
  paper-runs/delta_kg2_m6_q035_theta016_dosOFF/direct1 ... direct5/

Each run directory must contain:
  metrics/metrics.csv

Panels:
  A) End-to-end HSE MAE vs cycle, mean ± SD over replicates
  B) Stage-B Δ-MAE on non-PBE-ZBG subset vs cycle, mean ± SD over replicates

Outputs:
  <out-dir>/<out-prefix>.pdf/.svg/.png
  <out-dir>/<out-prefix>_fixed_budget_cycleXX.csv/.md
  <out-dir>/<out-prefix>_group_summary.csv/.md
  <out-dir>/<out-prefix>_cyclewise_stability.csv/.md
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


PALETTE: Dict[str, str] = {
    "navy": "#003566",
    "purple": "#8C00FF",
    "pink": "#ff4d6d",
    "green": "#2d6a4f",
    "ink": "#111111",
    "light": "#DDDDDD",
    "bg": "#FFFFFF",
}


@dataclass(frozen=True)
class GroupSummary:
    group: str
    n_runs: int
    cycles_max: int
    end2end_mae_last3_mean: float
    end2end_mae_final: float
    delta_mae_last3_mean: float
    delta_mae_final: float
    end2end_rmse_last3_mean: float
    end2end_rmse_final: float
    end2end_r2_last3_mean: float
    end2end_r2_final: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Δ-ML AL vs direct-HSE AL target ablation with Δ-MAE panel"
    )
    p.add_argument("--root", type=str, default=".", help="Root containing run directories")

    p.add_argument("--delta-dir", type=str, required=True, help="Base dir for Δ-ML AL runs, relative to root")
    p.add_argument("--direct-dir", type=str, required=True, help="Base dir for direct-HSE AL runs, relative to root")
    p.add_argument("--delta-glob", type=str, default="al*", help="Glob for Δ-ML run dirs")
    p.add_argument("--direct-glob", type=str, default="direct*", help="Glob for direct-HSE run dirs")
    p.add_argument("--metrics-file", type=str, default="metrics/metrics.csv")

    p.add_argument("--col-mae", type=str, default="end2end_hse_MAE")
    p.add_argument("--col-delta", type=str, default="delta_val_MAE_nonmetal")
    p.add_argument("--col-rmse", type=str, default="end2end_hse_RMSE")
    p.add_argument("--col-r2", type=str, default="end2end_hse_R2")

    p.add_argument("--out-dir", type=str, default="artifacts/gap3_delta_vs_direct_hse")
    p.add_argument("--out-prefix", type=str, default="fig3_delta_vs_direct_hse")
    p.add_argument("--last-n", type=int, default=3)
    p.add_argument("--no-table", action="store_true")

    p.add_argument("--match-cycle", type=int, default=24)
    p.add_argument("--seed-labels", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=7)

    p.add_argument("--delta-label", type=str, default="Δ-ML AL")
    p.add_argument("--direct-label", type=str, default="Direct-HSE AL")
    p.add_argument("--title", type=str, default="Δ-ML vs direct-HSE active learning")

    p.add_argument("--grid", action="store_true")
    p.add_argument("--no-grid", action="store_true")
    p.add_argument("--dpi", type=int, default=600)
    p.add_argument("--figwidth", type=float, default=7.2)
    p.add_argument("--figheight", type=float, default=5.8)
    p.add_argument("--font", type=str, default="")
    p.add_argument("--show-final-insets", action="store_true", default=True)
    p.add_argument("--no-final-insets", action="store_false", dest="show_final_insets")
    p.add_argument("--y-major-step", type=float, default=0.1, help="Major y tick spacing for both panels")
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
        cc = str(c).lower()
        if all(t in cc for t in tokens):
            return c
    return None


def discover_run_dirs(base: Path, pattern: str) -> List[Path]:
    if not base.exists():
        raise FileNotFoundError(str(base))
    return [p for p in sorted(base.glob(pattern)) if p.is_dir()]


def _pick_best_candidate(run_dir: Path, candidates: List[Path]) -> Optional[Path]:
    if not candidates:
        return None
    scored = [(c, len(c.relative_to(run_dir).parts)) for c in candidates]
    scored.sort(key=lambda t: (t[1], str(t[0])))
    return scored[0][0]


def resolve_metrics_path(run_dir: Path, rel_metrics_file: str) -> Path:
    direct = run_dir / rel_metrics_file
    if direct.exists():
        return direct

    candidates: List[Path] = []
    candidates.extend(run_dir.glob("**/metrics/metrics.csv"))
    candidates.extend(run_dir.glob("**/metrics.csv"))
    best = _pick_best_candidate(run_dir, [c for c in candidates if c.is_file()])
    if best is None:
        raise FileNotFoundError(f"No metrics file found under {run_dir}")
    return best


def load_metrics(
    run_dir: Path,
    rel_metrics_file: str,
    col_mae: str,
    col_delta: str,
    col_rmse: str,
    col_r2: str,
) -> pd.DataFrame:
    path = resolve_metrics_path(run_dir, rel_metrics_file)
    df = pd.read_csv(path)

    cyc_col = find_column(df, "cycle") or find_column(df, "al_cycle") or find_column(df, "iteration")
    if cyc_col is None:
        df["cycle"] = np.arange(len(df), dtype=float)
        cyc_col = "cycle"
    else:
        df[cyc_col] = pd.to_numeric(df[cyc_col], errors="coerce")

    mae_col = find_column(df, col_mae) or (col_mae if col_mae in df.columns else None)
    delta_col = find_column(df, col_delta) or (col_delta if col_delta in df.columns else None)
    rmse_col = find_column(df, col_rmse) or (col_rmse if col_rmse in df.columns else None)
    r2_col = find_column(df, col_r2) or (col_r2 if col_r2 in df.columns else None)

    if mae_col is None or delta_col is None or rmse_col is None or r2_col is None:
        raise KeyError(
            f"Missing required columns in {path}\n"
            f"Need: {col_mae}, {col_delta}, {col_rmse}, {col_r2}\n"
            f"Have: {list(df.columns)}"
        )

    out = pd.DataFrame({
        "cycle": pd.to_numeric(df[cyc_col], errors="coerce"),
        "mae": pd.to_numeric(df[mae_col], errors="coerce"),
        "delta": pd.to_numeric(df[delta_col], errors="coerce"),
        "rmse": pd.to_numeric(df[rmse_col], errors="coerce"),
        "r2": pd.to_numeric(df[r2_col], errors="coerce"),
    })
    out = out[np.isfinite(out["cycle"].to_numpy(float))].copy()
    return out.sort_values("cycle")


def load_group(
    run_dirs: List[Path],
    rel_metrics_file: str,
    col_mae: str,
    col_delta: str,
    col_rmse: str,
    col_r2: str,
) -> Dict[str, pd.DataFrame]:
    return {
        rd.name: load_metrics(rd, rel_metrics_file, col_mae, col_delta, col_rmse, col_r2)
        for rd in run_dirs
    }


def to_wide(dfs: Dict[str, pd.DataFrame], value_col: str) -> pd.DataFrame:
    series = []
    for name, df in dfs.items():
        series.append(pd.Series(
            df[value_col].to_numpy(float),
            index=pd.Index(df["cycle"].to_numpy(float), name="cycle"),
            name=name,
        ))
    wide = pd.concat(series, axis=1).sort_index()
    wide.index.name = "cycle"
    return wide


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
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan"), float("nan"), 0
    mu = float(np.mean(vals))
    sd = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
    return mu, sd, int(vals.size)


def truncate_wide_at_cycle(wide: pd.DataFrame, cycle: int) -> pd.DataFrame:
    idx = np.asarray(wide.index, dtype=float)
    return wide.loc[idx <= float(cycle)].copy()


def ylims_from_mean_std(mean: np.ndarray, std: np.ndarray, step: float = 0.1) -> Tuple[float, float]:
    m = np.isfinite(mean)
    s = np.isfinite(std)
    if not np.any(m):
        return (0.0, 1.0)
    lo = float(np.min(mean[m]))
    hi = float(np.max(mean[m]))
    if np.any(s):
        lo = min(lo, float(np.min((mean - std)[s])))
        hi = max(hi, float(np.max((mean + std)[s])))

    pad = 0.06 * (hi - lo) if hi > lo else step
    lo = max(0.0, lo - pad)
    hi = hi + pad

    # Expand to clean 0.1 tick boundaries.
    if step > 0:
        lo = max(0.0, np.floor(lo / step) * step)
        hi = np.ceil(hi / step) * step
    return (float(lo), float(hi))


def mean_last_n(y: np.ndarray, n: int) -> float:
    y = y[np.isfinite(y)]
    if y.size == 0:
        return float("nan")
    return float(np.mean(y[-min(int(n), y.size):]))


def last_value(y: np.ndarray) -> float:
    y = y[np.isfinite(y)]
    return float(y[-1]) if y.size else float("nan")


def df_to_md_table(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    rows = df.astype(str).values.tolist()
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def fixed_row(
    method: str,
    mae_wide: pd.DataFrame,
    delta_wide: pd.DataFrame,
    rmse_wide: pd.DataFrame,
    r2_wide: pd.DataFrame,
    match_cycle: int,
    acquired_labels: int,
    seed_labels: int,
    total_labels: int,
) -> Dict[str, object]:
    mae_mu, mae_sd, mae_n = mean_std_values(values_at_cycle(mae_wide, match_cycle))
    delta_mu, delta_sd, delta_n = mean_std_values(values_at_cycle(delta_wide, match_cycle))
    rmse_mu, rmse_sd, rmse_n = mean_std_values(values_at_cycle(rmse_wide, match_cycle))
    r2_mu, r2_sd, r2_n = mean_std_values(values_at_cycle(r2_wide, match_cycle))
    return {
        "method": method,
        "n_runs": min(mae_n, delta_n, rmse_n, r2_n),
        "fixed_cycle": int(match_cycle),
        "acquired_labels": int(acquired_labels),
        "seed_labels": int(seed_labels),
        "total_hse06_labels": int(total_labels),
        "end2end_hse_MAE_mean": mae_mu,
        "end2end_hse_MAE_std": mae_sd,
        "delta_val_MAE_nonmetal_mean": delta_mu,
        "delta_val_MAE_nonmetal_std": delta_sd,
        "end2end_hse_RMSE_mean": rmse_mu,
        "end2end_hse_RMSE_std": rmse_sd,
        "end2end_hse_R2_mean": r2_mu,
        "end2end_hse_R2_std": r2_sd,
    }


def make_fixed_budget_table(
    delta: Dict[str, pd.DataFrame],
    direct: Dict[str, pd.DataFrame],
    match_cycle: int,
    acquired_labels: int,
    seed_labels: int,
    total_labels: int,
    delta_label: str,
    direct_label: str,
) -> pd.DataFrame:
    return pd.DataFrame([
        fixed_row(delta_label, delta["mae"], delta["delta"], delta["rmse"], delta["r2"], match_cycle, acquired_labels, seed_labels, total_labels),
        fixed_row(direct_label, direct["mae"], direct["delta"], direct["rmse"], direct["r2"], match_cycle, acquired_labels, seed_labels, total_labels),
    ])


def add_match_cycle_marker(ax: plt.Axes, match_cycle: int) -> None:
    ax.axvline(match_cycle, color=PALETTE["ink"], linestyle=":", linewidth=1.1, alpha=0.95, zorder=5)
    ax.text(
        match_cycle,
        0.96,
        f"cycle {match_cycle}",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=8,
        color=PALETTE["ink"],
        bbox=dict(facecolor=PALETTE["bg"], edgecolor="none", alpha=0.78, pad=1.5),
        zorder=6,
    )


def summarize_group(name: str, dfs: Dict[str, pd.DataFrame], last_n: int) -> GroupSummary:
    cycles_max = -1
    mae_last, mae_final, delta_last, delta_final = [], [], [], []
    rmse_last, rmse_final, r2_last, r2_final = [], [], [], []

    for df in dfs.values():
        cyc = df["cycle"].to_numpy(float)
        if np.isfinite(cyc).any():
            cycles_max = max(cycles_max, int(np.nanmax(cyc)))

        mae = df["mae"].to_numpy(float)
        delta = df["delta"].to_numpy(float)
        rmse = df["rmse"].to_numpy(float)
        r2 = df["r2"].to_numpy(float)

        mae_last.append(mean_last_n(mae, last_n)); mae_final.append(last_value(mae))
        delta_last.append(mean_last_n(delta, last_n)); delta_final.append(last_value(delta))
        rmse_last.append(mean_last_n(rmse, last_n)); rmse_final.append(last_value(rmse))
        r2_last.append(mean_last_n(r2, last_n)); r2_final.append(last_value(r2))

    def _mu(xs: List[float]) -> float:
        a = np.asarray(xs, dtype=float)
        a = a[np.isfinite(a)]
        return float(np.mean(a)) if a.size else float("nan")

    return GroupSummary(
        group=name,
        n_runs=len(dfs),
        cycles_max=cycles_max,
        end2end_mae_last3_mean=_mu(mae_last),
        end2end_mae_final=_mu(mae_final),
        delta_mae_last3_mean=_mu(delta_last),
        delta_mae_final=_mu(delta_final),
        end2end_rmse_last3_mean=_mu(rmse_last),
        end2end_rmse_final=_mu(rmse_final),
        end2end_r2_last3_mean=_mu(r2_last),
        end2end_r2_final=_mu(r2_final),
    )


def plot_mean_std(
    ax: plt.Axes,
    x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    line_color: str,
    label: str,
    *,
    linestyle: str = "-",
    band_color: str = "",
    band_alpha: float = 0.18,
    lw: float = 2.2,
    ms: float = 3.0,
) -> None:
    band = np.isfinite(std)
    if band_color:
        ax.fill_between(
            x,
            np.where(band, mean - std, np.nan),
            np.where(band, mean + std, np.nan),
            color=band_color,
            alpha=band_alpha,
            linewidth=0,
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
    )


def add_final_inset(
    ax: plt.Axes,
    match_cycle: int,
    label1: str,
    mu1: float,
    sd1: float,
    label2: str,
    mu2: float,
    sd2: float,
    *,
    loc: str = "lower left",
) -> None:
    text = (
        f"cycle {match_cycle}:\n"
        f"{label1}: {mu1:.3f} ± {sd1:.3f}\n"
        f"{label2}: {mu2:.3f} ± {sd2:.3f}"
    )
    if loc == "lower left":
        x, y, ha, va = 0.025, 0.10, "left", "bottom"
    elif loc == "upper right":
        x, y, ha, va = 0.975, 0.90, "right", "top"
    else:
        x, y, ha, va = 0.025, 0.10, "left", "bottom"
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=8,
        bbox=dict(facecolor=PALETTE["bg"], edgecolor=PALETTE["light"], alpha=0.82, pad=2.0),
        zorder=10,
    )


def cyclewise_stability_rows(
    method: str,
    mae_wide: pd.DataFrame,
    delta_wide: pd.DataFrame,
    match_cycle: int,
) -> Dict[str, object]:
    mae_w = truncate_wide_at_cycle(mae_wide, match_cycle)
    delta_w = truncate_wide_at_cycle(delta_wide, match_cycle)
    _, _, mae_sd, _ = mean_std_n(mae_w)
    _, _, delta_sd, _ = mean_std_n(delta_w)
    return {
        "method": method,
        "fixed_cycle": match_cycle,
        "mean_cyclewise_sd_end2end_hse_MAE": float(np.nanmean(mae_sd)),
        "mean_cyclewise_sd_delta_nonpbe_zbg_MAE": float(np.nanmean(delta_sd)),
    }


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    delta_base = (root / args.delta_dir).resolve()
    direct_base = (root / args.direct_dir).resolve()
    out_dir = root / args.out_dir

    show_grid = True
    if args.no_grid:
        show_grid = False
    elif args.grid:
        show_grid = True

    style_matplotlib(args.font)

    delta_dirs = discover_run_dirs(delta_base, args.delta_glob)
    direct_dirs = discover_run_dirs(direct_base, args.direct_glob)
    if not delta_dirs:
        raise RuntimeError(f"No Δ-ML AL dirs found under {delta_base} matching {args.delta_glob}")
    if not direct_dirs:
        raise RuntimeError(f"No direct-HSE AL dirs found under {direct_base} matching {args.direct_glob}")

    delta_dfs = load_group(delta_dirs, args.metrics_file, args.col_mae, args.col_delta, args.col_rmse, args.col_r2)
    direct_dfs = load_group(direct_dirs, args.metrics_file, args.col_mae, args.col_delta, args.col_rmse, args.col_r2)

    delta = {
        "mae": to_wide(delta_dfs, "mae"),
        "delta": to_wide(delta_dfs, "delta"),
        "rmse": to_wide(delta_dfs, "rmse"),
        "r2": to_wide(delta_dfs, "r2"),
    }
    direct = {
        "mae": to_wide(direct_dfs, "mae"),
        "delta": to_wide(direct_dfs, "delta"),
        "rmse": to_wide(direct_dfs, "rmse"),
        "r2": to_wide(direct_dfs, "r2"),
    }

    match_cycle = int(args.match_cycle)
    acquired_labels = int(args.match_cycle * args.batch_size)
    total_labels = int(args.seed_labels + acquired_labels)

    fixed_budget_df = make_fixed_budget_table(
        delta,
        direct,
        match_cycle,
        acquired_labels,
        int(args.seed_labels),
        total_labels,
        args.delta_label,
        args.direct_label,
    )

    # Show only cycles up to the fixed comparison point.
    d_mae_plot = truncate_wide_at_cycle(delta["mae"], match_cycle)
    d_delta_plot = truncate_wide_at_cycle(delta["delta"], match_cycle)
    h_mae_plot = truncate_wide_at_cycle(direct["mae"], match_cycle)
    h_delta_plot = truncate_wide_at_cycle(direct["delta"], match_cycle)

    xDM, mDM, sDM, _ = mean_std_n(d_mae_plot)
    xDD, mDD, sDD, _ = mean_std_n(d_delta_plot)
    xHM, mHM, sHM, _ = mean_std_n(h_mae_plot)
    xHD, mHD, sHD, _ = mean_std_n(h_delta_plot)

    # Fixed-cycle values for insets.
    d_mae_mu, d_mae_sd, _ = mean_std_values(values_at_cycle(delta["mae"], match_cycle))
    h_mae_mu, h_mae_sd, _ = mean_std_values(values_at_cycle(direct["mae"], match_cycle))
    d_del_mu, d_del_sd, _ = mean_std_values(values_at_cycle(delta["delta"], match_cycle))
    h_del_mu, h_del_sd, _ = mean_std_values(values_at_cycle(direct["delta"], match_cycle))

    fig, (axA, axB) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(args.figwidth, args.figheight),
        sharex=True,
        constrained_layout=False,
    )

    # Panel A: routed end-to-end HSE MAE.
    plot_mean_std(
        axA,
        xDM,
        mDM,
        sDM,
        PALETTE["navy"],
        f"{args.delta_label} (n={len(delta_dfs)})",
        linestyle="-",
        band_color=PALETTE["navy"],
        band_alpha=0.18,
        lw=2.4,
        ms=3.2,
    )
    plot_mean_std(
        axA,
        xHM,
        mHM,
        sHM,
        PALETTE["purple"],
        f"{args.direct_label} (n={len(direct_dfs)})",
        linestyle="--",
        band_color=PALETTE["light"],
        band_alpha=0.35,
        lw=2.1,
        ms=3.0,
    )
    axA.set_ylabel("End-to-end HSE MAE (eV)")
    axA.text(0.01, 0.96, "A", transform=axA.transAxes, ha="left", va="top", fontsize=10, fontweight="bold")

    # Panel B: non-PBE-ZBG Delta-MAE.
    plot_mean_std(
        axB,
        xDD,
        mDD,
        sDD,
        PALETTE["pink"],
        f"{args.delta_label} (n={len(delta_dfs)})",
        linestyle="-",
        band_color=PALETTE["pink"],
        band_alpha=0.18,
        lw=2.4,
        ms=3.2,
    )
    plot_mean_std(
        axB,
        xHD,
        mHD,
        sHD,
        PALETTE["green"],
        f"{args.direct_label} (n={len(direct_dfs)})",
        linestyle="--",
        band_color=PALETTE["light"],
        band_alpha=0.35,
        lw=2.1,
        ms=3.0,
    )
    axB.set_ylabel("Nonmetal Δ-MAE\n(non-ZBG, eV)")
    axB.set_xlabel("Cycle")
    axB.text(0.01, 0.96, "B", transform=axB.transAxes, ha="left", va="top", fontsize=10, fontweight="bold")

    for ax in (axA, axB):
        add_match_cycle_marker(ax, match_cycle)
        full_frame(ax)
        if show_grid:
            ax.grid(True, which="major", axis="both", color=PALETTE["light"], linewidth=0.7, alpha=0.6)
        ax.tick_params(length=3.5, width=1.0)
        ax.set_xlim(left=0, right=match_cycle + 1.2)
        ax.yaxis.set_major_locator(MultipleLocator(args.y_major_step))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    axA.legend(frameon=True, facecolor=PALETTE["bg"], edgecolor=PALETTE["light"], loc="upper center")
    axB.legend(frameon=True, facecolor=PALETTE["bg"], edgecolor=PALETTE["light"], loc="upper center")

    axA.set_ylim(*ylims_from_mean_std(np.r_[mDM, mHM], np.r_[sDM, sHM], args.y_major_step))
    axB.set_ylim(*ylims_from_mean_std(np.r_[mDD, mHD], np.r_[sDD, sHD], args.y_major_step))

    if args.show_final_insets:
        add_final_inset(axA, match_cycle, args.delta_label, d_mae_mu, d_mae_sd, args.direct_label, h_mae_mu, h_mae_sd, loc="lower left")
        add_final_inset(axB, match_cycle, args.delta_label, d_del_mu, d_del_sd, args.direct_label, h_del_mu, h_del_sd, loc="lower left")

    fig.suptitle(args.title, y=0.995, fontsize=10, fontweight="bold")
    fig.subplots_adjust(left=0.12, right=0.98, top=0.93, bottom=0.10, hspace=0.18)

    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / args.out_prefix
    fig.savefig(str(base) + ".pdf", bbox_inches="tight")
    fig.savefig(str(base) + ".svg", bbox_inches="tight")
    fig.savefig(str(base) + ".png", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    if not args.no_table:
        fixed_csv = out_dir / f"{args.out_prefix}_fixed_budget_cycle{match_cycle}.csv"
        fixed_md = out_dir / f"{args.out_prefix}_fixed_budget_cycle{match_cycle}.md"
        fixed_budget_df.to_csv(fixed_csv, index=False)

        md_df = fixed_budget_df.copy()
        for c in [
            "end2end_hse_MAE_mean", "end2end_hse_MAE_std",
            "delta_val_MAE_nonmetal_mean", "delta_val_MAE_nonmetal_std",
            "end2end_hse_RMSE_mean", "end2end_hse_RMSE_std",
            "end2end_hse_R2_mean", "end2end_hse_R2_std",
        ]:
            md_df[c] = md_df[c].map(lambda v: f"{float(v):.4f}" if np.isfinite(float(v)) else "n/a")
        fixed_md.write_text(df_to_md_table(md_df), encoding="utf-8")

        summary = pd.DataFrame([
            asdict(summarize_group(args.delta_label, delta_dfs, args.last_n)),
            asdict(summarize_group(args.direct_label, direct_dfs, args.last_n)),
        ])
        summary_csv = out_dir / f"{args.out_prefix}_group_summary.csv"
        summary_md = out_dir / f"{args.out_prefix}_group_summary.md"
        summary.to_csv(summary_csv, index=False)
        summary_md_df = summary.copy()
        for c in summary_md_df.columns:
            if c not in {"group", "n_runs", "cycles_max"}:
                summary_md_df[c] = summary_md_df[c].map(
                    lambda v: f"{float(v):.4f}" if np.isfinite(float(v)) else "n/a"
                )
        summary_md.write_text(df_to_md_table(summary_md_df), encoding="utf-8")

        stability = pd.DataFrame([
            cyclewise_stability_rows(args.delta_label, delta["mae"], delta["delta"], match_cycle),
            cyclewise_stability_rows(args.direct_label, direct["mae"], direct["delta"], match_cycle),
        ])
        stability_csv = out_dir / f"{args.out_prefix}_cyclewise_stability.csv"
        stability_md = out_dir / f"{args.out_prefix}_cyclewise_stability.md"
        stability.to_csv(stability_csv, index=False)
        stability_md_df = stability.copy()
        for c in stability_md_df.columns:
            if c not in {"method", "fixed_cycle"}:
                stability_md_df[c] = stability_md_df[c].map(lambda v: f"{float(v):.4f}" if np.isfinite(float(v)) else "n/a")
        stability_md.write_text(df_to_md_table(stability_md_df), encoding="utf-8")

        print(f"[OK] Fixed-budget table: {fixed_csv}")
        print(f"[OK] Summary table:      {summary_csv}")
        print(f"[OK] Stability table:    {stability_csv}")
        print("\n[Fixed-cycle values]")
        print(fixed_budget_df.to_string(index=False))
        print("\n[Cycle-wise stability]")
        print(stability.to_string(index=False))

    print(f"[OK] Figure: {base}.pdf/.svg/.png")
    print(f"[INFO] Δ-ML dirs:      {len(delta_dirs)} from {delta_base} / {args.delta_glob}")
    print(f"[INFO] Direct-HSE dirs: {len(direct_dirs)} from {direct_base} / {args.direct_glob}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
