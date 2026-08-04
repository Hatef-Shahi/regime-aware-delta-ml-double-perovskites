#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Publication DOS ablation plot for the final Kg=2 workflow.

Compares DOS-off and DOS-on AL replicates under the same settings:
  Kg=2, Kr=5, m_reg=6, q_floor=0.35, theta=0.16.

Inputs expected under each run directory:
  metrics/metrics.csv
Optionally, for held-out test summaries:
  test_eval/test_metrics_cycle24.json
Optionally, for trajectory-overlap summaries:
  selections/selected_cycle*.csv

Outputs:
  <out-dir>/<out-prefix>.pdf/.svg/.png
  <out-dir>/<out-prefix>_fixed_budget_validation.csv/.md
  <out-dir>/<out-prefix>_cyclewise_stability.csv
  <out-dir>/<out-prefix>_test_metrics_by_run.csv
  <out-dir>/<out-prefix>_test_metrics_summary.csv
  <out-dir>/<out-prefix>_paired_test_differences.csv
  <out-dir>/<out-prefix>_selection_overlap.csv
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
from matplotlib.ticker import MultipleLocator


PALETTE: Dict[str, str] = {
    "off": "#003566",      # retained workflow
    "on": "#ff4d6d",       # DOS-on ablation
    "off_band": "#003566",
    "on_band": "#ff4d6d",
    "ink": "#111111",
    "light": "#DDDDDD",
    "bg": "#FFFFFF",
}

METRIC_KEYS = [
    "MAE",
    "RMSE",
    "R2",
    "Delta_NM_MAE",
    "Delta_NM_RMSE",
    "PBE_ZBG_MAE",
    "PBE_ZBG_RMSE",
    "non_PBE_ZBG_MAE",
    "non_PBE_ZBG_RMSE",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DOS-on vs DOS-off ablation for final Kg=2 workflow")
    p.add_argument("--root", type=str, default=".", help="Experiments root")
    p.add_argument("--dos-off-dir", type=str, required=True, help="Base dir containing DOS-off runs")
    p.add_argument("--dos-on-dir", type=str, required=True, help="Base dir containing DOS-on runs")
    p.add_argument("--dos-off-glob", type=str, default="al*", help="Glob for DOS-off runs")
    p.add_argument("--dos-on-glob", type=str, default="al*", help="Glob for DOS-on runs")
    p.add_argument("--metrics-file", type=str, default="metrics/metrics.csv")

    p.add_argument("--col-mae", type=str, default="end2end_hse_MAE")
    p.add_argument("--col-delta", type=str, default="delta_val_MAE_nonmetal")
    p.add_argument("--col-rmse", type=str, default="end2end_hse_RMSE")
    p.add_argument("--col-r2", type=str, default="end2end_hse_R2")

    p.add_argument("--match-cycle", type=int, default=24)
    p.add_argument("--seed-labels", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=7)

    p.add_argument("--dos-off-label", type=str, default="DOS-off")
    p.add_argument("--dos-on-label", type=str, default="DOS-on")
    p.add_argument("--title", type=str, default="DOS-targeting ablation")
    p.add_argument("--out-dir", type=str, default="artifacts/pub_final_figures/fig_dos_ablation")
    p.add_argument("--out-prefix", type=str, default="fig_dos_ablation")

    p.add_argument("--grid", action="store_true")
    p.add_argument("--no-grid", action="store_true")
    p.add_argument("--dpi", type=int, default=600)
    p.add_argument("--figwidth", type=float, default=7.2)
    p.add_argument("--figheight", type=float, default=5.8)
    p.add_argument("--font", type=str, default="")
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
    out = [p for p in sorted(base.glob(pattern)) if p.is_dir()]
    if not out:
        raise RuntimeError(f"No run directories found under {base} matching {pattern}")
    return out


def resolve_metrics_path(run_dir: Path, rel_metrics_file: str) -> Path:
    p = run_dir / rel_metrics_file
    if p.exists():
        return p
    candidates = sorted(list(run_dir.glob("**/metrics/metrics.csv")) + list(run_dir.glob("**/metrics.csv")))
    candidates = [c for c in candidates if c.is_file()]
    if not candidates:
        raise FileNotFoundError(f"No metrics file found under {run_dir}")
    candidates.sort(key=lambda c: (len(c.relative_to(run_dir).parts), str(c)))
    return candidates[0]


def load_metrics(run_dir: Path, rel_metrics_file: str, col_mae: str, col_delta: str, col_rmse: str, col_r2: str) -> pd.DataFrame:
    path = resolve_metrics_path(run_dir, rel_metrics_file)
    df = pd.read_csv(path)
    cyc_col = find_column(df, "cycle") or find_column(df, "al_cycle") or find_column(df, "iteration")
    if cyc_col is None:
        df["cycle"] = np.arange(len(df), dtype=float)
        cyc_col = "cycle"
    mae_col = find_column(df, col_mae)
    delta_col = find_column(df, col_delta)
    rmse_col = find_column(df, col_rmse)
    r2_col = find_column(df, col_r2)
    missing = []
    for name, col in [(col_mae, mae_col), (col_delta, delta_col), (col_rmse, rmse_col), (col_r2, r2_col)]:
        if col is None:
            missing.append(name)
    if missing:
        raise KeyError(f"Missing columns {missing} in {path}. Have: {list(df.columns)}")
    out = pd.DataFrame({
        "cycle": pd.to_numeric(df[cyc_col], errors="coerce"),
        "mae": pd.to_numeric(df[mae_col], errors="coerce"),
        "delta": pd.to_numeric(df[delta_col], errors="coerce"),
        "rmse": pd.to_numeric(df[rmse_col], errors="coerce"),
        "r2": pd.to_numeric(df[r2_col], errors="coerce"),
    })
    return out[np.isfinite(out["cycle"].to_numpy(float))].sort_values("cycle")


def load_group(run_dirs: List[Path], args: argparse.Namespace) -> Dict[str, pd.DataFrame]:
    return {
        rd.name: load_metrics(rd, args.metrics_file, args.col_mae, args.col_delta, args.col_rmse, args.col_r2)
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
    loc = np.where(np.isclose(idx, float(cycle)))[0]
    if loc.size == 0:
        raise ValueError(f"Cycle {cycle} not found. Available: {list(wide.index.to_numpy())}")
    return wide.iloc[loc[0]].dropna().to_numpy(float)


def mean_std_values(vals: np.ndarray) -> Tuple[float, float, int]:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan"), float("nan"), 0
    return float(vals.mean()), float(vals.std(ddof=1)) if vals.size > 1 else 0.0, int(vals.size)


def ylims_from_mean_std(means: np.ndarray, stds: np.ndarray) -> Tuple[float, float]:
    means = np.asarray(means, dtype=float)
    stds = np.asarray(stds, dtype=float)
    m = np.isfinite(means)
    if not np.any(m):
        return (0.0, 1.0)
    lo = float(np.nanmin(means[m]))
    hi = float(np.nanmax(means[m]))
    s = np.isfinite(stds)
    if np.any(s):
        lo = min(lo, float(np.nanmin((means - stds)[s])))
        hi = max(hi, float(np.nanmax((means + stds)[s])))
    pad = 0.06 * (hi - lo) if hi > lo else 0.1
    return max(0.0, lo - pad), hi + pad


def plot_mean_std(ax: plt.Axes, x: np.ndarray, mean: np.ndarray, std: np.ndarray, color: str, label: str, *, linestyle: str = "-", band_alpha: float = 0.18) -> None:
    band = np.isfinite(std)
    ax.fill_between(
        x,
        np.where(band, mean - std, np.nan),
        np.where(band, mean + std, np.nan),
        color=color,
        alpha=band_alpha,
        linewidth=0,
    )
    ax.plot(x, mean, color=color, linewidth=2.4, marker="o", markersize=3.2, linestyle=linestyle, label=label)


def add_cycle_marker(ax: plt.Axes, match_cycle: int) -> None:
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
        bbox=dict(facecolor=PALETTE["bg"], edgecolor="none", alpha=0.80, pad=1.5),
        zorder=6,
    )


def fixed_budget_table(off: Dict[str, pd.DataFrame], on: Dict[str, pd.DataFrame], match_cycle: int, seed_labels: int, batch_size: int, off_label: str, on_label: str) -> pd.DataFrame:
    acquired = int(match_cycle * batch_size)
    total = int(seed_labels + acquired)

    def row(label: str, mae_w: pd.DataFrame, delta_w: pd.DataFrame, rmse_w: pd.DataFrame, r2_w: pd.DataFrame) -> Dict[str, object]:
        mae_mu, mae_sd, mae_n = mean_std_values(values_at_cycle(mae_w, match_cycle))
        del_mu, del_sd, del_n = mean_std_values(values_at_cycle(delta_w, match_cycle))
        rmse_mu, rmse_sd, rmse_n = mean_std_values(values_at_cycle(rmse_w, match_cycle))
        r2_mu, r2_sd, r2_n = mean_std_values(values_at_cycle(r2_w, match_cycle))
        return {
            "method": label,
            "n_runs": min(mae_n, del_n, rmse_n, r2_n),
            "fixed_cycle": match_cycle,
            "seed_labels": seed_labels,
            "acquired_labels": acquired,
            "total_labels": total,
            "end2end_hse_MAE_mean": mae_mu,
            "end2end_hse_MAE_std": mae_sd,
            "delta_val_MAE_nonmetal_mean": del_mu,
            "delta_val_MAE_nonmetal_std": del_sd,
            "end2end_hse_RMSE_mean": rmse_mu,
            "end2end_hse_RMSE_std": rmse_sd,
            "end2end_hse_R2_mean": r2_mu,
            "end2end_hse_R2_std": r2_sd,
        }

    rows = [
        row(off_label, off["mae"], off["delta"], off["rmse"], off["r2"]),
        row(on_label, on["mae"], on["delta"], on["rmse"], on["r2"]),
    ]

    diff = {"method": f"{on_label} - {off_label}", "n_runs": "paired", "fixed_cycle": match_cycle, "seed_labels": seed_labels, "acquired_labels": acquired, "total_labels": total}
    for key in [
        "end2end_hse_MAE",
        "delta_val_MAE_nonmetal",
        "end2end_hse_RMSE",
        "end2end_hse_R2",
    ]:
        diff[f"{key}_mean"] = rows[1][f"{key}_mean"] - rows[0][f"{key}_mean"]
        diff[f"{key}_std"] = np.nan
    rows.append(diff)
    return pd.DataFrame(rows)


def cyclewise_stability_table(off: Dict[str, pd.DataFrame], on: Dict[str, pd.DataFrame], match_cycle: int, off_label: str, on_label: str) -> pd.DataFrame:
    rows = []
    for metric_key, label in [("mae", "end2end_hse_MAE"), ("delta", "delta_val_MAE_nonmetal")]:
        for group_label, group in [(off_label, off), (on_label, on)]:
            wide = truncate_wide_at_cycle(group[metric_key], match_cycle)
            _, _, sd, _ = mean_std_n(wide)
            vals = sd[np.isfinite(sd)]
            rows.append({
                "metric": label,
                "method": group_label,
                "mean_cyclewise_sd": float(vals.mean()) if vals.size else np.nan,
                "median_cyclewise_sd": float(np.median(vals)) if vals.size else np.nan,
                "max_cyclewise_sd": float(vals.max()) if vals.size else np.nan,
            })
    return pd.DataFrame(rows)


def read_test_metrics(base: Path, run_dirs: List[Path], label: str, match_cycle: int) -> pd.DataFrame:
    rows = []
    for rd in run_dirs:
        p = rd / "test_eval" / f"test_metrics_cycle{match_cycle}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        row = {"run": rd.name, "method": label}
        for k in ["cycle", "n_test_used"] + METRIC_KEYS:
            row[k] = d.get(k)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_test_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, sub in df.groupby("method", sort=False):
        row = {"method": method, "n_runs": len(sub)}
        for k in METRIC_KEYS:
            vals = pd.to_numeric(sub[k], errors="coerce").to_numpy(float) if k in sub else np.array([])
            vals = vals[np.isfinite(vals)]
            row[f"{k}_mean"] = float(vals.mean()) if vals.size else np.nan
            row[f"{k}_std"] = float(vals.std(ddof=1)) if vals.size > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def paired_test_differences(test_df: pd.DataFrame, off_label: str, on_label: str) -> pd.DataFrame:
    rows = []
    if test_df.empty:
        return pd.DataFrame()
    for run in sorted(set(test_df["run"])):
        sub = test_df[test_df["run"] == run]
        if not {off_label, on_label}.issubset(set(sub["method"])):
            continue
        off = sub[sub["method"] == off_label].iloc[0]
        on = sub[sub["method"] == on_label].iloc[0]
        row = {"run": run, "comparison": f"{on_label} - {off_label}"}
        for k in METRIC_KEYS:
            row[f"{k}_diff"] = pd.to_numeric(pd.Series([on.get(k)]), errors="coerce").iloc[0] - pd.to_numeric(pd.Series([off.get(k)]), errors="coerce").iloc[0]
        rows.append(row)
    return pd.DataFrame(rows)


def selection_overlap(off_base_dirs: List[Path], on_base_dirs: List[Path], match_cycle: int, off_label: str, on_label: str) -> pd.DataFrame:
    # Pair by identical run names: al1 with al1, etc.
    off_map = {p.name: p for p in off_base_dirs}
    on_map = {p.name: p for p in on_base_dirs}
    rows = []
    for run in sorted(set(off_map) & set(on_map)):
        off_sel = off_map[run] / "selections"
        on_sel = on_map[run] / "selections"
        for c in range(match_cycle + 1):
            p_off = off_sel / f"selected_cycle{c}.csv"
            p_on = on_sel / f"selected_cycle{c}.csv"
            if not p_off.exists() or not p_on.exists():
                continue
            try:
                a = pd.read_csv(p_off)["structure_id"].astype(str).tolist()
                b = pd.read_csv(p_on)["structure_id"].astype(str).tolist()
            except Exception:
                continue
            rows.append({
                "run": run,
                "cycle": c,
                "identical_ordered": int(a == b),
                "overlap_count": len(set(a) & set(b)),
                "n_selected_off": len(a),
                "n_selected_on": len(b),
                "overlap_fraction_of_7": len(set(a) & set(b)) / 7.0,
            })
    return pd.DataFrame(rows)


def df_to_md_table(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    rows = df.astype(str).values.tolist()
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    out_dir = (root / args.out_dir).resolve()
    off_base = (root / args.dos_off_dir).resolve()
    on_base = (root / args.dos_on_dir).resolve()

    style_matplotlib(args.font)
    show_grid = not args.no_grid
    if args.grid:
        show_grid = True

    off_dirs = discover_run_dirs(off_base, args.dos_off_glob)
    on_dirs = discover_run_dirs(on_base, args.dos_on_glob)

    off_dfs = load_group(off_dirs, args)
    on_dfs = load_group(on_dirs, args)

    off = {"mae": to_wide(off_dfs, "mae"), "delta": to_wide(off_dfs, "delta"), "rmse": to_wide(off_dfs, "rmse"), "r2": to_wide(off_dfs, "r2")}
    on = {"mae": to_wide(on_dfs, "mae"), "delta": to_wide(on_dfs, "delta"), "rmse": to_wide(on_dfs, "rmse"), "r2": to_wide(on_dfs, "r2")}

    match_cycle = int(args.match_cycle)
    off_mae_plot = truncate_wide_at_cycle(off["mae"], match_cycle)
    on_mae_plot = truncate_wide_at_cycle(on["mae"], match_cycle)
    off_delta_plot = truncate_wide_at_cycle(off["delta"], match_cycle)
    on_delta_plot = truncate_wide_at_cycle(on["delta"], match_cycle)

    xOM, mOM, sOM, _ = mean_std_n(off_mae_plot)
    xNM, mNM, sNM, _ = mean_std_n(on_mae_plot)
    xOD, mOD, sOD, _ = mean_std_n(off_delta_plot)
    xND, mND, sND, _ = mean_std_n(on_delta_plot)

    fixed_df = fixed_budget_table(off, on, match_cycle, args.seed_labels, args.batch_size, args.dos_off_label, args.dos_on_label)

    # Cycle-24 inset values
    off_mae_mu, off_mae_sd, _ = mean_std_values(values_at_cycle(off["mae"], match_cycle))
    on_mae_mu, on_mae_sd, _ = mean_std_values(values_at_cycle(on["mae"], match_cycle))
    off_del_mu, off_del_sd, _ = mean_std_values(values_at_cycle(off["delta"], match_cycle))
    on_del_mu, on_del_sd, _ = mean_std_values(values_at_cycle(on["delta"], match_cycle))

    fig, (axA, axB) = plt.subplots(2, 1, figsize=(args.figwidth, args.figheight), sharex=True, constrained_layout=False)

    plot_mean_std(axA, xOM, mOM, sOM, PALETTE["off"], f"{args.dos_off_label} (n={len(off_dfs)})", linestyle="-", band_alpha=0.18)
    plot_mean_std(axA, xNM, mNM, sNM, PALETTE["on"], f"{args.dos_on_label} (n={len(on_dfs)})", linestyle="--", band_alpha=0.18)
    axA.set_ylabel("End-to-end HSE MAE (eV)")
    axA.text(0.01, 0.96, "A", transform=axA.transAxes, ha="left", va="top", fontsize=10, fontweight="bold")

    plot_mean_std(axB, xOD, mOD, sOD, PALETTE["off"], f"{args.dos_off_label} (n={len(off_dfs)})", linestyle="-", band_alpha=0.18)
    plot_mean_std(axB, xND, mND, sND, PALETTE["on"], f"{args.dos_on_label} (n={len(on_dfs)})", linestyle="--", band_alpha=0.18)
    axB.set_ylabel("Stage-B Δ-MAE\n(non-PBE-ZBG, eV)")
    axB.set_xlabel("Cycle")
    axB.text(0.01, 0.96, "B", transform=axB.transAxes, ha="left", va="top", fontsize=10, fontweight="bold")

    inset_A = f"cycle {match_cycle}:\n{args.dos_off_label}: {off_mae_mu:.3f} ± {off_mae_sd:.3f}\n{args.dos_on_label}: {on_mae_mu:.3f} ± {on_mae_sd:.3f}"
    inset_B = f"cycle {match_cycle}:\n{args.dos_off_label}: {off_del_mu:.3f} ± {off_del_sd:.3f}\n{args.dos_on_label}: {on_del_mu:.3f} ± {on_del_sd:.3f}"
    axA.text(0.03, 0.12, inset_A, transform=axA.transAxes, ha="left", va="bottom", fontsize=8,
             bbox=dict(facecolor=PALETTE["bg"], edgecolor=PALETTE["light"], alpha=0.82, pad=2.0))
    axB.text(0.03, 0.12, inset_B, transform=axB.transAxes, ha="left", va="bottom", fontsize=8,
             bbox=dict(facecolor=PALETTE["bg"], edgecolor=PALETTE["light"], alpha=0.82, pad=2.0))

    for ax in (axA, axB):
        add_cycle_marker(ax, match_cycle)
        full_frame(ax)
        ax.set_xlim(left=0, right=match_cycle + 1.2)
        ax.yaxis.set_major_locator(MultipleLocator(0.05))
        if show_grid:
            ax.grid(True, which="major", axis="both", color=PALETTE["light"], linewidth=0.7, alpha=0.6)
        ax.tick_params(length=3.5, width=1.0)
        ax.legend(frameon=True, facecolor=PALETTE["bg"], edgecolor=PALETTE["light"], loc="upper center")

    axA.set_ylim(*ylims_from_mean_std(np.r_[mOM, mNM], np.r_[sOM, sNM]))
    axB.set_ylim(*ylims_from_mean_std(np.r_[mOD, mND], np.r_[sOD, sND]))

    fig.suptitle(args.title, y=0.995, fontsize=10, fontweight="bold")
    fig.subplots_adjust(left=0.12, right=0.98, top=0.93, bottom=0.10, hspace=0.18)

    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / args.out_prefix
    fig.savefig(str(base) + ".pdf", bbox_inches="tight")
    fig.savefig(str(base) + ".svg", bbox_inches="tight")
    fig.savefig(str(base) + ".png", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    # Tables
    fixed_csv = out_dir / f"{args.out_prefix}_fixed_budget_validation.csv"
    fixed_md = out_dir / f"{args.out_prefix}_fixed_budget_validation.md"
    fixed_df.to_csv(fixed_csv, index=False)
    fixed_md.write_text(df_to_md_table(fixed_df.round(4)), encoding="utf-8")

    stab_df = cyclewise_stability_table(off, on, match_cycle, args.dos_off_label, args.dos_on_label)
    stab_df.to_csv(out_dir / f"{args.out_prefix}_cyclewise_stability.csv", index=False)

    test_df = pd.concat([
        read_test_metrics(off_base, off_dirs, args.dos_off_label, match_cycle),
        read_test_metrics(on_base, on_dirs, args.dos_on_label, match_cycle),
    ], ignore_index=True)
    if not test_df.empty:
        test_df.to_csv(out_dir / f"{args.out_prefix}_test_metrics_by_run.csv", index=False)
        test_summary = summarize_test_metrics(test_df)
        test_summary.to_csv(out_dir / f"{args.out_prefix}_test_metrics_summary.csv", index=False)
        test_diff = paired_test_differences(test_df, args.dos_off_label, args.dos_on_label)
        if not test_diff.empty:
            test_diff.to_csv(out_dir / f"{args.out_prefix}_paired_test_differences.csv", index=False)

    overlap = selection_overlap(off_dirs, on_dirs, match_cycle, args.dos_off_label, args.dos_on_label)
    if not overlap.empty:
        overlap.to_csv(out_dir / f"{args.out_prefix}_selection_overlap.csv", index=False)
        overlap_summary = overlap.groupby("run", as_index=False).agg(
            identical_cycles=("identical_ordered", "sum"),
            compared_cycles=("cycle", "count"),
            mean_overlap_count=("overlap_count", "mean"),
            mean_overlap_fraction=("overlap_fraction_of_7", "mean"),
        )
        overlap_summary.to_csv(out_dir / f"{args.out_prefix}_selection_overlap_summary.csv", index=False)

    print(f"[OK] Figure: {base}.pdf/.svg/.png")
    print(f"[OK] Validation fixed-budget table: {fixed_csv}")
    print(f"[OK] Cycle-wise stability table: {out_dir / f'{args.out_prefix}_cyclewise_stability.csv'}")
    if not test_df.empty:
        print(f"[OK] Test metrics: {out_dir / f'{args.out_prefix}_test_metrics_summary.csv'}")
    if not overlap.empty:
        print(f"[OK] Selection overlap: {out_dir / f'{args.out_prefix}_selection_overlap_summary.csv'}")
    print(f"[INFO] DOS-off dirs: {len(off_dirs)} from {off_base} / {args.dos_off_glob}")
    print(f"[INFO] DOS-on dirs:  {len(on_dirs)} from {on_base} / {args.dos_on_glob}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
