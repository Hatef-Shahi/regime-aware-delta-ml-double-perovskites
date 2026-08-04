#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gap 3 figure/table — Delta-ML AL vs Direct-HSE AL target ablation.

Purpose
-------
Compare two active-learning target formulations under the same fixed-cycle
budget:
  1) Delta-ML AL:      Stage-B target = HSE - PBE
  2) Direct-HSE AL:    Stage-B target = HSE

Expected layout from experiments/random-sets:
  al1/ ... al5/                 # Delta-ML AL runs
  direct1/ ... direct5/         # Direct-HSE AL runs

Each run directory must contain metrics/metrics.csv.

Plots:
  A) end2end_hse_MAE vs cycle, mean ± std over replicates
  B) end2end_hse_RMSE vs cycle, mean ± std over replicates

Tables:
  table_gap3_fixed_budget_cycleXX.csv/.md
  table_gap3_group_summary.csv/.md

Run from experiments/random-sets:
  python3 fig4_gap3_delta_vs_direct_hse.py \
    --root . \
    --delta-dir . --delta-glob 'al*' \
    --direct-dir . --direct-glob 'direct*' \
    --match-cycle 24 \
    --seed-labels 100 \
    --batch-size 7

Run from experiments root:
  python3 plots/fig4_gap3_delta_vs_direct_hse.py \
    --root random-sets \
    --delta-dir . --delta-glob 'al*' \
    --direct-dir . --direct-glob 'direct*' \
    --match-cycle 24
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


PALETTE: Dict[str, str] = {
    "navy": "#0D1164",
    "purple": "#8C00FF",
    "pink": "#FF3F7F",
    "gold": "#FFC400",
    "green": "#638C6D",
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
    end2end_rmse_last3_mean: float
    end2end_rmse_final: float
    end2end_r2_last3_mean: float
    end2end_r2_final: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gap 3: Delta-ML AL vs Direct-HSE AL target ablation")
    p.add_argument("--root", type=str, default=".", help="Root containing run directories")

    p.add_argument("--delta-dir", type=str, default=".", help="Base dir for Delta-ML AL runs, relative to root")
    p.add_argument("--direct-dir", type=str, default=".", help="Base dir for Direct-HSE AL runs, relative to root")
    p.add_argument("--delta-glob", type=str, default="al*", help="Glob for Delta-ML run dirs")
    p.add_argument("--direct-glob", type=str, default="direct*", help="Glob for Direct-HSE run dirs")
    p.add_argument("--metrics-file", type=str, default="metrics/metrics.csv")

    p.add_argument("--col-mae", type=str, default="end2end_hse_MAE")
    p.add_argument("--col-rmse", type=str, default="end2end_hse_RMSE")
    p.add_argument("--col-r2", type=str, default="end2end_hse_R2")

    p.add_argument("--out-dir", type=str, default="artifacts/gap3_delta_vs_direct_hse")
    p.add_argument("--last-n", type=int, default=3)
    p.add_argument("--no-table", action="store_true")

    p.add_argument("--match-cycle", type=int, default=24)
    p.add_argument("--seed-labels", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=7)

    p.add_argument("--grid", action="store_true")
    p.add_argument("--no-grid", action="store_true")
    p.add_argument("--dpi", type=int, default=600)
    p.add_argument("--figwidth", type=float, default=7.2)
    p.add_argument("--figheight", type=float, default=6.0)
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
    return re.sub(r"[^a-z0-9]+", "", name.lower())


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


def load_metrics(run_dir: Path, rel_metrics_file: str, col_mae: str, col_rmse: str, col_r2: str) -> pd.DataFrame:
    path = resolve_metrics_path(run_dir, rel_metrics_file)
    df = pd.read_csv(path)

    cyc_col = find_column(df, "cycle") or find_column(df, "al_cycle") or find_column(df, "iteration")
    if cyc_col is None:
        df["cycle"] = np.arange(len(df), dtype=float)
        cyc_col = "cycle"
    else:
        df[cyc_col] = pd.to_numeric(df[cyc_col], errors="coerce")

    mae_col = find_column(df, col_mae) or (col_mae if col_mae in df.columns else None)
    rmse_col = find_column(df, col_rmse) or (col_rmse if col_rmse in df.columns else None)
    r2_col = find_column(df, col_r2) or (col_r2 if col_r2 in df.columns else None)
    if mae_col is None or rmse_col is None or r2_col is None:
        raise KeyError(
            f"Missing required columns in {path}\n"
            f"Need: {col_mae}, {col_rmse}, {col_r2}\n"
            f"Have: {list(df.columns)}"
        )

    out = pd.DataFrame({
        "cycle": pd.to_numeric(df[cyc_col], errors="coerce"),
        "mae": pd.to_numeric(df[mae_col], errors="coerce"),
        "rmse": pd.to_numeric(df[rmse_col], errors="coerce"),
        "r2": pd.to_numeric(df[r2_col], errors="coerce"),
    })
    out = out[np.isfinite(out["cycle"].to_numpy(float))].copy()
    return out.sort_values("cycle")


def load_group(run_dirs: List[Path], rel_metrics_file: str, col_mae: str, col_rmse: str, col_r2: str) -> Dict[str, pd.DataFrame]:
    return {rd.name: load_metrics(rd, rel_metrics_file, col_mae, col_rmse, col_r2) for rd in run_dirs}


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


def ylims_from_mean_std(mean: np.ndarray, std: np.ndarray) -> Tuple[float, float]:
    m = np.isfinite(mean)
    s = np.isfinite(std)
    if not np.any(m):
        return (0.0, 1.0)
    lo = float(np.min(mean[m]))
    hi = float(np.max(mean[m]))
    if np.any(s):
        lo = min(lo, float(np.min((mean - std)[s])))
        hi = max(hi, float(np.max((mean + std)[s])))
    pad = 0.06 * (hi - lo) if hi > lo else 0.1
    return (max(0.0, lo - pad), hi + pad)


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


def fixed_row(method: str, mae_wide: pd.DataFrame, rmse_wide: pd.DataFrame, r2_wide: pd.DataFrame, match_cycle: int, total_labels: int) -> Dict[str, object]:
    mae_mu, mae_sd, mae_n = mean_std_values(values_at_cycle(mae_wide, match_cycle))
    rmse_mu, rmse_sd, rmse_n = mean_std_values(values_at_cycle(rmse_wide, match_cycle))
    r2_mu, r2_sd, r2_n = mean_std_values(values_at_cycle(r2_wide, match_cycle))
    return {
        "method": method,
        "n_runs": min(mae_n, rmse_n, r2_n),
        "fixed_cycle": int(match_cycle),
        "total_hse06_labels": int(total_labels),
        "end2end_hse_MAE_mean": mae_mu,
        "end2end_hse_MAE_std": mae_sd,
        "end2end_hse_RMSE_mean": rmse_mu,
        "end2end_hse_RMSE_std": rmse_sd,
        "end2end_hse_R2_mean": r2_mu,
        "end2end_hse_R2_std": r2_sd,
    }


def make_fixed_budget_table(delta: Dict[str, pd.DataFrame], direct: Dict[str, pd.DataFrame], match_cycle: int, total_labels: int) -> pd.DataFrame:
    return pd.DataFrame([
        fixed_row("Delta-ML AL", delta["mae"], delta["rmse"], delta["r2"], match_cycle, total_labels),
        fixed_row("Direct-HSE AL", direct["mae"], direct["rmse"], direct["r2"], match_cycle, total_labels),
    ])


def add_match_cycle_marker(ax: plt.Axes, match_cycle: int, total_labels: int) -> None:
    ax.axvline(match_cycle, color=PALETTE["ink"], linestyle=":", linewidth=0.9, alpha=0.95, zorder=1)
    ax.text(
        match_cycle, 0.96,
        f"cycle {match_cycle}\n{total_labels} labels",
        transform=ax.get_xaxis_transform(),
        ha="center", va="top", fontsize=8, color=PALETTE["ink"],
        bbox=dict(facecolor=PALETTE["bg"], edgecolor="none", alpha=0.75, pad=1.5),
    )


def summarize_group(name: str, dfs: Dict[str, pd.DataFrame], last_n: int) -> GroupSummary:
    cycles_max = -1
    mae_last, mae_final, rmse_last, rmse_final, r2_last, r2_final = [], [], [], [], [], []
    for df in dfs.values():
        cyc = df["cycle"].to_numpy(float)
        if np.isfinite(cyc).any():
            cycles_max = max(cycles_max, int(np.nanmax(cyc)))
        mae = df["mae"].to_numpy(float)
        rmse = df["rmse"].to_numpy(float)
        r2 = df["r2"].to_numpy(float)
        mae_last.append(mean_last_n(mae, last_n)); mae_final.append(last_value(mae))
        rmse_last.append(mean_last_n(rmse, last_n)); rmse_final.append(last_value(rmse))
        r2_last.append(mean_last_n(r2, last_n)); r2_final.append(last_value(r2))

    def _mu(xs: List[float]) -> float:
        a = np.asarray(xs, dtype=float)
        a = a[np.isfinite(a)]
        return float(np.mean(a)) if a.size else float("nan")

    return GroupSummary(
        group=name, n_runs=len(dfs), cycles_max=cycles_max,
        end2end_mae_last3_mean=_mu(mae_last), end2end_mae_final=_mu(mae_final),
        end2end_rmse_last3_mean=_mu(rmse_last), end2end_rmse_final=_mu(rmse_final),
        end2end_r2_last3_mean=_mu(r2_last), end2end_r2_final=_mu(r2_final),
    )


def plot_mean_std(ax: plt.Axes, x: np.ndarray, mean: np.ndarray, std: np.ndarray, line_color: str, label: str, *, linestyle: str = "-", band_color: str = "", band_alpha: float = 0.18, lw: float = 2.2, ms: float = 3.0) -> None:
    band = np.isfinite(std)
    if band_color:
        ax.fill_between(x, np.where(band, mean - std, np.nan), np.where(band, mean + std, np.nan), color=band_color, alpha=band_alpha, linewidth=0)
    ax.plot(x, mean, color=line_color, linewidth=lw, marker="o", markersize=ms, linestyle=linestyle, label=label)


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
        raise RuntimeError(f"No Delta-ML AL dirs found under {delta_base} matching {args.delta_glob}")
    if not direct_dirs:
        raise RuntimeError(f"No Direct-HSE AL dirs found under {direct_base} matching {args.direct_glob}")

    delta_dfs = load_group(delta_dirs, args.metrics_file, args.col_mae, args.col_rmse, args.col_r2)
    direct_dfs = load_group(direct_dirs, args.metrics_file, args.col_mae, args.col_rmse, args.col_r2)

    delta = {"mae": to_wide(delta_dfs, "mae"), "rmse": to_wide(delta_dfs, "rmse"), "r2": to_wide(delta_dfs, "r2")}
    direct = {"mae": to_wide(direct_dfs, "mae"), "rmse": to_wide(direct_dfs, "rmse"), "r2": to_wide(direct_dfs, "r2")}

    match_cycle = int(args.match_cycle)
    total_labels = int(args.seed_labels + args.match_cycle * args.batch_size)

    fixed_budget_df = make_fixed_budget_table(delta, direct, match_cycle, total_labels)

    # Show only cycles up to the fixed comparison point.
    d_mae_plot = truncate_wide_at_cycle(delta["mae"], match_cycle)
    d_rmse_plot = truncate_wide_at_cycle(delta["rmse"], match_cycle)
    h_mae_plot = truncate_wide_at_cycle(direct["mae"], match_cycle)
    h_rmse_plot = truncate_wide_at_cycle(direct["rmse"], match_cycle)

    xDM, mDM, sDM, _ = mean_std_n(d_mae_plot)
    xDR, mDR, sDR, _ = mean_std_n(d_rmse_plot)
    xHM, mHM, sHM, _ = mean_std_n(h_mae_plot)
    xHR, mHR, sHR, _ = mean_std_n(h_rmse_plot)

    fig, (axA, axB) = plt.subplots(nrows=2, ncols=1, figsize=(args.figwidth, args.figheight), sharex=True, constrained_layout=False)

    plot_mean_std(axA, xDM, mDM, sDM, PALETTE["navy"], f"Delta-ML AL (n={len(delta_dfs)})", linestyle="-", band_color=PALETTE["navy"], band_alpha=0.18, lw=2.4, ms=3.2)
    plot_mean_std(axA, xHM, mHM, sHM, PALETTE["purple"], f"Direct-HSE AL (n={len(direct_dfs)})", linestyle="--", band_color=PALETTE["light"], band_alpha=0.35, lw=2.1, ms=3.0)
    axA.set_ylabel("End-to-end MAE (eV)")
    axA.text(0.01, 0.96, "A", transform=axA.transAxes, ha="left", va="top", fontsize=10, fontweight="bold")

    plot_mean_std(axB, xDR, mDR, sDR, PALETTE["pink"], f"Delta-ML AL (n={len(delta_dfs)})", linestyle="-", band_color=PALETTE["pink"], band_alpha=0.18, lw=2.4, ms=3.2)
    plot_mean_std(axB, xHR, mHR, sHR, PALETTE["green"], f"Direct-HSE AL (n={len(direct_dfs)})", linestyle="--", band_color=PALETTE["light"], band_alpha=0.35, lw=2.1, ms=3.0)
    axB.set_ylabel("End-to-end RMSE (eV)")
    axB.set_xlabel("Cycle")
    axB.text(0.01, 0.96, "B", transform=axB.transAxes, ha="left", va="top", fontsize=10, fontweight="bold")

    for ax in (axA, axB):
        add_match_cycle_marker(ax, match_cycle, total_labels)
        full_frame(ax)
        if show_grid:
            ax.grid(True, which="major", axis="both", color=PALETTE["light"], linewidth=0.7, alpha=0.6)
        ax.tick_params(length=3.5, width=1.0)
        ax.set_xlim(left=0, right=match_cycle + 1.2)

    axA.legend(
        frameon=True,
        facecolor=PALETTE["bg"],
        edgecolor=PALETTE["light"],
        loc="upper center",
    )
    axB.legend(
        frameon=True,
        facecolor=PALETTE["bg"],
        edgecolor=PALETTE["light"],
        loc="upper center",
    )

    axA.set_ylim(*ylims_from_mean_std(np.r_[mDM, mHM], np.r_[sDM, sHM]))
    axB.set_ylim(*ylims_from_mean_std(np.r_[mDR, mHR], np.r_[sDR, sHR]))

    fig.suptitle("Delta-ML vs Direct-HSE target ablation", y=0.995, fontsize=10, fontweight="bold")
    fig.subplots_adjust(left=0.11, right=0.98, top=0.93, bottom=0.10, hspace=0.18)

    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / "fig_gap3_delta_vs_direct_hse"
    fig.savefig(str(base) + ".pdf", bbox_inches="tight")
    fig.savefig(str(base) + ".svg", bbox_inches="tight")
    fig.savefig(str(base) + ".png", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    if not args.no_table:
        fixed_csv = out_dir / f"table_gap3_fixed_budget_cycle{match_cycle}.csv"
        fixed_md = out_dir / f"table_gap3_fixed_budget_cycle{match_cycle}.md"
        fixed_budget_df.to_csv(fixed_csv, index=False)

        md_df = fixed_budget_df.copy()
        for c in [
            "end2end_hse_MAE_mean", "end2end_hse_MAE_std",
            "end2end_hse_RMSE_mean", "end2end_hse_RMSE_std",
            "end2end_hse_R2_mean", "end2end_hse_R2_std",
        ]:
            md_df[c] = md_df[c].map(lambda v: f"{float(v):.4f}" if np.isfinite(float(v)) else "—")
        fixed_md.write_text(df_to_md_table(md_df), encoding="utf-8")

        summary = pd.DataFrame([
            asdict(summarize_group("Delta-ML AL", delta_dfs, args.last_n)),
            asdict(summarize_group("Direct-HSE AL", direct_dfs, args.last_n)),
        ])
        summary_csv = out_dir / "table_gap3_group_summary.csv"
        summary_md = out_dir / "table_gap3_group_summary.md"
        summary.to_csv(summary_csv, index=False)
        summary_md_df = summary.copy()
        for c in summary_md_df.columns:
            if c not in {"group", "n_runs", "cycles_max"}:
                summary_md_df[c] = summary_md_df[c].map(lambda v: f"{float(v):.4f}" if np.isfinite(float(v)) else "—")
        summary_md.write_text(df_to_md_table(summary_md_df), encoding="utf-8")

    print(f"[OK] Figure: {base}.pdf/.svg/.png")
    if not args.no_table:
        print(f"[OK] Fixed-budget table: {out_dir / f'table_gap3_fixed_budget_cycle{match_cycle}.csv'}")
        print(f"[OK] Summary table:      {out_dir / 'table_gap3_group_summary.csv'}")
    print(f"[INFO] Delta dirs:  {len(delta_dirs)} from {delta_base} / {args.delta_glob}")
    print(f"[INFO] Direct dirs: {len(direct_dirs)} from {direct_base} / {args.direct_glob}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
