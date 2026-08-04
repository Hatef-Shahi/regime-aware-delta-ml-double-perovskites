#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Publication-level m_reg × q_floor selector-sensitivity plot.

Inputs
------
Expected run layout from the experiments root:
    sweep/m5_q030/metrics/metrics.csv
    sweep/m5_q035/metrics/metrics.csv
    ...
    sweep/m7_q040/metrics/metrics.csv

Main outputs
------------
1) fig_sweep_heatmap_fixed_cycle.pdf/.svg/.png
   Two-panel fixed-budget heatmap:
      A) End-to-end HSE MAE
      B) Δ-MAE for non-PBE-ZBG candidates

2) table_sweep_fixed_cycle<cycle>.csv/.md
   Fixed-cycle table with extra metrics.

3) table_sweep_all_cycles_summary.csv/.md
   Best/final/last-N diagnostics.

4) fig_sweep_learning_curves.pdf/.svg/.png
   Optional 3×3 learning-curve grid unless --no-curves is used.

Notes
-----
- This script intentionally avoids the internal label "Stage-B".
- The heatmap cell borders are explicitly drawn so cells remain separated even
  when adjacent values have similar colors.
- The star marks the retained operating point, not necessarily the best value
  in every panel.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter


PALETTE: Dict[str, str] = {
    "navy": "#003566",
    "teal": "#0a9396",
    "gold": "#ee9b00",
    "pink": "#ff4d6d",
    "green": "#2d6a4f",
    "ink": "#111111",
    "light": "#DDDDDD",
    "grid": "#FFFFFF",
    "bg": "#FFFFFF",
}


@dataclass(frozen=True)
class RunKey:
    m_reg_mult: int
    q_floor: float
    run_dir: Path


@dataclass(frozen=True)
class RunMetrics:
    cycle: np.ndarray
    columns: Dict[str, np.ndarray]

    def get(self, name: str) -> np.ndarray:
        return self.columns.get(name, np.full_like(self.cycle, np.nan, dtype=float))


METRIC_CANDIDATES: Dict[str, List[str]] = {
    "end2end_hse_MAE": ["end2end_hse_MAE", "end2end_hse_val_MAE"],
    "end2end_hse_RMSE": ["end2end_hse_RMSE", "end2end_hse_val_RMSE"],
    "end2end_hse_R2": ["end2end_hse_R2", "end2end_hse_val_R2"],
    "delta_val_MAE_nonmetal": ["delta_val_MAE_nonmetal", "delta_MAE_non_PBE_ZBG", "non_PBE_ZBG_delta_MAE"],
    "delta_val_RMSE_nonmetal": ["delta_val_RMSE_nonmetal", "delta_RMSE_non_PBE_ZBG", "non_PBE_ZBG_delta_RMSE"],
    "delta_val_R2_nonmetal": ["delta_val_R2_nonmetal", "delta_R2_non_PBE_ZBG", "non_PBE_ZBG_delta_R2"],
    "gate_auc": ["gate_auc"],
    "gate_ap": ["gate_ap"],
    "gate_f1": ["gate_f1"],
    "gate_prec": ["gate_prec"],
    "gate_rec": ["gate_rec"],
    "train_n_total": ["train_n_total"],
    "train_n_nonmetal_used": ["train_n_nonmetal_used"],
}


LOWER_IS_BETTER = {
    "end2end_hse_MAE": True,
    "end2end_hse_RMSE": True,
    "delta_val_MAE_nonmetal": True,
    "delta_val_RMSE_nonmetal": True,
    "gate_auc": False,
    "gate_f1": False,
    "gate_prec": False,
    "gate_rec": False,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Publication-level m_reg × q_floor sensitivity analysis")
    p.add_argument("--root", type=str, default=".", help="Experiments root")
    p.add_argument("--runs-dir", type=str, default="sweep", help="Directory containing m*_q* run folders")
    p.add_argument("--out-dir", type=str, default="artifacts/pub_final_figures/fig_sweep_mreg_qfloor")
    p.add_argument("--metrics-file", type=str, default="metrics/metrics.csv")
    p.add_argument("--match-cycle", type=int, default=24)
    p.add_argument("--seed-labels", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=7)
    p.add_argument("--last-n", type=int, default=3)
    p.add_argument("--retained-m", type=int, default=6, help="m_reg value to mark with a star")
    p.add_argument("--retained-q", type=float, default=0.35, help="q_floor value to mark with a star")
    p.add_argument("--title", type=str, default="Selector sensitivity: shortlist width and score floor")
    p.add_argument("--out-prefix", type=str, default="fig_sweep_heatmap_fixed_cycle")
    p.add_argument("--dpi", type=int, default=600)
    p.add_argument("--font", type=str, default="")
    p.add_argument("--grid", action="store_true")
    p.add_argument("--no-grid", action="store_true")
    p.add_argument("--figwidth", type=float, default=7.2)
    p.add_argument("--figheight", type=float, default=3.25)
    p.add_argument("--curve-figwidth", type=float, default=7.2)
    p.add_argument("--curve-figheight", type=float, default=7.2)
    p.add_argument("--no-curves", action="store_true", help="Only write heatmap/table outputs")
    p.add_argument("--cell-fontsize", type=float, default=8.0)
    p.add_argument("--cell-border-lw", type=float, default=1.2)
    p.add_argument("--show-best-outline", action="store_true", help="Also outline each panel's best cell")
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
        "axes.titlesize": 9,
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


def find_first_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for cand in candidates:
        col = find_column(df, cand)
        if col is not None:
            return col
    return None


def parse_run_key(run_dir: Path) -> Optional[RunKey]:
    """Accept names such as m5_q030, m05_q030, run_v3_m5_q030."""
    name = run_dir.name
    m = re.search(r"(?:^|_)m(?P<m>\d{1,3})(?=_|$)", name)
    q = re.search(r"(?:^|_)q(?P<q>\d{2,3})(?=_|$)", name)
    if not m or not q:
        return None
    m_val = int(m.group("m"))
    q_raw = int(q.group("q"))
    q_val = q_raw / 100.0
    return RunKey(m_reg_mult=m_val, q_floor=float(q_val), run_dir=run_dir)


def discover_runs(root: Path, runs_dir: str) -> Dict[Tuple[int, float], RunKey]:
    base = root / runs_dir
    if not base.exists():
        raise FileNotFoundError(f"Missing runs directory: {base}")

    out: Dict[Tuple[int, float], RunKey] = {}
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        rk = parse_run_key(d)
        if rk is None:
            continue
        out[(rk.m_reg_mult, rk.q_floor)] = rk
    return out


def read_metrics_csv(run_dir: Path, rel_metrics_file: str) -> RunMetrics:
    path = run_dir / rel_metrics_file
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics file: {path}")

    df = pd.read_csv(path)

    cycle_col = find_column(df, "cycle") or find_column(df, "al_cycle") or find_column(df, "iteration")
    if cycle_col is None:
        cycle = np.arange(len(df), dtype=float)
    else:
        cycle = pd.to_numeric(df[cycle_col], errors="coerce").to_numpy(dtype=float)

    cols: Dict[str, np.ndarray] = {}
    for canonical, candidates in METRIC_CANDIDATES.items():
        col = find_first_column(df, candidates)
        if col is None:
            cols[canonical] = np.full(len(df), np.nan, dtype=float)
        else:
            cols[canonical] = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)

    finite_cycle = np.isfinite(cycle)
    order = np.argsort(cycle[finite_cycle])
    cycle2 = cycle[finite_cycle][order]
    cols2 = {k: v[finite_cycle][order] for k, v in cols.items()}
    return RunMetrics(cycle=cycle2, columns=cols2)


def value_at_cycle(rm: RunMetrics, metric: str, cycle: int) -> float:
    idx = np.where(np.isclose(rm.cycle, float(cycle)))[0]
    if idx.size == 0:
        return float("nan")
    y = rm.get(metric)
    return float(y[idx[-1]]) if idx[-1] < len(y) else float("nan")


def last_k_mean(rm: RunMetrics, metric: str, k: int = 3) -> float:
    y = rm.get(metric)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return float("nan")
    return float(np.mean(y[-min(k, y.size):]))


def best_value(rm: RunMetrics, metric: str, lower_is_better: bool = True) -> float:
    y = rm.get(metric)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return float("nan")
    return float(np.min(y) if lower_is_better else np.max(y))


def best_cycle(rm: RunMetrics, metric: str, lower_is_better: bool = True) -> float:
    y = rm.get(metric)
    mask = np.isfinite(rm.cycle) & np.isfinite(y)
    if not np.any(mask):
        return float("nan")
    yy = y[mask]
    cc = rm.cycle[mask]
    i = int(np.argmin(yy) if lower_is_better else np.argmax(yy))
    return float(cc[i])


def auc_norm(rm: RunMetrics, metric: str) -> float:
    x = rm.cycle
    y = rm.get(metric)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 2:
        return float("nan")
    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))
    if x_max <= x_min:
        return float("nan")
    x_n = (x - x_min) / (x_max - x_min)
    y_n = np.zeros_like(y) if y_max <= y_min else (y - y_min) / (y_max - y_min)
    integrate = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return float(integrate(y_n, x_n))


def build_tables(
    metrics_by_key: Dict[Tuple[int, float], RunMetrics],
    match_cycle: int,
    seed_labels: int,
    batch_size: int,
    last_n: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows_fixed: List[Dict[str, object]] = []
    rows_all: List[Dict[str, object]] = []
    acquired_labels = int(match_cycle * batch_size)
    total_labels = int(seed_labels + acquired_labels)

    for (m, q), rm in sorted(metrics_by_key.items()):
        fixed = {
            "m_reg_mult": int(m),
            "q_floor": float(q),
            "fixed_cycle": int(match_cycle),
            "seed_labels": int(seed_labels),
            "acquired_labels": int(acquired_labels),
            "total_labels": int(total_labels),
        }
        for metric in METRIC_CANDIDATES.keys():
            fixed[metric] = value_at_cycle(rm, metric, match_cycle)
        rows_fixed.append(fixed)

        rows_all.append({
            "m_reg_mult": int(m),
            "q_floor": float(q),
            "cycles_n": int(len(rm.cycle)),
            "last_cycle": int(np.nanmax(rm.cycle)) if len(rm.cycle) else -1,
            "best_end2end_hse_MAE": best_value(rm, "end2end_hse_MAE", True),
            "best_end2end_hse_MAE_cycle": best_cycle(rm, "end2end_hse_MAE", True),
            "best_delta_val_MAE_nonmetal": best_value(rm, "delta_val_MAE_nonmetal", True),
            "best_delta_val_MAE_nonmetal_cycle": best_cycle(rm, "delta_val_MAE_nonmetal", True),
            "end2end_hse_MAE_lastN_mean": last_k_mean(rm, "end2end_hse_MAE", last_n),
            "end2end_hse_RMSE_lastN_mean": last_k_mean(rm, "end2end_hse_RMSE", last_n),
            "delta_val_MAE_nonmetal_lastN_mean": last_k_mean(rm, "delta_val_MAE_nonmetal", last_n),
            "delta_val_RMSE_nonmetal_lastN_mean": last_k_mean(rm, "delta_val_RMSE_nonmetal", last_n),
            "gate_auc_lastN_mean": last_k_mean(rm, "gate_auc", last_n),
            "gate_f1_lastN_mean": last_k_mean(rm, "gate_f1", last_n),
            "gate_rec_lastN_mean": last_k_mean(rm, "gate_rec", last_n),
            "gate_prec_lastN_mean": last_k_mean(rm, "gate_prec", last_n),
            "end2end_hse_MAE_auc_norm": auc_norm(rm, "end2end_hse_MAE"),
            "delta_val_MAE_nonmetal_auc_norm": auc_norm(rm, "delta_val_MAE_nonmetal"),
        })

    fixed_df = pd.DataFrame(rows_fixed)
    fixed_df = fixed_df.sort_values(
        ["end2end_hse_MAE", "end2end_hse_RMSE", "delta_val_MAE_nonmetal", "gate_auc"],
        ascending=[True, True, True, False],
        na_position="last",
    ).reset_index(drop=True)
    fixed_df.insert(0, "rank", np.arange(1, len(fixed_df) + 1))

    all_df = pd.DataFrame(rows_all).sort_values(["m_reg_mult", "q_floor"]).reset_index(drop=True)
    return fixed_df, all_df


def pivot_metric(fixed_df: pd.DataFrame, metric: str, m_vals: List[int], q_vals: List[float]) -> np.ndarray:
    arr = np.full((len(m_vals), len(q_vals)), np.nan, dtype=float)
    for i, m in enumerate(m_vals):
        for j, q in enumerate(q_vals):
            sub = fixed_df[
                (fixed_df["m_reg_mult"].astype(int) == int(m))
                & (np.isclose(fixed_df["q_floor"].astype(float), float(q)))
            ]
            if len(sub):
                arr[i, j] = float(sub.iloc[0][metric])
    return arr


def best_pos_for(data: np.ndarray, lower_is_better: bool = True) -> Optional[Tuple[int, int]]:
    if not np.isfinite(data).any():
        return None
    fn = np.nanargmin if lower_is_better else np.nanargmax
    flat = int(fn(data))
    return tuple(np.unravel_index(flat, data.shape))  # type: ignore[return-value]


def retained_pos(m_vals: List[int], q_vals: List[float], retained_m: int, retained_q: float) -> Optional[Tuple[int, int]]:
    if retained_m not in m_vals:
        return None
    q_idx = None
    for j, q in enumerate(q_vals):
        if np.isclose(q, retained_q):
            q_idx = j
            break
    if q_idx is None:
        return None
    return (m_vals.index(retained_m), q_idx)


def draw_cell_borders(ax: plt.Axes, n_rows: int, n_cols: int, lw: float) -> None:
    # Minor gridlines at cell edges.
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color=PALETTE["grid"], linestyle="-", linewidth=lw)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Explicit rectangle borders make boundaries survive PDF/SVG rendering.
    for i in range(n_rows):
        for j in range(n_cols):
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1.0, 1.0, fill=False,
                                   edgecolor=PALETTE["bg"], linewidth=lw, zorder=4))


def annotate_heatmap(
    ax: plt.Axes,
    data: np.ndarray,
    fmt: str = ".3f",
    retained: Optional[Tuple[int, int]] = None,
    best: Optional[Tuple[int, int]] = None,
    show_best_outline: bool = False,
    fontsize: float = 8.0,
) -> None:
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return
    threshold = float(np.nanmedian(finite))

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if not np.isfinite(v):
                txt = "—"
                color = PALETTE["ink"]
            else:
                txt = format(v, fmt)
                color = "white" if v < threshold else PALETTE["ink"]
            suffix = "★" if retained is not None and (i, j) == retained else ""
            ax.text(
                j, i, f"{txt}{suffix}", ha="center", va="center",
                fontsize=fontsize, color=color, zorder=6,
                fontweight="bold" if suffix else "normal",
            )

    if retained is not None:
        i, j = retained
        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1.0, 1.0, fill=False,
                               edgecolor=PALETTE["ink"], linewidth=1.8, zorder=7))

    if show_best_outline and best is not None and best != retained:
        i, j = best
        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1.0, 1.0, fill=False,
                               edgecolor=PALETTE["gold"], linewidth=1.8, zorder=7))


def plot_heatmaps(
    fixed_df: pd.DataFrame,
    m_vals: List[int],
    q_vals: List[float],
    out_dir: Path,
    out_prefix: str,
    dpi: int,
    figwidth: float,
    figheight: float,
    title: str,
    retained_m: int,
    retained_q: float,
    cell_fontsize: float,
    cell_border_lw: float,
    show_best_outline: bool,
) -> None:
    mae = pivot_metric(fixed_df, "end2end_hse_MAE", m_vals, q_vals)
    dnm = pivot_metric(fixed_df, "delta_val_MAE_nonmetal", m_vals, q_vals)

    retained = retained_pos(m_vals, q_vals, retained_m, retained_q)
    best_mae = best_pos_for(mae, lower_is_better=True)
    best_dnm = best_pos_for(dnm, lower_is_better=True)

    fig, axes = plt.subplots(1, 2, figsize=(figwidth, figheight), constrained_layout=False)

    panels = [
        (axes[0], mae, "End-to-end HSE MAE (eV)", "A", best_mae),
        (axes[1], dnm, "Δ-MAE for non-ZBG candidates (eV)", "B", best_dnm),
    ]

    for ax, data, panel_title, panel_label, panel_best in panels:
        im = ax.imshow(data, cmap="viridis_r", aspect="auto", zorder=1)
        draw_cell_borders(ax, data.shape[0], data.shape[1], cell_border_lw)
        annotate_heatmap(
            ax,
            data,
            fmt=".3f",
            retained=retained,
            best=panel_best,
            show_best_outline=show_best_outline,
            fontsize=cell_fontsize,
        )
        ax.set_title(panel_title, pad=7)
        ax.set_xticks(np.arange(len(q_vals)))
        ax.set_xticklabels([f"{q:.2f}" for q in q_vals])
        ax.set_yticks(np.arange(len(m_vals)))
        ax.set_yticklabels([str(m) for m in m_vals])
        ax.set_xlabel(r"$q_{\mathrm{floor}}$")
        ax.text(
            0.02, 0.98, panel_label, transform=ax.transAxes,
            ha="left", va="top", fontsize=10, fontweight="bold",
            color=PALETTE["ink"],
            bbox=dict(facecolor=PALETTE["bg"], edgecolor="none", alpha=0.78, pad=1.5),
            zorder=8,
        )
        full_frame(ax)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)
        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))

    axes[0].set_ylabel(r"$m_{\mathrm{reg}}$")
    fig.suptitle(title, y=0.995, fontsize=10, fontweight="bold")
    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.18, top=0.86, wspace=0.34)

    base = out_dir / out_prefix
    fig.savefig(str(base) + ".pdf", bbox_inches="tight")
    fig.savefig(str(base) + ".svg", bbox_inches="tight")
    fig.savefig(str(base) + ".png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def ylims_for_curves(metrics_by_key: Dict[Tuple[int, float], RunMetrics]) -> Tuple[float, float]:
    vals = []
    for rm in metrics_by_key.values():
        vals.append(rm.get("end2end_hse_MAE"))
        vals.append(rm.get("delta_val_MAE_nonmetal"))
    y = np.concatenate(vals) if vals else np.array([0.0, 1.0])
    y = y[np.isfinite(y)]
    if y.size == 0:
        return (0.0, 1.0)
    lo, hi = float(np.min(y)), float(np.max(y))
    pad = 0.06 * (hi - lo) if hi > lo else 0.1
    return max(0.0, lo - pad), hi + pad


def plot_learning_grid(
    metrics_by_key: Dict[Tuple[int, float], RunMetrics],
    m_vals: List[int],
    q_vals: List[float],
    out_dir: Path,
    dpi: int,
    figwidth: float,
    figheight: float,
    show_grid: bool,
    match_cycle: int,
) -> None:
    fig, axes = plt.subplots(
        len(m_vals), len(q_vals), figsize=(figwidth, figheight), sharex=True, sharey=True,
        constrained_layout=False,
    )
    if len(m_vals) == 1 and len(q_vals) == 1:
        axes = np.array([[axes]])
    elif len(m_vals) == 1:
        axes = np.array([axes])
    elif len(q_vals) == 1:
        axes = np.array([[a] for a in axes])

    y_lim = ylims_for_curves(metrics_by_key)
    handles = None
    for i, m in enumerate(m_vals):
        for j, q in enumerate(q_vals):
            ax = axes[i, j]
            rm = metrics_by_key.get((m, q))
            if rm is None:
                ax.text(0.5, 0.5, "missing", transform=ax.transAxes, ha="center", va="center")
                full_frame(ax)
                continue
            x = rm.cycle
            y_e = rm.get("end2end_hse_MAE")
            y_d = rm.get("delta_val_MAE_nonmetal")
            h1 = ax.plot(x, y_e, marker="o", markersize=2.5, linewidth=1.35,
                         color=PALETTE["navy"], label="End-to-end HSE MAE")[0]
            h2 = ax.plot(x, y_d, marker="o", markersize=2.5, linewidth=1.35,
                         color=PALETTE["pink"], label="Δ-MAE for non-ZBG candidates (eV)")[0]
            if handles is None:
                handles = (h1, h2)
            ax.axvline(match_cycle, color=PALETTE["ink"], linestyle=":", linewidth=0.8, alpha=0.8)
            if show_grid:
                ax.grid(True, which="major", axis="both", color=PALETTE["light"], linewidth=0.6, alpha=0.55)
            ax.set_ylim(*y_lim)
            full_frame(ax)
            ax.tick_params(length=3.0, width=0.9)
            if i == 0:
                ax.set_title(rf"$q_{{\mathrm{{floor}}}}={q:.2f}$", pad=6)
            if j == 0:
                ax.set_ylabel(rf"$m_{{\mathrm{{reg}}}}={m}$\nMAE (eV)")
            if i == len(m_vals) - 1:
                ax.set_xlabel("Cycle")

    if handles is not None:
        fig.legend(
            handles=list(handles), labels=[h.get_label() for h in handles],
            loc="lower center", ncol=2, frameon=True, facecolor=PALETTE["bg"],
            edgecolor=PALETTE["light"], bbox_to_anchor=(0.5, 0.02),
        )
    fig.suptitle(r"Learning-curve sensitivity: $m_{\mathrm{reg}}$ × $q_{\mathrm{floor}}$", y=0.985, fontsize=10, fontweight="bold")
    fig.subplots_adjust(bottom=0.12, top=0.92, left=0.10, right=0.98, wspace=0.16, hspace=0.18)
    base = out_dir / "fig_sweep_learning_curves"
    fig.savefig(str(base) + ".pdf", bbox_inches="tight")
    fig.savefig(str(base) + ".svg", bbox_inches="tight")
    fig.savefig(str(base) + ".png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def df_to_md_table(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for _, row in df.iterrows():
        vals = []
        for c in headers:
            v = row[c]
            if isinstance(v, float):
                vals.append(f"{v:.4f}" if np.isfinite(v) else "—")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    out_dir = (root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    style_matplotlib(args.font)
    show_grid = True
    if args.no_grid:
        show_grid = False
    elif args.grid:
        show_grid = True

    run_keys = discover_runs(root, args.runs_dir)
    if not run_keys:
        raise RuntimeError(f"No runs found in {root / args.runs_dir}. Expected names like m5_q030.")

    m_vals = sorted({m for m, _ in run_keys.keys()})
    q_vals = sorted({q for _, q in run_keys.keys()})

    metrics_by_key: Dict[Tuple[int, float], RunMetrics] = {}
    for key, rk in sorted(run_keys.items()):
        metrics_by_key[key] = read_metrics_csv(rk.run_dir, args.metrics_file)

    fixed_df, all_df = build_tables(metrics_by_key, args.match_cycle, args.seed_labels, args.batch_size, args.last_n)

    fixed_csv = out_dir / f"table_sweep_fixed_cycle{args.match_cycle}.csv"
    fixed_md = out_dir / f"table_sweep_fixed_cycle{args.match_cycle}.md"
    all_csv = out_dir / "table_sweep_all_cycles_summary.csv"
    all_md = out_dir / "table_sweep_all_cycles_summary.md"
    fixed_df.to_csv(fixed_csv, index=False)
    fixed_md.write_text(df_to_md_table(fixed_df), encoding="utf-8")
    all_df.to_csv(all_csv, index=False)
    all_md.write_text(df_to_md_table(all_df), encoding="utf-8")

    plot_heatmaps(
        fixed_df,
        m_vals,
        q_vals,
        out_dir,
        args.out_prefix,
        args.dpi,
        args.figwidth,
        args.figheight,
        args.title,
        args.retained_m,
        args.retained_q,
        args.cell_fontsize,
        args.cell_border_lw,
        args.show_best_outline,
    )

    if not args.no_curves:
        plot_learning_grid(metrics_by_key, m_vals, q_vals, out_dir, args.dpi,
                           args.curve_figwidth, args.curve_figheight, show_grid, args.match_cycle)

    retained = fixed_df[
        (fixed_df["m_reg_mult"].astype(int) == int(args.retained_m))
        & (np.isclose(fixed_df["q_floor"].astype(float), float(args.retained_q)))
    ]
    best = fixed_df.iloc[0]

    print(f"[OK] Wrote: {fixed_csv}")
    print(f"[OK] Wrote: {all_csv}")
    print(f"[OK] Wrote: {out_dir / (args.out_prefix + '.pdf')}")
    if not args.no_curves:
        print(f"[OK] Wrote: {out_dir / 'fig_sweep_learning_curves.pdf'}")

    print("[BEST fixed-cycle by e2e MAE → RMSE → Δ-MAE → gate AUC]")
    print(
        f"  rank=1 m={int(best['m_reg_mult'])} q={float(best['q_floor']):.2f} "
        f"e2e_MAE={float(best['end2end_hse_MAE']):.4f} "
        f"e2e_RMSE={float(best['end2end_hse_RMSE']):.4f} "
        f"delta_non_ZBG_MAE={float(best['delta_val_MAE_nonmetal']):.4f} "
        f"gate_AUC={float(best['gate_auc']):.4f}"
    )

    if len(retained):
        r = retained.iloc[0]
        print("[RETAINED setting]")
        print(
            f"  m={int(r['m_reg_mult'])} q={float(r['q_floor']):.2f} "
            f"rank={int(r['rank'])} "
            f"e2e_MAE={float(r['end2end_hse_MAE']):.4f} "
            f"delta_non_PBE_ZBG_MAE={float(r['delta_val_MAE_nonmetal']):.4f}"
        )
    else:
        print(f"[WARN] retained setting m={args.retained_m}, q={args.retained_q:.2f} not found in table")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
