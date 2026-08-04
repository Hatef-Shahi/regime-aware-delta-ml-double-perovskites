#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Publication-style gate-threshold sensitivity plot.

This simplified version is designed for the final paper figure. It avoids the
old six-panel strip-heatmap and shows only the two metrics that justify the
retained threshold:

  A) End-to-end HSE MAE at the fixed cycle
  B) Delta-MAE for non-ZBG candidates at the fixed cycle

Gate precision/recall/F1 are still written to the output CSV/Markdown tables,
but are not plotted because recall is saturated and the gate panel adds little
visual information.

Expected layout:
  sweep-thr/theta012/metrics/metrics.csv
  sweep-thr/theta014/metrics/metrics.csv
  ...

Example:
  python3 plots/theta_sensitivity_plot_npj.py \
    --root . \
    --runs-dir sweep-thr \
    --match-cycle 24 \
    --retained-theta 0.16 \
    --out-dir artifacts/pub_final_figures/fig_theta_sensitivity \
    --out-prefix fig_theta_sensitivity_summary \
    --dpi 600
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


PALETTE: Dict[str, str] = {
    "navy": "#003566",
    "pink": "#ff4d6d",
    "teal": "#0a9396",
    "green": "#2d6a4f",
    "purple": "#7b2cbf",
    "gold": "#ee9b00",
    "ink": "#111111",
    "light": "#DDDDDD",
    "bg": "#FFFFFF",
}

METRIC_CANDIDATES: Dict[str, List[str]] = {
    "end2end_hse_MAE": ["end2end_hse_MAE", "end2end_hse_val_MAE"],
    "end2end_hse_RMSE": ["end2end_hse_RMSE", "end2end_hse_val_RMSE"],
    "end2end_hse_R2": ["end2end_hse_R2", "end2end_hse_val_R2"],
    "delta_val_MAE_nonmetal": ["delta_val_MAE_nonmetal"],
    "delta_val_RMSE_nonmetal": ["delta_val_RMSE_nonmetal"],
    "delta_val_R2_nonmetal": ["delta_val_R2_nonmetal"],
    "gate_auc": ["gate_auc"],
    "gate_ap": ["gate_ap"],
    "gate_f1": ["gate_f1"],
    "gate_prec": ["gate_prec", "gate_precision"],
    "gate_rec": ["gate_rec", "gate_recall"],
    "gate_fp": ["gate_fp", "gate_FP"],
    "gate_fn": ["gate_fn", "gate_FN"],
    "gate_tp": ["gate_tp", "gate_TP"],
    "gate_tn": ["gate_tn", "gate_TN"],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Two-panel gate-threshold sensitivity plot")
    p.add_argument("--root", type=str, default=".")
    p.add_argument("--runs-dir", type=str, default="sweep-thr")
    p.add_argument("--metrics-file", type=str, default="metrics/metrics.csv")
    p.add_argument("--out-dir", type=str, default="artifacts/theta_sensitivity_final")
    p.add_argument("--out-prefix", type=str, default="fig_theta_sensitivity_summary")
    p.add_argument("--match-cycle", type=int, default=24)
    p.add_argument("--seed-labels", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=7)
    p.add_argument("--retained-theta", type=float, default=0.16)
    p.add_argument("--title", type=str, default="Gate-threshold sensitivity")
    p.add_argument("--dpi", type=int, default=600)
    p.add_argument("--figwidth", type=float, default=7.2)
    p.add_argument("--figheight", type=float, default=4.1)
    p.add_argument("--font", type=str, default="")
    p.add_argument("--grid", action="store_true")
    p.add_argument("--no-grid", action="store_true")
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


def find_first_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for cand in candidates:
        col = find_column(df, cand)
        if col is not None:
            return col
    return None


def parse_theta_from_dir(run_dir: Path) -> Optional[float]:
    name = run_dir.name.lower()
    m = re.search(r"(?:^|_)theta(?P<t>\d{2,3})(?=_|$)", name)
    if not m:
        return None
    return int(m.group("t")) / 100.0


def read_metric_at_cycle(path: Path, match_cycle: int) -> Dict[str, float]:
    df = pd.read_csv(path)
    cyc_col = find_column(df, "cycle") or find_column(df, "al_cycle") or find_column(df, "iteration")
    if cyc_col is None:
        df["cycle"] = np.arange(len(df), dtype=float)
        cyc_col = "cycle"
    df[cyc_col] = pd.to_numeric(df[cyc_col], errors="coerce")

    sub = df[np.isclose(df[cyc_col].astype(float), float(match_cycle))]
    if sub.empty:
        raise ValueError(f"Cycle {match_cycle} not found in {path}")
    row = sub.iloc[-1]

    out: Dict[str, float] = {}
    for canonical, candidates in METRIC_CANDIDATES.items():
        col = find_first_column(df, candidates)
        out[canonical] = float(pd.to_numeric(row[col], errors="coerce")) if col is not None else float("nan")
    return out


def discover_table(root: Path, runs_dir: str, metrics_file: str, match_cycle: int, seed_labels: int, batch_size: int) -> pd.DataFrame:
    base = root / runs_dir
    if not base.exists():
        raise FileNotFoundError(base)

    rows = []
    for rd in sorted(base.iterdir()):
        if not rd.is_dir():
            continue
        theta = parse_theta_from_dir(rd)
        if theta is None:
            continue
        metrics_path = rd / metrics_file
        if not metrics_path.exists():
            raise FileNotFoundError(metrics_path)
        d = read_metric_at_cycle(metrics_path, match_cycle)
        rows.append({
            "theta": theta,
            "fixed_cycle": match_cycle,
            "acquired_labels": match_cycle * batch_size,
            "seed_labels": seed_labels,
            "total_labels": seed_labels + match_cycle * batch_size,
            **d,
            "run_dir": str(rd),
        })

    if not rows:
        raise RuntimeError(f"No theta runs found under {base}; expected folders like theta016")
    return pd.DataFrame(rows).sort_values("theta").reset_index(drop=True)


def fmt_value(v: float) -> str:
    return f"{v:.3f}" if np.isfinite(v) else "—"


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.075,
        1.05,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
        color=PALETTE["ink"],
    )


def annotate_points(ax: plt.Axes, x: np.ndarray, y: np.ndarray, dy: float = 0.006) -> None:
    for xx, yy in zip(x, y):
        if np.isfinite(yy):
            ax.text(xx, yy + dy, f"{yy:.3f}", ha="center", va="bottom", fontsize=8)


def set_clean_ylim(ax: plt.Axes, y: np.ndarray, pad_frac: float = 0.32) -> None:
    finite = y[np.isfinite(y)]
    if finite.size == 0:
        return
    lo, hi = float(np.min(finite)), float(np.max(finite))
    pad = pad_frac * (hi - lo) if hi > lo else 0.02
    pad = max(pad, 0.010)
    ax.set_ylim(max(0.0, lo - pad), hi + pad)


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

    df = discover_table(root, args.runs_dir, args.metrics_file, args.match_cycle, args.seed_labels, args.batch_size)
    theta = df["theta"].to_numpy(float)
    mae = df["end2end_hse_MAE"].to_numpy(float)
    delta = df["delta_val_MAE_nonmetal"].to_numpy(float)

    # Ranking table: theta=0.16 can be retained because it minimizes the non-ZBG correction error
    # and remains close to the best end-to-end HSE MAE.
    df["rank_end2end_mae"] = df["end2end_hse_MAE"].rank(method="min", ascending=True)
    df["rank_delta_non_zbg_mae"] = df["delta_val_MAE_nonmetal"].rank(method="min", ascending=True)
    df["rank_rmse"] = df["end2end_hse_RMSE"].rank(method="min", ascending=True)
    df["rank_gate_f1"] = df["gate_f1"].rank(method="min", ascending=False)

    retained_mask = np.isclose(theta, args.retained_theta)
    if not np.any(retained_mask):
        raise ValueError(f"Retained theta {args.retained_theta} not present in theta values: {theta}")

    fig, (axA, axB) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(args.figwidth, args.figheight),
        sharex=True,
        constrained_layout=False,
    )

    # Panel A
    axA.plot(theta, mae, color=PALETTE["navy"], marker="o", linewidth=2.2, markersize=4.2)
    axA.axvline(args.retained_theta, color=PALETTE["ink"], linestyle=":", linewidth=1.0, alpha=0.9)
    axA.scatter(theta[retained_mask], mae[retained_mask], s=80, marker="*", color=PALETTE["gold"], edgecolor=PALETTE["ink"], linewidth=0.7, zorder=5)
    axA.set_ylabel("End-to-end HSE\nMAE (eV)")
    annotate_points(axA, theta, mae, dy=0.004)
    set_clean_ylim(axA, mae)
    add_panel_label(axA, "A")
    axA.text(
        args.retained_theta,
        0.96,
        rf"retained $\theta={args.retained_theta:.2f}$",
        transform=axA.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=8,
        bbox=dict(facecolor=PALETTE["bg"], edgecolor="none", alpha=0.75, pad=1.0),
    )

    # Panel B
    axB.plot(theta, delta, color=PALETTE["pink"], marker="o", linewidth=2.2, markersize=4.2)
    axB.axvline(args.retained_theta, color=PALETTE["ink"], linestyle=":", linewidth=1.0, alpha=0.9)
    axB.scatter(theta[retained_mask], delta[retained_mask], s=80, marker="*", color=PALETTE["gold"], edgecolor=PALETTE["ink"], linewidth=0.7, zorder=5)
    axB.set_ylabel("Δ-MAE for\nnon-ZBG candidates (eV)")
    axB.set_xlabel(r"Gate threshold $\theta$")
    annotate_points(axB, theta, delta, dy=0.0015)
    set_clean_ylim(axB, delta)
    add_panel_label(axB, "B")
    axB.text(
        args.retained_theta,
        0.96,
        rf"retained $\theta={args.retained_theta:.2f}$",
        transform=axB.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=8,
        bbox=dict(facecolor=PALETTE["bg"], edgecolor="none", alpha=0.75, pad=1.0),
    )

    for ax in (axA, axB):
        full_frame(ax)
        if show_grid:
            ax.grid(True, which="major", axis="both", color=PALETTE["light"], linewidth=0.7, alpha=0.6)
        ax.tick_params(length=3.5, width=1.0)
        ax.set_xlim(float(np.min(theta)) - 0.005, float(np.max(theta)) + 0.005)
        ax.yaxis.set_major_locator(MultipleLocator(0.02 if ax is axA else 0.005))

    axB.set_xticks(theta)
    axB.set_xticklabels([f"{t:.2f}" for t in theta])

    fig.suptitle(args.title, y=0.985, fontsize=10, fontweight="bold")
    fig.subplots_adjust(left=0.14, right=0.98, top=0.91, bottom=0.13, hspace=0.22)

    base = out_dir / args.out_prefix
    fig.savefig(str(base) + ".pdf", bbox_inches="tight")
    fig.savefig(str(base) + ".svg", bbox_inches="tight")
    fig.savefig(str(base) + ".png", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    # Tables: keep gate metrics for text/caption, but do not plot them.
    fixed_csv = out_dir / f"{args.out_prefix}_fixed_cycle{args.match_cycle}.csv"
    fixed_md = out_dir / f"{args.out_prefix}_fixed_cycle{args.match_cycle}.md"
    df.to_csv(fixed_csv, index=False)

    display_cols = [
        "theta", "fixed_cycle", "acquired_labels", "seed_labels", "total_labels",
        "end2end_hse_MAE", "end2end_hse_RMSE", "end2end_hse_R2",
        "delta_val_MAE_nonmetal", "delta_val_RMSE_nonmetal", "delta_val_R2_nonmetal",
        "gate_prec", "gate_rec", "gate_f1", "gate_auc", "gate_fp", "gate_fn",
        "rank_end2end_mae", "rank_delta_non_zbg_mae", "rank_rmse", "rank_gate_f1",
    ]
    display_cols = [c for c in display_cols if c in df.columns]
    md_df = df[display_cols].copy()
    for c in md_df.columns:
        if pd.api.types.is_float_dtype(md_df[c]):
            md_df[c] = md_df[c].map(lambda v: f"{v:.4f}" if np.isfinite(v) else "—")
    fixed_md.write_text(md_df.to_markdown(index=False), encoding="utf-8")

    retained = df[retained_mask].iloc[0]
    best_mae = df.loc[df["end2end_hse_MAE"].idxmin()]
    best_delta = df.loc[df["delta_val_MAE_nonmetal"].idxmin()]
    summary = pd.DataFrame([
        {"quantity": "retained_theta", "value": args.retained_theta},
        {"quantity": "retained_end2end_hse_MAE", "value": retained["end2end_hse_MAE"]},
        {"quantity": "retained_delta_MAE_non_ZBG", "value": retained["delta_val_MAE_nonmetal"]},
        {"quantity": "best_end2end_theta", "value": best_mae["theta"]},
        {"quantity": "best_end2end_hse_MAE", "value": best_mae["end2end_hse_MAE"]},
        {"quantity": "best_delta_theta", "value": best_delta["theta"]},
        {"quantity": "best_delta_MAE_non_ZBG", "value": best_delta["delta_val_MAE_nonmetal"]},
    ])
    summary.to_csv(out_dir / f"{args.out_prefix}_selection_summary.csv", index=False)

    print(f"[OK] Figure: {base}.pdf/.svg/.png")
    print(f"[OK] Fixed-cycle table: {fixed_csv}")
    print(f"[OK] Selection summary: {out_dir / f'{args.out_prefix}_selection_summary.csv'}")
    print("[INFO] Gate metrics were saved to the fixed-cycle table, not plotted.")
    print(f"[INFO] Retained theta={args.retained_theta:.2f}: e2e_MAE={retained['end2end_hse_MAE']:.4f}, delta_nonZBG_MAE={retained['delta_val_MAE_nonmetal']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
