#!/usr/bin/env python3
"""
Plot predicted HSE-level band gap vs uncertainty for the remaining candidate pool.

Purpose
-------
Visualize the remaining unlabelled pool after fixed-budget AL:

    x-axis: predicted HSE06 band gap
    y-axis: uncertainty sigma_delta
    color: predicted gap class

This version:
    - does NOT highlight recommended candidates
    - highlights both:
        1. narrow-gap region
        2. useful/visible-gap window
    - writes plotted points and summary CSVs

Typical use
-----------
python3 plots/fig_remaining_pool_hse_vs_uncertainty.py \
  --input artifacts/pub_final_tables/remaining_pool_candidates_dosON_consensus/table_remaining_pool_candidates_dosON_consensus_aggregated_predictions.csv \
  --narrow-min 0.05 \
  --narrow-max 1.0 \
  --visible-min 1.0 \
  --visible-max 2.5 \
  --sigma-quantile 0.75 \
  --out-dir artifacts/pub_final_figures/fig_remaining_pool_hse_uncertainty \
  --out-prefix fig_remaining_pool_hse_uncertainty_dosON \
  --dpi 600 \
  --figwidth 7.2 \
  --figheight 4.8
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PALETTE: Dict[str, str] = {
    "metal": "#8a8a8a",
    "narrow": "#ee9b00",
    "visible": "#0a9396",
    "wide": "#003566",
    "ink": "#111111",
    "light": "#d9d9d9",
    "narrow_band": "#ee9b00",
    "visible_band": "#0a9396",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Remaining-pool predicted HSE gap vs uncertainty plot"
    )

    p.add_argument("--input", required=True, help="Aggregated or filtered prediction CSV")

    # Band-gap windows
    p.add_argument("--narrow-min", type=float, default=0.05)
    p.add_argument("--narrow-max", type=float, default=1.0)
    p.add_argument("--visible-min", type=float, default=1.0)
    p.add_argument("--visible-max", type=float, default=2.5)

    # Backward-compatible aliases
    p.add_argument("--min-gap", type=float, default=None, help="Alias for --visible-min")
    p.add_argument("--max-gap", type=float, default=None, help="Alias for --visible-max")

    # uncertainty
    p.add_argument("--sigma-quantile", type=float, default=0.75)
    p.add_argument("--max-sigma", type=float, default=np.nan)

    p.add_argument("--out-dir", required=True)
    p.add_argument("--out-prefix", default="fig_remaining_pool_hse_uncertainty")

    p.add_argument("--title", default="Remaining-pool HSE-level predictions and uncertainty")
    p.add_argument("--dpi", type=int, default=600)
    p.add_argument("--figwidth", type=float, default=7.2)
    p.add_argument("--figheight", type=float, default=4.8)
    p.add_argument("--font", default="")

    return p.parse_args()


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def find_column(df: pd.DataFrame, expected: str) -> Optional[str]:
    exp = _norm(expected)
    mapping = {_norm(c): str(c) for c in df.columns}
    if exp in mapping:
        return mapping[exp]

    toks = [t for t in re.split(r"[_\W]+", str(expected).lower()) if t]
    for c in df.columns:
        cc = str(c).lower()
        if all(t in cc for t in toks):
            return str(c)

    return None


def choose_col(df: pd.DataFrame, candidates: Iterable[str], required: bool = True) -> Optional[str]:
    for cand in candidates:
        c = find_column(df, cand)
        if c is not None and c in df.columns:
            return c
    if required:
        raise KeyError(
            f"Could not find required column. Tried {list(candidates)}. "
            f"Available: {list(df.columns)}"
        )
    return None


def classify_gap(g: float, narrow_min: float, narrow_max: float, visible_min: float, visible_max: float) -> str:
    if not np.isfinite(g):
        return "invalid"
    if g <= narrow_min:
        return "metal/ZBG-like"
    if narrow_min < g < narrow_max:
        return "narrow-gap"
    if visible_min <= g <= visible_max:
        return "visible-range"
    return "wide-gap"


def setup_style(font: str = "") -> None:
    rc = {
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 1.05,
        "xtick.major.width": 1.05,
        "ytick.major.width": 1.05,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
    if font.strip():
        rc["font.family"] = font.strip()
    plt.rcParams.update(rc)


def main() -> None:
    args = parse_args()
    setup_style(args.font)

    # Backward-compatible aliases
    if args.min_gap is not None:
        args.visible_min = args.min_gap
    if args.max_gap is not None:
        args.visible_max = args.max_gap

    df = pd.read_csv(args.input)

    col_hse = choose_col(
        df,
        [
            "hse_gap_pred",
            "predicted_HSE_gap_eV",
            "predicted_hse_gap",
            "hse_pred",
            "hse_gap_model",
        ],
        required=True,
    )

    col_sigma = choose_col(
        df,
        [
            "sigma_delta",
            "uncertainty_sigma_eV",
            "sigma_gpr",
            "sigma",
            "uncertainty",
        ],
        required=True,
    )

    col_class = choose_col(df, ["predicted_class"], required=False)

    plot_df = df.copy()
    plot_df["hse_gap_pred_plot"] = pd.to_numeric(plot_df[col_hse], errors="coerce")
    plot_df["sigma_plot"] = pd.to_numeric(plot_df[col_sigma], errors="coerce")

    if col_class is not None:
        plot_df["class_plot"] = plot_df[col_class].astype(str)
    else:
        plot_df["class_plot"] = [
            classify_gap(
                g,
                narrow_min=args.narrow_min,
                narrow_max=args.narrow_max,
                visible_min=args.visible_min,
                visible_max=args.visible_max,
            )
            for g in plot_df["hse_gap_pred_plot"].to_numpy(float)
        ]

    # Reclassify to ensure consistency with requested windows
    plot_df["class_plot"] = [
        classify_gap(
            g,
            narrow_min=args.narrow_min,
            narrow_max=args.narrow_max,
            visible_min=args.visible_min,
            visible_max=args.visible_max,
        )
        for g in plot_df["hse_gap_pred_plot"].to_numpy(float)
    ]

    plot_df = plot_df.loc[
        np.isfinite(plot_df["hse_gap_pred_plot"])
        & np.isfinite(plot_df["sigma_plot"])
    ].copy()

    if plot_df.empty:
        raise RuntimeError("No finite predicted HSE gap / uncertainty points found.")

    if np.isfinite(args.max_sigma):
        sigma_cut = float(args.max_sigma)
    else:
        q = min(max(float(args.sigma_quantile), 0.0), 1.0)
        sigma_cut = float(plot_df["sigma_plot"].quantile(q))

    class_order = ["metal/ZBG-like", "narrow-gap", "visible-range", "wide-gap"]
    class_colors = {
        "metal/ZBG-like": PALETTE["metal"],
        "narrow-gap": PALETTE["narrow"],
        "visible-range": PALETTE["visible"],
        "wide-gap": PALETTE["wide"],
    }

    fig, ax = plt.subplots(figsize=(args.figwidth, args.figheight), dpi=300)

    # Shaded regions
    ax.axvspan(
        args.narrow_min,
        args.narrow_max,
        color=PALETTE["narrow_band"],
        alpha=0.11,
        linewidth=0,
        label=f"narrow-gap region ({args.narrow_min:.2f}–{args.narrow_max:.1f} eV)",
    )

    ax.axvspan(
        args.visible_min,
        args.visible_max,
        color=PALETTE["visible_band"],
        alpha=0.10,
        linewidth=0,
        label=f"useful gap window ({args.visible_min:.1f}–{args.visible_max:.1f} eV)",
    )

    # Uncertainty cutoff
    ax.axhline(
        sigma_cut,
        color=PALETTE["ink"],
        linestyle="--",
        linewidth=1.1,
        label=f"uncertainty cutoff = {sigma_cut:.3f} eV",
    )

    # Scatter by class
    for cls in class_order:
        sub = plot_df.loc[plot_df["class_plot"] == cls]
        if sub.empty:
            continue
        ax.scatter(
            sub["hse_gap_pred_plot"],
            sub["sigma_plot"],
            s=24,
            color=class_colors[cls],
            alpha=0.58,
            edgecolors="none",
            label=f"{cls} (n={len(sub)})",
        )

    other = plot_df.loc[~plot_df["class_plot"].isin(class_order)]
    if not other.empty:
        ax.scatter(
            other["hse_gap_pred_plot"],
            other["sigma_plot"],
            s=22,
            color="#bbbbbb",
            alpha=0.45,
            edgecolors="none",
            label=f"other (n={len(other)})",
        )

    ax.set_xlabel("Predicted HSE06 band gap (eV)")
    ax.set_ylabel(r"Predicted uncertainty $\sigma_\Delta$ (eV)")
    ax.set_title(args.title)
    ax.grid(True, color=PALETTE["light"], alpha=0.35)

    # Counts
    n_total = len(plot_df)
    n_narrow = int(
        (
            (plot_df["hse_gap_pred_plot"] > args.narrow_min)
            & (plot_df["hse_gap_pred_plot"] < args.narrow_max)
        ).sum()
    )
    n_visible = int(
        (
            (plot_df["hse_gap_pred_plot"] >= args.visible_min)
            & (plot_df["hse_gap_pred_plot"] <= args.visible_max)
        ).sum()
    )
    n_narrow_low_sigma = int(
        (
            (plot_df["hse_gap_pred_plot"] > args.narrow_min)
            & (plot_df["hse_gap_pred_plot"] < args.narrow_max)
            & (plot_df["sigma_plot"] <= sigma_cut)
        ).sum()
    )
    n_visible_low_sigma = int(
        (
            (plot_df["hse_gap_pred_plot"] >= args.visible_min)
            & (plot_df["hse_gap_pred_plot"] <= args.visible_max)
            & (plot_df["sigma_plot"] <= sigma_cut)
        ).sum()
    )

    ax.text(
        0.03,
        0.97,
        f"remaining-pool predictions: n={n_total}\n"
        f"narrow-gap region: n={n_narrow}; low/moderate σ: n={n_narrow_low_sigma}\n"
        f"useful-gap window: n={n_visible}; low/moderate σ: n={n_visible_low_sigma}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.0,
        bbox=dict(facecolor="white", edgecolor=PALETTE["light"], alpha=0.90, pad=2.0),
    )

    ax.legend(
        loc="upper right",
        frameon=True,
        fontsize=7.3,
        borderpad=0.35,
        handletextpad=0.4,
    )

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / args.out_prefix

    for ext in [".png", ".pdf", ".svg"]:
        fig.savefig(base.with_suffix(ext), dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    plot_df.to_csv(base.with_name(base.name + "_points.csv"), index=False)

    summary = pd.DataFrame([{
        "n_points": n_total,
        "n_narrow_gap_region": n_narrow,
        "n_narrow_gap_low_moderate_sigma": n_narrow_low_sigma,
        "n_useful_gap_window": n_visible,
        "n_useful_gap_low_moderate_sigma": n_visible_low_sigma,
        "sigma_cutoff": sigma_cut,
        "narrow_min": args.narrow_min,
        "narrow_max": args.narrow_max,
        "visible_min": args.visible_min,
        "visible_max": args.visible_max,
        "sigma_quantile": args.sigma_quantile,
    }])
    summary.to_csv(base.with_name(base.name + "_summary.csv"), index=False)

    print("[OK] wrote:")
    print(" ", base.with_suffix(".png"))
    print(" ", base.with_suffix(".pdf"))
    print(" ", base.with_suffix(".svg"))
    print(" ", base.with_name(base.name + "_points.csv"))
    print(" ", base.with_name(base.name + "_summary.csv"))
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()