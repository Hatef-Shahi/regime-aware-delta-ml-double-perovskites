#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Publication-style K_g quota-sensitivity plot.

Compares the allocation of each 7-member acquisition batch between:
  K_g = ZBG/gate candidates
  K_r = non-ZBG/regression candidates

Expected layout from experiments root:
  k-gate/kg1/metrics/metrics.csv
  k-gate/kg2/metrics/metrics.csv
  k-gate/kg3/metrics/metrics.csv

Main figure:
  A) End-to-end HSE MAE vs quota allocation
  B) Delta-MAE for non-ZBG candidates vs quota allocation

Gate metrics are written to tables but not plotted.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

PALETTE: Dict[str, str] = {
    "navy": "#003566",
    "pink": "#ff4d6d",
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
    "train_n_total": ["train_n_total"],
    "train_n_nonmetal_used": ["train_n_nonmetal_used"],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Publication-style K_g quota-sensitivity plot")
    p.add_argument("--root", type=str, default=".")
    p.add_argument("--runs-dir", type=str, default="k-gate")
    p.add_argument("--out-dir", type=str, default="artifacts/fig_kgate_sensitivity")
    p.add_argument("--out-prefix", type=str, default="fig_kgate_sensitivity_summary")
    p.add_argument("--metrics-file", type=str, default="metrics/metrics.csv")
    p.add_argument("--match-cycle", type=int, default=24)
    p.add_argument("--seed-labels", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=7)
    p.add_argument("--retained-kg", type=int, default=2)
    p.add_argument("--title", type=str, default="ZBG-quota sensitivity")
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


def parse_kg_from_dir(run_dir: Path) -> Optional[int]:
    name = run_dir.name.lower()
    m = re.search(r"(?:^|[_-])kg(?:[_-])?(\d+)(?=$|[_-])", name)
    if m:
        return int(m.group(1))
    m = re.search(r"(?:^|[_-])k[_-]?gate(?:[_-])?(\d+)(?=$|[_-])", name)
    if m:
        return int(m.group(1))
    return None


def discover_runs(root: Path, runs_dir: str) -> Dict[int, Path]:
    base = root / runs_dir
    if not base.exists():
        raise FileNotFoundError(f"Missing runs directory: {base}")
    out: Dict[int, Path] = {}
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        kg = parse_kg_from_dir(d)
        if kg is not None:
            out[int(kg)] = d
    if not out:
        raise RuntimeError(f"No K_g run directories found in {base}; expected names like kg1, kg2, kg3")
    return out


def read_metrics_at_cycle(run_dir: Path, rel_metrics_file: str, cycle: int) -> Dict[str, float]:
    path = run_dir / rel_metrics_file
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics file: {path}")
    df = pd.read_csv(path)
    cyc_col = find_column(df, "cycle") or find_column(df, "al_cycle") or find_column(df, "iteration")
    if cyc_col is None:
        df["cycle"] = np.arange(len(df), dtype=float)
        cyc_col = "cycle"
    df[cyc_col] = pd.to_numeric(df[cyc_col], errors="coerce")
    matches = df[np.isclose(df[cyc_col].astype(float), float(cycle))]
    if matches.empty:
        raise ValueError(f"Cycle {cycle} not found in {path}; available cycles: {df[cyc_col].dropna().tolist()}")
    row = matches.iloc[-1]
    out: Dict[str, float] = {}
    for canonical, candidates in METRIC_CANDIDATES.items():
        col = find_first_column(df, candidates)
        if col is None:
            out[canonical] = np.nan
        else:
            out[canonical] = float(pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0])
    return out


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.01,
        0.94,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
        bbox=dict(facecolor=PALETTE["bg"], edgecolor="none", alpha=0.75, pad=1.2),
    )


def annotate_values(ax: plt.Axes, x: np.ndarray, y: np.ndarray, dy: float = 0.0) -> None:
    for xi, yi in zip(x, y):
        if not np.isfinite(yi):
            continue
        ax.text(
            xi,
            yi + dy,
            f"{yi:.3f}",
            ha="center",
            va="bottom" if dy >= 0 else "top",
            fontsize=8,
            color=PALETTE["ink"],
        )


def plot_summary(df: pd.DataFrame, args: argparse.Namespace, out_dir: Path) -> None:
    df = df.sort_values("k_gate").reset_index(drop=True)
    x = df["k_gate"].to_numpy(float)
    mae = df["end2end_hse_MAE"].to_numpy(float)
    dnm = df["delta_val_MAE_nonmetal"].to_numpy(float)

    show_grid = not args.no_grid
    if args.grid:
        show_grid = True

    fig, (axA, axB) = plt.subplots(
        2,
        1,
        figsize=(args.figwidth, args.figheight),
        sharex=True,
        constrained_layout=False,
    )

    retained = int(args.retained_kg)
    if retained in set(df["k_gate"].astype(int)):
        retained_row = df[df["k_gate"].astype(int) == retained].iloc[0]
        retained_kr = int(retained_row["k_reg"])
        retained_label = f"retained: {retained} ZBG + {retained_kr} non-ZBG"
    else:
        retained_label = f"retained $K_g={retained}$"

    for ax in (axA, axB):
        ax.axvline(retained, color=PALETTE["ink"], linestyle=":", linewidth=1.0, alpha=0.85, zorder=1)
        ax.text(
            retained,
            0.96,
            retained_label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8,
            bbox=dict(facecolor=PALETTE["bg"], edgecolor="none", alpha=0.78, pad=1.3),
        )
        if show_grid:
            ax.grid(True, which="major", axis="both", color=PALETTE["light"], linewidth=0.7, alpha=0.55)
        full_frame(ax)
        ax.tick_params(length=3.5, width=1.0)
        ax.set_xlim(min(x) - 0.15, max(x) + 0.15)

    axA.plot(x, mae, marker="o", markersize=5.0, linewidth=2.2, color=PALETTE["navy"])
    axB.plot(x, dnm, marker="o", markersize=5.0, linewidth=2.2, color=PALETTE["pink"])

    # retained markers
    retained_mask = np.isclose(x, float(retained))
    if retained_mask.any():
        axA.scatter(x[retained_mask], mae[retained_mask], s=110, marker="*", color=PALETTE["gold"], edgecolor=PALETTE["ink"], linewidth=0.7, zorder=5)
        axB.scatter(x[retained_mask], dnm[retained_mask], s=110, marker="*", color=PALETTE["gold"], edgecolor=PALETTE["ink"], linewidth=0.7, zorder=5)

    annotate_values(axA, x, mae, dy=0.003)
    annotate_values(axB, x, dnm, dy=0.0015)

    axA.set_ylabel("End-to-end HSE\nMAE (eV)")
    axB.set_ylabel("Δ-MAE for\nnon-ZBG candidates (eV)")
    axB.set_xlabel("Quota allocation per 7-label cycle")

    tick_labels = []
    for _, row in df.iterrows():
        kg = int(row["k_gate"])
        kr = int(row["k_reg"])
        tick_labels.append(f"{kg} ZBG\n{kr} non-ZBG")
    axB.set_xticks(x)
    axB.set_xticklabels(tick_labels)

    axA.yaxis.set_major_locator(MultipleLocator(0.05))
    axA.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    axB.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))

    # Add modest y-padding after labels are known.
    for ax, y in [(axA, mae), (axB, dnm)]:
        finite = y[np.isfinite(y)]
        if finite.size:
            lo, hi = float(np.min(finite)), float(np.max(finite))
            pad = 0.16 * (hi - lo) if hi > lo else 0.02
            ax.set_ylim(max(0.0, lo - pad), hi + pad)

    add_panel_label(axA, "A")
    add_panel_label(axB, "B")

    fig.suptitle(args.title, y=0.995, fontsize=10, fontweight="bold")
    fig.subplots_adjust(left=0.12, right=0.98, top=0.90, bottom=0.18, hspace=0.34)

    base = out_dir / args.out_prefix

    # extra top padding for value labels
    for _ax in fig.axes:
        try:
            _lo, _hi = _ax.get_ylim()
            _span = _hi - _lo
            if _span > 0:
                _ax.set_ylim(_lo, _hi + max(0.010, 0.28 * _span))
        except Exception:
            pass

    fig.savefig(str(base) + ".pdf", bbox_inches="tight")
    fig.savefig(str(base) + ".svg", bbox_inches="tight")
    fig.savefig(str(base) + ".png", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)


def df_to_md_table(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for _, row in df.iterrows():
        vals = []
        for c in headers:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                vals.append(f"{float(v):.4f}" if np.isfinite(float(v)) else "n/a")
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

    run_dirs = discover_runs(root, args.runs_dir)
    rows = []
    acquired = int(args.match_cycle * args.batch_size)
    total = int(args.seed_labels + acquired)

    for kg, rd in sorted(run_dirs.items()):
        metrics = read_metrics_at_cycle(rd, args.metrics_file, args.match_cycle)
        row = {
            "k_gate": int(kg),
            "k_reg": int(args.batch_size - kg),
            "fixed_cycle": int(args.match_cycle),
            "acquired_labels": acquired,
            "seed_labels": int(args.seed_labels),
            "total_labels": total,
        }
        row.update(metrics)
        rows.append(row)

    fixed_df = pd.DataFrame(rows).sort_values("k_gate").reset_index(drop=True)
    # Ranking kept for the table only. The retained point is controlled by --retained-kg.
    fixed_df["rank_by_end2end_mae"] = fixed_df["end2end_hse_MAE"].rank(method="min", ascending=True).astype(int)
    fixed_df["rank_by_nonZBG_delta_mae"] = fixed_df["delta_val_MAE_nonmetal"].rank(method="min", ascending=True).astype(int)

    fixed_csv = out_dir / f"{args.out_prefix}_fixed_cycle{args.match_cycle}.csv"
    fixed_md = out_dir / f"{args.out_prefix}_fixed_cycle{args.match_cycle}.md"
    fixed_df.to_csv(fixed_csv, index=False)
    fixed_md.write_text(df_to_md_table(fixed_df), encoding="utf-8")

    plot_summary(fixed_df, args, out_dir)

    print(f"[OK] Wrote: {out_dir / (args.out_prefix + '.pdf')}")
    print(f"[OK] Wrote: {out_dir / (args.out_prefix + '.svg')}")
    print(f"[OK] Wrote: {out_dir / (args.out_prefix + '.png')}")
    print(f"[OK] Wrote: {fixed_csv}")
    print("[INFO] Retained allocation is specified by --retained-kg, not automatically chosen from test/validation ranking.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
