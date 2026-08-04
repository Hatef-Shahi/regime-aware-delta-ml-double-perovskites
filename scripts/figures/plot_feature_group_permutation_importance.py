#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot feature-group permutation importance for the retained DOS-on AL workflow.

Visual revisions included:
1. Use "baseline-ZBG" instead of "PBE-ZBG" in figure footers.
2. Add "mean ± SD across five AL replicates" note directly to the figures.
3. Use explicit Δ-regression x-axis label:
   "Δ-MAE increase after permutation / eV"

Input:
  artifacts/pub_final_figures/fig_feature_group_permutation_importance/
      feature_group_permutation_importance_summary.csv
      feature_group_permutation_importance_per_run.csv

Outputs:
  artifacts/pub_final_figures/fig_feature_group_permutation_importance/
      fig_feature_group_permutation_importance_combined.png/pdf/svg
      fig_gate_classifier_feature_group_permutation_importance.png/pdf/svg
      fig_delta_regression_feature_group_permutation_importance.png/pdf/svg
      feature_group_permutation_importance_summary_rounded.csv
      metadata.json
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# Paths
# ----------------------------
FIG_DIR = Path("artifacts/pub_final_figures/fig_feature_group_permutation_importance")

SUMMARY_CSV = FIG_DIR / "feature_group_permutation_importance_summary.csv"
PER_RUN_CSV = FIG_DIR / "feature_group_permutation_importance_per_run.csv"

DPI = 600


# ----------------------------
# Global plotting style
# ----------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.titlesize": 12,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
})


def wrap_label(text: str, width: int = 30) -> str:
    """Wrap long y-axis labels."""
    return "\n".join(textwrap.wrap(str(text), width=width))


def safe_filename_branch(branch: str) -> str:
    """Convert branch label to safe filename."""
    return (
        branch.replace("Δ", "delta")
        .replace(" ", "_")
        .replace("-", "_")
        .lower()
    )


def signed_xlim(values: np.ndarray, errors: np.ndarray) -> tuple[float, float]:
    """Set x-limits that preserve negative permutation importance."""
    lo = np.nanmin(values - errors)
    hi = np.nanmax(values + errors)

    lo = min(0.0, lo)
    hi = max(0.0, hi)

    span = hi - lo
    pad = 0.12 * span if span > 0 else 0.01

    return lo - pad, hi + pad


def branch_sort_key(branch: str, sub: pd.DataFrame) -> pd.DataFrame:
    """Sort bars so the largest positive importance appears at the top."""
    return sub.sort_values("importance_mean", ascending=True)


def add_common_axis_formatting(ax) -> None:
    """Remove unnecessary frame elements and keep zero line visible."""
    ax.axvline(0.0, linewidth=1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)


def plot_single_branch(df: pd.DataFrame, branch: str, outdir: Path) -> None:
    """Plot one horizontal bar figure for one model branch."""
    sub = df[df["branch"] == branch].copy()

    if sub.empty:
        print(f"[WARN] no rows for branch: {branch}")
        return

    sub = branch_sort_key(branch, sub)

    labels = [wrap_label(x, 34) for x in sub["feature_group"]]
    y = np.arange(len(sub))

    values = sub["importance_mean"].to_numpy(float)
    errors = sub["importance_sd_across_runs"].fillna(0.0).to_numpy(float)

    fig_h = max(3.6, 0.58 * len(sub) + 1.6)
    fig, ax = plt.subplots(figsize=(7.4, fig_h), dpi=300)

    ax.barh(
    y,
    values,
    xerr=errors,
    capsize=2,
    linewidth=0.0,
    error_kw={
        "elinewidth": 0.8,
        "capthick": 0.8,
    },
)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)

    if branch == "Gate classifier":
        ax.set_xlabel("AUC decrease after permutation")
        ax.set_title("Gate classifier: feature-group permutation importance")
        footer_1 = "Held-out baseline-ZBG subset; n = 7 per replicate"
    elif branch == "Δ-regression":
        ax.set_xlabel("Δ-MAE increase after permutation / eV")
        ax.set_title("Δ-regression: feature-group permutation importance")
        footer_1 = "Held-out non-ZBG subset; n = 23 per replicate"
    else:
        ax.set_xlabel("Permutation importance")
        ax.set_title(f"{branch}: feature-group permutation importance")
        footer_1 = "Held-out test-set analysis"

    footer_2 = "Bars show mean ± SD across five AL replicates"

    ax.text(
        0.0,
        -0.23,
        footer_1 + "\n" + footer_2,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
    )

    ax.set_xlim(*signed_xlim(values, errors))
    add_common_axis_formatting(ax)

    fig.tight_layout()

    safe = safe_filename_branch(branch)

    for ext in ["png", "pdf", "svg"]:
        out = outdir / f"fig_{safe}_feature_group_permutation_importance.{ext}"
        fig.savefig(out, dpi=DPI, bbox_inches="tight")

    plt.close(fig)


def plot_combined(df: pd.DataFrame, outdir: Path) -> None:
    """Plot combined two-panel figure."""
    branches = ["Gate classifier", "Δ-regression"]
    panel_labels = ["a", "b"]

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(13.6, 4.9),
        dpi=300,
        constrained_layout=True,
    )

    for ax, branch, panel in zip(axes, branches, panel_labels):
        sub = df[df["branch"] == branch].copy()

        if sub.empty:
            ax.axis("off")
            ax.set_title(f"{panel}  {branch} unavailable")
            continue

        sub = branch_sort_key(branch, sub)

        labels = [wrap_label(x, 28) for x in sub["feature_group"]]
        y = np.arange(len(sub))

        values = sub["importance_mean"].to_numpy(float)
        errors = sub["importance_sd_across_runs"].fillna(0.0).to_numpy(float)

        ax.barh(y, values, xerr=errors, capsize=3, linewidth=0.0)

        ax.set_yticks(y)
        ax.set_yticklabels(labels)

        if branch == "Gate classifier":
            ax.set_xlabel("AUC decrease after permutation")
            ax.set_title(f"{panel}  Gate classifier")
            footer_1 = "baseline-ZBG test subset; n = 7 per replicate"
        else:
            ax.set_xlabel("Δ-MAE increase after permutation / eV")
            ax.set_title(f"{panel}  Δ-regression")
            footer_1 = "non-ZBG test subset; n = 23 per replicate"

        footer_2 = "mean ± SD across five AL replicates"

        ax.text(
            0.0,
            -0.23,
            footer_1 + "\n" + footer_2,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
        )

        ax.set_xlim(*signed_xlim(values, errors))
        add_common_axis_formatting(ax)

    for ext in ["png", "pdf", "svg"]:
        out = outdir / f"fig_feature_group_permutation_importance_combined.{ext}"
        fig.savefig(out, dpi=DPI, bbox_inches="tight")

    plt.close(fig)


def write_metadata(df: pd.DataFrame, outdir: Path) -> None:
    """Write figure metadata."""
    metadata = {
        "figure": "Feature-group permutation importance",
        "workflow": "DOS-on retained active-learning workflow",
        "cycle": 24,
        "runs": ["al1", "al2", "al3", "al4", "al5"],
        "evaluation": "held-out test set",
        "gate_metric": "AUC decrease after feature-group permutation",
        "delta_regression_metric": "Delta-MAE increase after feature-group permutation",
        "error_bars": "SD across five AL replicates",
        "n_repeats_per_run": 50,
        "pbe_zero_threshold_eV": 0.05,
        "metal_threshold_eV": 0.05,
        "terminology": {
            "baseline_ZBG": "PBE-zero-band-gap regime; E_g^PBE <= 0.05 eV",
            "non_ZBG": "E_g^PBE > 0.05 eV",
        },
        "branches": sorted(df["branch"].dropna().unique().tolist()),
        "feature_groups": sorted(df["feature_group"].dropna().unique().tolist()),
        "interpretation_note": (
            "Permutation importance quantifies model sensitivity under the retained "
            "active-learning-labelled distribution. It should not be interpreted as "
            "universal physical causality."
        ),
        "gate_subset_note": (
            "Gate importance is based on the held-out baseline-ZBG subset, which "
            "contains only seven structures per replicate; therefore, the gate ranking "
            "should be interpreted cautiously."
        ),
    }

    out = outdir / "metadata.json"
    out.write_text(json.dumps(metadata, indent=2))


def main() -> None:
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(f"Missing summary CSV: {SUMMARY_CSV}")

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(SUMMARY_CSV)

    required = {
        "branch",
        "metric",
        "feature_group",
        "importance_mean",
        "importance_sd_across_runs",
        "n_features",
        "n_eval",
        "baseline_metric",
    }

    missing = required.difference(df.columns)

    if missing:
        raise ValueError(f"Missing required columns in {SUMMARY_CSV}: {sorted(missing)}")

    # Clean rounded summary for SI/table inspection.
    rounded = df.copy()

    for col in ["importance_mean", "importance_sd_across_runs", "baseline_metric"]:
        rounded[col] = pd.to_numeric(rounded[col], errors="coerce").round(6)

    rounded.to_csv(
        FIG_DIR / "feature_group_permutation_importance_summary_rounded.csv",
        index=False,
    )

    # Single-branch figures.
    plot_single_branch(df, "Gate classifier", FIG_DIR)
    plot_single_branch(df, "Δ-regression", FIG_DIR)

    # Combined two-panel figure.
    plot_combined(df, FIG_DIR)

    # Metadata.
    write_metadata(df, FIG_DIR)

    print("[OK] wrote feature-importance plots to:")
    print(FIG_DIR)

    print("\nSummary:")
    print(rounded.to_string(index=False))


if __name__ == "__main__":
    main()
