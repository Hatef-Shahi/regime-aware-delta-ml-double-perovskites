#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Per-structure mean ± replicate-SD parity plot.

Each unique structure is plotted once:
    x = true value
    y = mean prediction across AL replicate models
    vertical error bar = ±1 SD across replicate predictions

Supported modes
---------------
Test:
    <run>/test_eval/test_pred_cycle<CYCLE>.csv

Validation:
    <run>/metrics/val_pred_cycle<CYCLE>.csv
    <run>/predictions/val_pred_cycle<CYCLE>.csv
    <run>/predictions/validation_predictions_cycle<CYCLE>.csv
    <run>/predictions/all_predictions_cycle<CYCLE>.csv

Outputs
-------
<out-dir>/<out-prefix>.png
<out-dir>/<out-prefix>.pdf
<out-dir>/<out-prefix>.svg
<out-dir>/<out-prefix>_pooled_points.csv
<out-dir>/<out-prefix>_structure_summary.csv
<out-dir>/<out-prefix>_metrics.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PALETTE: Dict[str, str] = {
    "navy": "#003566",
    "teal": "#0A9396",
    "gold": "#EE9B00",
    "red": "#FF4D6D",
    "green": "#2D6A4F",
    "gray": "#888888",
    "light": "#D9D9D9",
    "ink": "#111111",
    "bg": "#FFFFFF",
}


# =============================================================================
# Command-line arguments
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot one mean prediction per unique structure with "
            "vertical replicate-SD error bars."
        )
    )

    parser.add_argument("--root", default=".")
    parser.add_argument("--runs-dir", required=True)
    parser.add_argument("--run-glob", default="al*")
    parser.add_argument("--mode", choices=["val", "test"], default="test")
    parser.add_argument("--cycle", type=int, default=24)

    parser.add_argument("--pbe-zero-thr", type=float, default=0.05)
    parser.add_argument("--metal-thr", type=float, default=0.05)
    parser.add_argument("--gate-decision-thr", type=float, default=0.16)

    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--out-prefix", default="fig_parity_mean_sd")
    parser.add_argument("--title", default="")

    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--figwidth", type=float, default=11.8)
    parser.add_argument("--figheight", type=float, default=5.2)

    parser.add_argument("--hse-band", type=float, default=0.50)
    parser.add_argument("--delta-band", type=float, default=0.25)
    parser.add_argument("--delta-min", type=float, default=0.25)
    parser.add_argument("--delta-max", type=float, default=2.00)

    parser.add_argument("--grid", action="store_true")
    parser.add_argument("--font", default="")

    return parser.parse_args()


# =============================================================================
# General utilities
# =============================================================================

def style_matplotlib(font: str = "") -> None:
    settings = {
        "figure.facecolor": PALETTE["bg"],
        "axes.facecolor": PALETTE["bg"],
        "axes.edgecolor": PALETTE["ink"],
        "axes.labelcolor": PALETTE["ink"],
        "xtick.color": PALETTE["ink"],
        "ytick.color": PALETTE["ink"],
        "text.color": PALETTE["ink"],
        "axes.linewidth": 1.05,
        "xtick.major.width": 1.05,
        "ytick.major.width": 1.05,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "font.size": 9,
        "axes.titlesize": 10.5,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }

    if font.strip():
        settings["font.family"] = font.strip()

    plt.rcParams.update(settings)


def natural_key(text: str) -> List[object]:
    """Natural sorting: al1, al2, ..., al10."""
    return [
        int(token) if token.isdigit() else token.lower()
        for token in re.split(r"(\d+)", str(text))
    ]


def normalize_name(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def find_column(
    df: pd.DataFrame,
    expected: str,
) -> Optional[str]:
    normalized_expected = normalize_name(expected)
    mapping = {
        normalize_name(column): str(column)
        for column in df.columns
    }

    if normalized_expected in mapping:
        return mapping[normalized_expected]

    tokens = [
        token
        for token in re.split(r"[_\W]+", expected.lower())
        if token
    ]

    for column in df.columns:
        column_lower = str(column).lower()

        if all(token in column_lower for token in tokens):
            return str(column)

    return None


def choose_numeric(
    df: pd.DataFrame,
    candidates: Iterable[str],
    required: bool = True,
) -> Optional[str]:
    best_column = None
    best_finite_count = -1

    for candidate in candidates:
        column = find_column(df, candidate)

        if column is None or column not in df.columns:
            continue

        values = pd.to_numeric(
            df[column],
            errors="coerce",
        ).to_numpy(float)

        finite_count = int(np.isfinite(values).sum())

        if finite_count > best_finite_count:
            best_column = column
            best_finite_count = finite_count

    if best_column is None and required:
        raise KeyError(
            "Missing required numeric column.\n"
            f"Tried: {list(candidates)}\n"
            f"Available columns: {list(df.columns)}"
        )

    return best_column


def choose_any(
    df: pd.DataFrame,
    candidates: Iterable[str],
    required: bool = False,
) -> Optional[str]:
    best_column = None
    best_nonnull_count = -1

    for candidate in candidates:
        column = find_column(df, candidate)

        if column is None or column not in df.columns:
            continue

        nonnull_count = int(df[column].notna().sum())

        if nonnull_count > best_nonnull_count:
            best_column = column
            best_nonnull_count = nonnull_count

    if best_column is None and required:
        raise KeyError(
            "Missing required column.\n"
            f"Tried: {list(candidates)}\n"
            f"Available columns: {list(df.columns)}"
        )

    return best_column


def to_numeric_array(
    df: pd.DataFrame,
    column: Optional[str],
    default: float = np.nan,
) -> np.ndarray:
    if column is None or column not in df.columns:
        return np.full(
            len(df),
            default,
            dtype=float,
        )

    return pd.to_numeric(
        df[column],
        errors="coerce",
    ).to_numpy(float)


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Tuple[float, float, float, int]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    n_valid = int(valid.sum())

    if n_valid == 0:
        return np.nan, np.nan, np.nan, 0

    true_values = y_true[valid]
    predicted_values = y_pred[valid]
    error = predicted_values - true_values

    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error ** 2)))

    ss_residual = float(np.sum(error ** 2))
    ss_total = float(
        np.sum(
            (true_values - np.mean(true_values)) ** 2
        )
    )

    r2 = (
        float(1.0 - ss_residual / ss_total)
        if ss_total > 0.0
        else np.nan
    )

    return mae, rmse, r2, n_valid


def parity_limits(
    x: np.ndarray,
    y_mean: np.ndarray,
    y_sd: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y_mean = np.asarray(y_mean, dtype=float)

    if y_sd is None:
        y_sd = np.zeros_like(y_mean)
    else:
        y_sd = np.asarray(y_sd, dtype=float)

    valid = (
        np.isfinite(x)
        & np.isfinite(y_mean)
        & np.isfinite(y_sd)
    )

    if not np.any(valid):
        return 0.0, 1.0

    lower_predictions = y_mean[valid] - y_sd[valid]
    upper_predictions = y_mean[valid] + y_sd[valid]

    lower = float(
        min(
            np.min(x[valid]),
            np.min(lower_predictions),
        )
    )

    upper = float(
        max(
            np.max(x[valid]),
            np.max(upper_predictions),
        )
    )

    padding = (
        0.06 * (upper - lower)
        if upper > lower
        else 0.2
    )

    return lower - padding, upper + padding


def add_parity_guides(
    ax: plt.Axes,
    limits: Tuple[float, float],
    band: float,
) -> None:
    lower, upper = limits
    x_line = np.linspace(lower, upper, 300)

    ax.plot(
        x_line,
        x_line,
        color=PALETTE["gray"],
        linewidth=1.2,
        zorder=0,
    )

    ax.plot(
        x_line,
        x_line + band,
        color=PALETTE["light"],
        linewidth=0.9,
        linestyle="--",
        zorder=0,
    )

    ax.plot(
        x_line,
        x_line - band,
        color=PALETTE["light"],
        linewidth=0.9,
        linestyle="--",
        zorder=0,
    )


def add_optional_grid(
    ax: plt.Axes,
    show_grid: bool,
) -> None:
    if show_grid:
        ax.grid(
            True,
            color=PALETTE["light"],
            alpha=0.35,
            linewidth=0.7,
            zorder=0,
        )


def save_figure(
    figure: plt.Figure,
    output_base: Path,
    dpi: int,
) -> None:
    output_base.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    for extension in [".png", ".pdf", ".svg"]:
        figure.savefig(
            output_base.with_suffix(extension),
            dpi=dpi,
            bbox_inches="tight",
        )

    plt.close(figure)


# =============================================================================
# Prediction-file discovery
# =============================================================================

def prediction_file_for_mode(
    run_directory: Path,
    mode: str,
    cycle: int,
) -> Path:
    if mode == "val":
        candidates = [
            run_directory
            / "metrics"
            / f"val_pred_cycle{cycle}.csv",

            run_directory
            / "predictions"
            / f"val_pred_cycle{cycle}.csv",

            run_directory
            / "predictions"
            / f"validation_predictions_cycle{cycle}.csv",

            run_directory
            / "predictions"
            / f"all_predictions_cycle{cycle}.csv",
        ]
    else:
        candidates = [
            run_directory
            / "test_eval"
            / f"test_pred_cycle{cycle}.csv",

            run_directory
            / "metrics"
            / f"test_pred_cycle{cycle}.csv",

            run_directory
            / "predictions"
            / f"test_pred_cycle{cycle}.csv",
        ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    possible_files = sorted(
        run_directory.rglob(f"*cycle{cycle}*.csv")
    )

    if mode == "val":
        possible_files = [
            path
            for path in possible_files
            if (
                "val_pred" in path.name.lower()
                or "validation" in path.name.lower()
            )
        ]
    else:
        possible_files = [
            path
            for path in possible_files
            if (
                "test_pred" in path.name.lower()
                or "test" in str(path.parent).lower()
            )
        ]

    if possible_files:
        return possible_files[0]

    raise FileNotFoundError(
        f"Could not locate a {mode} prediction CSV for "
        f"cycle {cycle} under {run_directory}"
    )


# =============================================================================
# Prediction normalization
# =============================================================================

def normalize_prediction_file(
    raw: pd.DataFrame,
    run_name: str,
    args: argparse.Namespace,
) -> pd.DataFrame:
    id_column = choose_any(
        raw,
        [
            "structure_id",
            "material_id",
            "mp_id",
            "id",
            "name",
        ],
        required=False,
    )

    hse_true_column = choose_numeric(
        raw,
        [
            "hse_true",
            "hse_gap_true",
            "true_hse",
            "y_true_hse",
            "hse_gap",
        ],
    )

    hse_pred_column = choose_numeric(
        raw,
        [
            "hse_pred",
            "hse_gap_pred",
            "pred_hse",
            "y_pred_hse",
            "end2end_hse_pred",
            "pred_gap",
        ],
    )

    pbe_column = choose_numeric(
        raw,
        [
            "pbe_gap_effective",
            "pbe_gap_eff",
            "pbe_gap",
            "pbe_true",
            "pbe_gap_true",
            "baseline_gap",
            "pbe_baseline",
            "pbe_baseline_gap",
            "g_b",
        ],
    )

    delta_true_column = choose_numeric(
        raw,
        [
            "delta_true",
            "true_delta",
            "delta",
        ],
        required=False,
    )

    delta_pred_column = choose_numeric(
        raw,
        [
            "delta_pred_stageB",
            "delta_pred",
            "pred_delta",
            "delta_hat",
            "predicted_delta",
        ],
        required=False,
    )

    pbe_zero_flag_column = choose_numeric(
        raw,
        [
            "is_pbe0",
            "is_pbe_zero",
            "baseline_zbg",
            "is_zbg",
            "is_PBE_ZBG",
        ],
        required=False,
    )

    gate_probability_column = choose_numeric(
        raw,
        [
            "p_gate",
            "gate_probability",
            "prob_gate",
            "gate_prob",
        ],
        required=False,
    )

    label_status_column = choose_any(
        raw,
        [
            "label_status",
            "status",
            "hse_label_status",
        ],
        required=False,
    )

    # Critical fix:
    # initialize the output with the same number of rows before assigning
    # the scalar run name.
    normalized = pd.DataFrame(
        index=np.arange(len(raw))
    )

    normalized["structure_id"] = (
        raw[id_column]
        .astype(str)
        .str.strip()
        .to_numpy()
        if id_column is not None
        else np.arange(len(raw)).astype(str)
    )

    normalized["run"] = str(run_name)

    normalized["hse_true"] = to_numeric_array(
        raw,
        hse_true_column,
    )

    normalized["hse_pred"] = to_numeric_array(
        raw,
        hse_pred_column,
    )

    normalized["pbe_gap_effective"] = to_numeric_array(
        raw,
        pbe_column,
    )

    if delta_true_column is not None:
        normalized["delta_true"] = to_numeric_array(
            raw,
            delta_true_column,
        )
    else:
        normalized["delta_true"] = (
            normalized["hse_true"]
            - normalized["pbe_gap_effective"]
        )

    if delta_pred_column is not None:
        normalized["delta_pred"] = to_numeric_array(
            raw,
            delta_pred_column,
        )
    else:
        normalized["delta_pred"] = (
            normalized["hse_pred"]
            - normalized["pbe_gap_effective"]
        )

    normalized["is_pbe_zero_flag"] = to_numeric_array(
        raw,
        pbe_zero_flag_column,
        default=np.nan,
    )

    normalized["p_gate"] = to_numeric_array(
        raw,
        gate_probability_column,
        default=np.nan,
    )

    if label_status_column is not None:
        normalized["label_status"] = (
            raw[label_status_column]
            .astype(str)
            .to_numpy()
        )
    else:
        normalized["label_status"] = ""

    # -------------------------------------------------------------------------
    # Baseline-ZBG assignment
    # -------------------------------------------------------------------------
    normalized["is_baseline_zbg"] = (
        (
            np.isfinite(
                normalized["pbe_gap_effective"]
            )
            & (
                normalized["pbe_gap_effective"]
                <= args.pbe_zero_thr
            )
        )
        |
        (
            np.isfinite(
                normalized["is_pbe_zero_flag"]
            )
            & (
                normalized["is_pbe_zero_flag"] == 1
            )
        )
    )

    # -------------------------------------------------------------------------
    # HSE-level truth classes in the baseline-ZBG regime
    # -------------------------------------------------------------------------
    normalized["is_recoverable_zbg"] = (
        normalized["is_baseline_zbg"]
        & np.isfinite(normalized["hse_true"])
        & (normalized["hse_true"] > args.metal_thr)
    )

    normalized["is_true_zbg_metal"] = (
        normalized["is_baseline_zbg"]
        & np.isfinite(normalized["hse_true"])
        & (normalized["hse_true"] <= args.metal_thr)
    )

    # -------------------------------------------------------------------------
    # Gate decision
    # -------------------------------------------------------------------------
    has_gate_probability = bool(
        np.isfinite(normalized["p_gate"]).any()
    )

    if has_gate_probability:
        normalized["gate_rescued"] = (
            np.isfinite(normalized["p_gate"])
            & (
                normalized["p_gate"]
                >= args.gate_decision_thr
            )
        )
    else:
        # Fallback for older prediction tables.
        normalized["gate_rescued"] = (
            np.isfinite(normalized["hse_pred"])
            & (
                normalized["hse_pred"]
                > args.metal_thr
            )
        )

    normalized["correct_rescue_zbg_nonmetal"] = (
        normalized["is_recoverable_zbg"]
        & normalized["gate_rescued"]
    )

    normalized["missed_zbg_nonmetal"] = (
        normalized["is_recoverable_zbg"]
        & (~normalized["gate_rescued"])
    )

    normalized["false_rescue_zbg_metal"] = (
        normalized["is_true_zbg_metal"]
        & normalized["gate_rescued"]
    )

    normalized["correctly_unrecovered_zbg_metal"] = (
        normalized["is_true_zbg_metal"]
        & (~normalized["gate_rescued"])
    )

    normalized["zbg_correct_gate_outcome"] = (
        normalized["correct_rescue_zbg_nonmetal"]
        | normalized["correctly_unrecovered_zbg_metal"]
    )

    normalized["zbg_incorrect_gate_outcome"] = (
        normalized["missed_zbg_nonmetal"]
        | normalized["false_rescue_zbg_metal"]
    )

    # -------------------------------------------------------------------------
    # Non-ZBG Δ-panel mask
    # -------------------------------------------------------------------------
    label_status_upper = (
        normalized["label_status"]
        .astype(str)
        .str.upper()
    )

    normalized["used_in_delta_panel"] = (
        (~normalized["is_baseline_zbg"])
        & np.isfinite(normalized["hse_true"])
        & np.isfinite(normalized["delta_true"])
        & np.isfinite(normalized["delta_pred"])
        & (normalized["hse_true"] > args.metal_thr)
        & (
            ~label_status_upper.str.contains(
                "METAL_ASSUMED",
                na=False,
            )
        )
    )

    return normalized


# =============================================================================
# Pool all replicate prediction files
# =============================================================================

def load_pooled_predictions(
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, List[str]]:
    root = Path(args.root).resolve()
    runs_base = root / args.runs_dir

    run_directories = sorted(
        [
            directory
            for directory in runs_base.glob(args.run_glob)
            if directory.is_dir()
        ],
        key=lambda path: natural_key(path.name),
    )

    if not run_directories:
        raise FileNotFoundError(
            f"No run directories found under {runs_base} "
            f"matching {args.run_glob}"
        )

    frames: List[pd.DataFrame] = []
    run_names: List[str] = []

    for run_directory in run_directories:
        prediction_path = prediction_file_for_mode(
            run_directory=run_directory,
            mode=args.mode,
            cycle=args.cycle,
        )

        raw = pd.read_csv(prediction_path)

        normalized = normalize_prediction_file(
            raw=raw,
            run_name=run_directory.name,
            args=args,
        )

        normalized["source_file"] = str(prediction_path)

        frames.append(normalized)
        run_names.append(run_directory.name)

        print(
            f"[LOAD] {run_directory.name}: "
            f"{prediction_path}"
        )

    pooled = pd.concat(
        frames,
        ignore_index=True,
    )

    valid_main = (
        np.isfinite(pooled["hse_true"])
        & np.isfinite(pooled["hse_pred"])
    )

    pooled = pooled.loc[valid_main].copy()

    loaded_runs = sorted(
        pooled["run"].dropna().astype(str).unique().tolist(),
        key=natural_key,
    )

    if len(loaded_runs) != len(run_names):
        raise RuntimeError(
            "Replicate-count mismatch after loading predictions.\n"
            f"Run directories: {run_names}\n"
            f"Loaded run labels: {loaded_runs}"
        )

    print(
        f"[INFO] detected {len(loaded_runs)} AL replicates: "
        + ", ".join(loaded_runs)
    )

    return pooled, loaded_runs


# =============================================================================
# Collapse to one row per unique structure
# =============================================================================

def category_for_structure(
    group: pd.DataFrame,
) -> str:
    if not bool(group["is_baseline_zbg"].iloc[0]):
        return "non-ZBG"

    correct = (
        group["zbg_correct_gate_outcome"]
        .astype(bool)
    )

    incorrect = (
        group["zbg_incorrect_gate_outcome"]
        .astype(bool)
    )

    if bool(correct.all()):
        return "baseline-ZBG correct"

    if bool(incorrect.all()):
        return "baseline-ZBG incorrect"

    return "baseline-ZBG mixed"


def aggregate_structures(
    pooled: pd.DataFrame,
    expected_replicates: int,
) -> pd.DataFrame:
    rows: List[dict] = []

    for structure_id, group in pooled.groupby(
        "structure_id",
        sort=False,
    ):
        group = group.copy()

        hse_true_values = pd.to_numeric(
            group["hse_true"],
            errors="coerce",
        ).to_numpy(float)

        delta_true_values = pd.to_numeric(
            group["delta_true"],
            errors="coerce",
        ).to_numpy(float)

        pbe_values = pd.to_numeric(
            group["pbe_gap_effective"],
            errors="coerce",
        ).to_numpy(float)

        n_runs = int(group["run"].nunique())

        if n_runs != expected_replicates:
            print(
                "[WARN] structure does not contain all replicate "
                f"predictions: {structure_id}; "
                f"found {n_runs}, expected {expected_replicates}"
            )

        finite_hse_true = hse_true_values[
            np.isfinite(hse_true_values)
        ]

        finite_delta_true = delta_true_values[
            np.isfinite(delta_true_values)
        ]

        finite_pbe = pbe_values[
            np.isfinite(pbe_values)
        ]

        if (
            finite_hse_true.size > 1
            and np.ptp(finite_hse_true) > 1e-8
        ):
            print(
                "[WARN] inconsistent HSE truth across replicates "
                f"for {structure_id}"
            )

        if (
            finite_delta_true.size > 1
            and np.ptp(finite_delta_true) > 1e-8
        ):
            print(
                "[WARN] inconsistent Δ truth across replicates "
                f"for {structure_id}"
            )

        hse_predictions = pd.to_numeric(
            group["hse_pred"],
            errors="coerce",
        )

        delta_predictions = pd.to_numeric(
            group["delta_pred"],
            errors="coerce",
        )

        rows.append(
            {
                "structure_id": str(structure_id),
                "n_runs": n_runs,

                "hse_true": float(
                    np.nanmean(hse_true_values)
                ),
                "hse_pred_mean": float(
                    hse_predictions.mean()
                ),
                "hse_pred_sd": float(
                    hse_predictions.std(ddof=1)
                ),

                "delta_true": float(
                    np.nanmean(delta_true_values)
                ),
                "delta_pred_mean": float(
                    delta_predictions.mean()
                ),
                "delta_pred_sd": float(
                    delta_predictions.std(ddof=1)
                ),

                "pbe_gap_effective": float(
                    np.nanmean(pbe_values)
                ),

                "is_baseline_zbg": bool(
                    group["is_baseline_zbg"].iloc[0]
                ),

                "used_in_delta_panel": bool(
                    group["used_in_delta_panel"].all()
                ),

                "gate_correct_fraction": float(
                    group["zbg_correct_gate_outcome"]
                    .astype(float)
                    .mean()
                ),

                "gate_incorrect_fraction": float(
                    group["zbg_incorrect_gate_outcome"]
                    .astype(float)
                    .mean()
                ),

                "category": category_for_structure(group),
            }
        )

    summary = pd.DataFrame(rows)

    summary["hse_pred_sd"] = (
        pd.to_numeric(
            summary["hse_pred_sd"],
            errors="coerce",
        )
        .fillna(0.0)
    )

    summary["delta_pred_sd"] = (
        pd.to_numeric(
            summary["delta_pred_sd"],
            errors="coerce",
        )
        .fillna(0.0)
    )

    return summary


# =============================================================================
# Plot the per-structure mean ± SD figure
# =============================================================================

def make_mean_sd_plot(
    pooled: pd.DataFrame,
    structure_summary: pd.DataFrame,
    run_names: List[str],
    args: argparse.Namespace,
    output_base: Path,
) -> pd.DataFrame:
    style_matplotlib(args.font)

    n_replicates = len(run_names)

    if n_replicates == 0:
        raise RuntimeError(
            "No AL replicate labels were detected."
        )

    all_mask = (
        np.isfinite(structure_summary["hse_true"])
        & np.isfinite(structure_summary["hse_pred_mean"])
    )

    baseline_zbg_mask = (
        structure_summary["is_baseline_zbg"]
        .to_numpy(bool)
        & all_mask
    )

    non_zbg_mask = (
        ~structure_summary["is_baseline_zbg"]
        .to_numpy(bool)
        & all_mask
    )

    delta_mask = (
        structure_summary["used_in_delta_panel"]
        .to_numpy(bool)
        & np.isfinite(structure_summary["delta_true"])
        & np.isfinite(structure_summary["delta_pred_mean"])
    )

    hse_mae, hse_rmse, hse_r2, hse_n = regression_metrics(
        structure_summary.loc[
            all_mask,
            "hse_true",
        ].to_numpy(float),

        structure_summary.loc[
            all_mask,
            "hse_pred_mean",
        ].to_numpy(float),
    )

    zbg_mae, zbg_rmse, zbg_r2, zbg_n = regression_metrics(
        structure_summary.loc[
            baseline_zbg_mask,
            "hse_true",
        ].to_numpy(float),

        structure_summary.loc[
            baseline_zbg_mask,
            "hse_pred_mean",
        ].to_numpy(float),
    )

    non_zbg_mae, non_zbg_rmse, non_zbg_r2, non_zbg_n = (
        regression_metrics(
            structure_summary.loc[
                non_zbg_mask,
                "hse_true",
            ].to_numpy(float),

            structure_summary.loc[
                non_zbg_mask,
                "hse_pred_mean",
            ].to_numpy(float),
        )
    )

    delta_mae, delta_rmse, delta_r2, delta_n = (
        regression_metrics(
            structure_summary.loc[
                delta_mask,
                "delta_true",
            ].to_numpy(float),

            structure_summary.loc[
                delta_mask,
                "delta_pred_mean",
            ].to_numpy(float),
        )
    )

    figure, (axis_a, axis_b) = plt.subplots(
        1,
        2,
        figsize=(
            args.figwidth,
            args.figheight,
        ),
        dpi=300,
    )

    # -------------------------------------------------------------------------
    # Panel A: end-to-end HSE06 parity
    # -------------------------------------------------------------------------
    limits_a = parity_limits(
        structure_summary.loc[
            all_mask,
            "hse_true",
        ].to_numpy(float),

        structure_summary.loc[
            all_mask,
            "hse_pred_mean",
        ].to_numpy(float),

        structure_summary.loc[
            all_mask,
            "hse_pred_sd",
        ].to_numpy(float),
    )

    add_parity_guides(
        axis_a,
        limits_a,
        band=args.hse_band,
    )

    category_styles = {
        "non-ZBG": {
            "marker": "o",
            "color": PALETTE["navy"],
            "edgecolor": "none",
            "label": "non-ZBG",
        },
        "baseline-ZBG correct": {
            "marker": "^",
            "color": PALETTE["green"],
            "edgecolor": "none",
            "label": "baseline-ZBG correct gate outcome",
        },
        "baseline-ZBG incorrect": {
            "marker": "^",
            "color": PALETTE["red"],
            "edgecolor": PALETTE["ink"],
            "label": "baseline-ZBG incorrect gate outcome",
        },
        "baseline-ZBG mixed": {
            "marker": "D",
            "color": PALETTE["gold"],
            "edgecolor": PALETTE["ink"],
            "label": "baseline-ZBG mixed gate outcome",
        },
    }

    for category, plot_style in category_styles.items():
        subset = structure_summary.loc[
            (structure_summary["category"] == category)
            & all_mask
        ].copy()

        if subset.empty:
            continue

        marker_size = (
            5.5
            if plot_style["marker"] == "o"
            else 7.0
        )

        axis_a.errorbar(
            subset["hse_true"],
            subset["hse_pred_mean"],
            yerr=subset["hse_pred_sd"],
            fmt=plot_style["marker"],
            markersize=marker_size,
            color=plot_style["color"],
            markerfacecolor=plot_style["color"],
            markeredgecolor=plot_style["edgecolor"],
            markeredgewidth=0.55,
            ecolor=plot_style["color"],
            elinewidth=0.85,
            capsize=2.2,
            capthick=0.85,
            alpha=0.90,
            linestyle="none",
            label=(
                f'{plot_style["label"]} '
                f'(n={len(subset)})'
            ),
            zorder=3,
        )

    axis_a.set_xlim(limits_a)
    axis_a.set_ylim(limits_a)
    axis_a.set_aspect(
        "equal",
        adjustable="box",
    )

    axis_a.set_xlabel(
        "True HSE06 band gap (eV)"
    )

    axis_a.set_ylabel(
        "Mean predicted HSE06 band gap (eV)"
    )

    axis_a.set_title(
        "A. End-to-end HSE06 parity"
    )

    add_optional_grid(
        axis_a,
        args.grid,
    )

    annotation_a = (
        "Metrics from per-structure replicate means "
        f"(n={hse_n})\n"
        f"overall: MAE={hse_mae:.3f}, "
        f"RMSE={hse_rmse:.3f}, "
        f"R²={hse_r2:.3f}\n"
        f"baseline-ZBG: n={zbg_n}, "
        f"MAE={zbg_mae:.3f}, "
        f"RMSE={zbg_rmse:.3f}\n"
        f"non-ZBG: n={non_zbg_n}, "
        f"MAE={non_zbg_mae:.3f}, "
        f"RMSE={non_zbg_rmse:.3f}"
    )

    axis_a.text(
        0.03,
        0.97,
        annotation_a,
        transform=axis_a.transAxes,
        ha="left",
        va="top",
        fontsize=7.4,
        linespacing=1.05,
        bbox={
            "facecolor": "white",
            "edgecolor": PALETTE["light"],
            "alpha": 0.84,
            "pad": 2.0,
        },
        zorder=6,
    )

    axis_a.legend(
        loc="lower right",
        frameon=True,
        fontsize=7.3,
        borderpad=0.35,
        handletextpad=0.4,
        labelspacing=0.3,
    )

    # -------------------------------------------------------------------------
    # Panel B: non-ZBG Δ parity
    # -------------------------------------------------------------------------
    limits_b = (
        args.delta_min,
        args.delta_max,
    )

    add_parity_guides(
        axis_b,
        limits_b,
        band=args.delta_band,
    )

    delta_subset = structure_summary.loc[
        delta_mask
    ].copy()

    axis_b.errorbar(
        delta_subset["delta_true"],
        delta_subset["delta_pred_mean"],
        yerr=delta_subset["delta_pred_sd"],
        fmt="o",
        markersize=5.5,
        color=PALETTE["teal"],
        markerfacecolor=PALETTE["teal"],
        markeredgecolor="none",
        ecolor=PALETTE["teal"],
        elinewidth=0.85,
        capsize=2.2,
        capthick=0.85,
        alpha=0.90,
        linestyle="none",
        label=(
            f"non-ZBG structures "
            f"(n={delta_n})"
        ),
        zorder=3,
    )

    axis_b.set_xlim(limits_b)
    axis_b.set_ylim(limits_b)
    axis_b.set_aspect(
        "equal",
        adjustable="box",
    )

    axis_b.set_xlabel(
        "True Δ = HSE06 − PBE gap (eV)"
    )

    axis_b.set_ylabel(
        "Mean predicted Δ (eV)"
    )

    axis_b.set_title(
        "B. Δ parity for non-ZBG structures"
    )

    add_optional_grid(
        axis_b,
        args.grid,
    )

    annotation_b = (
        "Metrics from per-structure replicate means "
        f"(n={delta_n})\n"
        f"MAE={delta_mae:.3f}, "
        f"RMSE={delta_rmse:.3f}, "
        f"R²={delta_r2:.3f}"
    )

    axis_b.text(
        0.03,
        0.97,
        annotation_b,
        transform=axis_b.transAxes,
        ha="left",
        va="top",
        fontsize=7.4,
        linespacing=1.05,
        bbox={
            "facecolor": "white",
            "edgecolor": PALETTE["light"],
            "alpha": 0.84,
            "pad": 2.0,
        },
        zorder=6,
    )

    axis_b.legend(
        loc="lower right",
        frameon=True,
        fontsize=7.4,
        borderpad=0.35,
        handletextpad=0.4,
    )

    # -------------------------------------------------------------------------
    # Overall title and footer
    # -------------------------------------------------------------------------
    if args.mode == "test":
        set_name = "held-out test"
    else:
        set_name = "fixed-validation"

    default_title = (
        f"Per-structure {set_name} parity across "
        f"{n_replicates} AL replicates"
    )

    figure.suptitle(
        args.title.strip() or default_title,
        y=1.005,
        fontsize=11.5,
        fontweight="bold",
    )

    figure.text(
        0.5,
        0.008,
        (
            "Each marker represents one unique structure and is positioned "
            f"at the mean prediction across {n_replicates} AL models; "
            "vertical error bars denote ±1 SD across replicate predictions."
        ),
        ha="center",
        va="bottom",
        fontsize=7.8,
    )

    figure.tight_layout(
        rect=(0.0, 0.045, 1.0, 0.95),
        w_pad=1.25,
    )

    save_figure(
        figure=figure,
        output_base=output_base,
        dpi=args.dpi,
    )

    metrics_table = pd.DataFrame(
        [
            {
                "representation": "mean_prediction_per_structure",
                "metric_basis": (
                    "metrics calculated after averaging replicate "
                    "predictions for each unique structure"
                ),
                "mode": args.mode,
                "cycle": args.cycle,
                "n_replicates": n_replicates,
                "subset": "HSE_overall",
                "n_unique_structures": hse_n,
                "MAE": hse_mae,
                "RMSE": hse_rmse,
                "R2": hse_r2,
            },
            {
                "representation": "mean_prediction_per_structure",
                "metric_basis": (
                    "metrics calculated after averaging replicate "
                    "predictions for each unique structure"
                ),
                "mode": args.mode,
                "cycle": args.cycle,
                "n_replicates": n_replicates,
                "subset": "HSE_baseline_ZBG",
                "n_unique_structures": zbg_n,
                "MAE": zbg_mae,
                "RMSE": zbg_rmse,
                "R2": zbg_r2,
            },
            {
                "representation": "mean_prediction_per_structure",
                "metric_basis": (
                    "metrics calculated after averaging replicate "
                    "predictions for each unique structure"
                ),
                "mode": args.mode,
                "cycle": args.cycle,
                "n_replicates": n_replicates,
                "subset": "HSE_non_ZBG",
                "n_unique_structures": non_zbg_n,
                "MAE": non_zbg_mae,
                "RMSE": non_zbg_rmse,
                "R2": non_zbg_r2,
            },
            {
                "representation": "mean_prediction_per_structure",
                "metric_basis": (
                    "metrics calculated after averaging replicate "
                    "predictions for each unique structure"
                ),
                "mode": args.mode,
                "cycle": args.cycle,
                "n_replicates": n_replicates,
                "subset": "Delta_non_ZBG",
                "n_unique_structures": delta_n,
                "MAE": delta_mae,
                "RMSE": delta_rmse,
                "R2": delta_r2,
            },
        ]
    )

    return metrics_table


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()

    pooled, run_names = load_pooled_predictions(
        args
    )

    expected_replicates = len(run_names)

    structure_summary = aggregate_structures(
        pooled=pooled,
        expected_replicates=expected_replicates,
    )

    output_directory = Path(
        args.out_dir
    ).resolve()

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_base = (
        output_directory
        / args.out_prefix
    )

    metrics_table = make_mean_sd_plot(
        pooled=pooled,
        structure_summary=structure_summary,
        run_names=run_names,
        args=args,
        output_base=output_base,
    )

    pooled_path = output_base.with_name(
        output_base.name
        + "_pooled_points.csv"
    )

    structure_path = output_base.with_name(
        output_base.name
        + "_structure_summary.csv"
    )

    metrics_path = output_base.with_name(
        output_base.name
        + "_metrics.csv"
    )

    pooled.to_csv(
        pooled_path,
        index=False,
    )

    structure_summary.to_csv(
        structure_path,
        index=False,
    )

    metrics_table.to_csv(
        metrics_path,
        index=False,
    )

    print("\n[OK] wrote figure files:")
    print(" ", output_base.with_suffix(".png"))
    print(" ", output_base.with_suffix(".pdf"))
    print(" ", output_base.with_suffix(".svg"))

    print("\n[OK] wrote data tables:")
    print(" ", pooled_path)
    print(" ", structure_path)
    print(" ", metrics_path)

    print("\n[INFO] replicate count:")
    print(" ", expected_replicates)

    print("\n[INFO] replicate names:")
    print(" ", ", ".join(run_names))

    print("\n[INFO] unique structures:")
    print(" ", structure_summary["structure_id"].nunique())

    print("\nPer-structure mean-prediction metrics:")
    print(metrics_table.to_string(index=False))


if __name__ == "__main__":
    main()

