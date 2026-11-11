#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path.cwd()
GATE_ROOT = ROOT / "paper-runs" / "delta_kg2_m6_q035_theta016_dosON"
OUTDIR = ROOT / "artifacts" / "diagnostics" / "gate_threshold_false_rescues"
OUTDIR.mkdir(parents=True, exist_ok=True)

CYCLE = 24
CURRENT_THETA = 0.16
PBE_ZERO_THR = 0.05
HSE_ZERO_THR = 0.05
EPS = 1e-12


def read_one(path):
    df = pd.read_csv(path)

    required = [
        "p_gate",
        "hse_true",
        "pbe_gap_effective",
    ]
    for c in required:
        if c not in df.columns:
            raise KeyError(f"{path} missing required column: {c}")

    id_col = "structure_id" if "structure_id" in df.columns else df.columns[0]

    out = pd.DataFrame({
        "run": path.parts[-3],
        "structure_id": df[id_col].astype(str).str.strip(),
        "p_gate": pd.to_numeric(df["p_gate"], errors="coerce"),
        "hse_true": pd.to_numeric(df["hse_true"], errors="coerce"),
        "pbe_gap_effective": pd.to_numeric(df["pbe_gap_effective"], errors="coerce"),
        "hse_pred": pd.to_numeric(df["hse_pred"], errors="coerce") if "hse_pred" in df.columns else np.nan,
    })

    if "formula" in df.columns:
        out["formula"] = df["formula"].astype(str)
    else:
        out["formula"] = ""

    out["is_pbe_gap_zero"] = out["pbe_gap_effective"].le(PBE_ZERO_THR)
    out["true_finite_hse"] = out["hse_true"].gt(HSE_ZERO_THR)
    out["true_metal_like"] = out["hse_true"].le(HSE_ZERO_THR)

    out["rescued_at_current_theta"] = out["p_gate"].ge(CURRENT_THETA)
    out["metal_at_current_theta"] = out["p_gate"].lt(CURRENT_THETA)

    out["current_decision_correct"] = (
        (out["true_finite_hse"] & out["rescued_at_current_theta"])
        | (out["true_metal_like"] & out["metal_at_current_theta"])
    )

    return out


def metrics_at_threshold(zbg, theta):
    pred_rescue = zbg["p_gate"].ge(theta)
    true_rescue = zbg["true_finite_hse"]

    tp = int((pred_rescue & true_rescue).sum())
    fp = int((pred_rescue & (~true_rescue)).sum())
    tn = int(((~pred_rescue) & (~true_rescue)).sum())
    fn = int(((~pred_rescue) & true_rescue).sum())

    precision = tp / (tp + fp) if (tp + fp) else np.nan
    recall = tp / (tp + fn) if (tp + fn) else np.nan
    f1 = 2 * precision * recall / (precision + recall) if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) else np.nan

    return {
        "theta": theta,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():
    files = sorted(GATE_ROOT.glob(f"al*/test_eval/test_pred_cycle{CYCLE}.csv"))
    if len(files) != 5:
        raise FileNotFoundError(
            f"expected 5 test_pred_cycle{CYCLE}.csv files under {GATE_ROOT}, "
            f"found {len(files)}: {files}"
        )

    all_df = pd.concat([read_one(p) for p in files], ignore_index=True)
    zbg = all_df[all_df["is_pbe_gap_zero"]].copy()

    if zbg.empty:
        raise RuntimeError("No PBE-gap-zero test rows found.")

    zbg.to_csv(OUTDIR / "zbg_test_pgate_by_run.csv", index=False)

    # Per-structure summary across the five replicate gates.
    per_structure = (
        zbg
        .groupby("structure_id", as_index=False)
        .agg(
            formula=("formula", "first"),
            hse_true=("hse_true", "first"),
            pbe_gap_effective=("pbe_gap_effective", "first"),
            p_gate_mean=("p_gate", "mean"),
            p_gate_sd=("p_gate", "std"),
            p_gate_min=("p_gate", "min"),
            p_gate_max=("p_gate", "max"),
            n_runs=("run", "nunique"),
            n_rescued_current=("rescued_at_current_theta", "sum"),
            n_metal_current=("metal_at_current_theta", "sum"),
            n_correct_current=("current_decision_correct", "sum"),
        )
    )

    per_structure["true_class"] = np.where(
        per_structure["hse_true"].gt(HSE_ZERO_THR),
        "rescued_finite_HSE",
        "true_metal_like",
    )

    per_structure = per_structure.sort_values(
        ["true_class", "p_gate_mean"],
        ascending=[True, True],
    )

    per_structure.to_csv(OUTDIR / "zbg_test_pgate_by_structure.csv", index=False)

    true_metals = zbg[zbg["true_metal_like"]].copy()
    true_rescues = zbg[zbg["true_finite_hse"]].copy()

    max_metal_p = float(true_metals["p_gate"].max()) if len(true_metals) else np.nan
    min_rescue_p = float(true_rescues["p_gate"].min()) if len(true_rescues) else np.nan

    theta_needed_to_fix_metals = max_metal_p + EPS
    theta_max_to_keep_all_rescues = min_rescue_p

    perfect_interval_exists = (
        np.isfinite(theta_needed_to_fix_metals)
        and np.isfinite(theta_max_to_keep_all_rescues)
        and theta_needed_to_fix_metals <= theta_max_to_keep_all_rescues
    )

    # Threshold sweep.
    thresholds = sorted(set(
        [0.0, 0.05, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
        + [float(x) for x in zbg["p_gate"].dropna().values]
        + [theta_needed_to_fix_metals, theta_max_to_keep_all_rescues]
    ))

    sweep = pd.DataFrame([metrics_at_threshold(zbg, t) for t in thresholds])
    sweep.to_csv(OUTDIR / "zbg_test_threshold_sweep.csv", index=False)

    summary = pd.DataFrame([{
        "current_theta": CURRENT_THETA,
        "n_zbg_predictions": int(len(zbg)),
        "n_zbg_structures": int(zbg["structure_id"].nunique()),
        "n_true_metal_predictions": int(len(true_metals)),
        "n_true_rescue_predictions": int(len(true_rescues)),
        "max_p_gate_among_true_metals": max_metal_p,
        "min_p_gate_among_true_rescues": min_rescue_p,
        "theta_needed_to_route_all_true_metals_as_metal": theta_needed_to_fix_metals,
        "largest_theta_that_preserves_all_true_rescues": theta_max_to_keep_all_rescues,
        "perfect_test_threshold_interval_exists": bool(perfect_interval_exists),
        "safe_interval_lower_open": theta_needed_to_fix_metals,
        "safe_interval_upper_closed": theta_max_to_keep_all_rescues,
    }])

    summary.to_csv(OUTDIR / "zbg_test_threshold_requirement_summary.csv", index=False)

    print("\nPBE-gap-zero test p_gate by structure")
    cols = [
        "structure_id", "formula", "true_class", "hse_true",
        "p_gate_mean", "p_gate_sd", "p_gate_min", "p_gate_max",
        "n_rescued_current", "n_metal_current", "n_correct_current",
    ]
    print(per_structure[cols].to_string(index=False))

    print("\nThreshold requirement summary")
    print(summary.to_string(index=False))

    print("\nThreshold sweep around current region")
    show = sweep[(sweep["theta"] >= 0.10) & (sweep["theta"] <= 0.50)].copy()
    print(show.to_string(index=False))

    print("\nWrote:")
    print(OUTDIR / "zbg_test_pgate_by_run.csv")
    print(OUTDIR / "zbg_test_pgate_by_structure.csv")
    print(OUTDIR / "zbg_test_threshold_sweep.csv")
    print(OUTDIR / "zbg_test_threshold_requirement_summary.csv")


if __name__ == "__main__":
    main()
