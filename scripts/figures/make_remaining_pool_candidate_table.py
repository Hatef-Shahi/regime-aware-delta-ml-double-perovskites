#!/usr/bin/env python3
"""
Create a publication-ready candidate table from remaining unlabelled-pool predictions.

Purpose
-------
After the fixed-budget AL run, use the trained DOS-on Δ-ML model predictions
to rank remaining unlabelled candidates for follow-up HSE validation.

This script creates ONE table only:
    Reliable predicted candidates from the remaining unlabelled pool.

It filters candidates by:
    - predicted HSE gap range
    - uncertainty threshold
    - remaining/unlabelled status, when available
    - optional baseline-ZBG / non-ZBG classification

Outputs
-------
1. Full filtered candidate CSV
2. Top-N publication table CSV
3. Top-N publication table Markdown
4. Optional LaTeX table

Typical use
-----------
python3 plots/make_remaining_pool_candidate_table.py \
  --root . \
  --run-dir paper-runs/delta_kg2_m6_q035_theta016_dosON/al3 \
  --cycle 24 \
  --min-hse-gap 1.0 \
  --max-hse-gap 2.5 \
  --sigma-quantile 0.75 \
  --top-n 20 \
  --out-dir artifacts/pub_final_tables/remaining_pool_candidates_dosON \
  --out-prefix table_remaining_pool_candidates_dosON

Notes
-----
- If you want one replicate as the final model, use a representative/final run.
- If you want consensus across al1-al5, use --runs-dir instead of --run-dir.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Make reliable predicted-candidate table from remaining unlabelled pool."
    )

    p.add_argument("--root", default=".", help="Experiment root directory")

    # Either one run or multiple runs
    p.add_argument("--run-dir", default="", help="Single run directory, e.g. paper-runs/.../al3")
    p.add_argument("--runs-dir", default="", help="Directory containing replicate runs, e.g. paper-runs/.../dosON")
    p.add_argument("--run-glob", default="al*", help="Replicate glob when --runs-dir is used")

    p.add_argument("--cycle", type=int, default=24)

    # Candidate filters
    p.add_argument("--min-hse-gap", type=float, default=1.0)
    p.add_argument("--max-hse-gap", type=float, default=2.5)
    p.add_argument("--pbe-zero-thr", type=float, default=0.05)

    # uncertainty filters
    p.add_argument(
        "--sigma-quantile",
        type=float,
        default=0.75,
        help="Keep candidates with sigma <= this quantile within the prediction pool. Use 1.0 to keep all.",
    )
    p.add_argument(
        "--max-sigma",
        type=float,
        default=np.nan,
        help="Optional absolute sigma cutoff. If supplied, overrides --sigma-quantile.",
    )

    # table size
    p.add_argument("--top-n", type=int, default=20)

    # Optional class windows
    p.add_argument("--narrow-max", type=float, default=1.0)
    p.add_argument("--visible-min", type=float, default=1.0)
    p.add_argument("--visible-max", type=float, default=2.5)

    # Optional column overrides
    p.add_argument("--col-id", default="")
    p.add_argument("--col-formula", default="")
    p.add_argument("--col-pbe-gap", default="")
    p.add_argument("--col-hse-pred", default="")
    p.add_argument("--col-delta-pred", default="")
    p.add_argument("--col-sigma", default="")
    p.add_argument("--col-score", default="")
    p.add_argument("--col-is-labeled", default="")
    p.add_argument("--col-is-selected", default="")
    p.add_argument("--col-bank-file", default="")

    # Output
    p.add_argument("--out-dir", required=True)
    p.add_argument("--out-prefix", default="table_remaining_pool_candidates")
    p.add_argument("--latex", action="store_true")

    return p.parse_args()


# -----------------------------
# Column utilities
# -----------------------------

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


def choose_col(
    df: pd.DataFrame,
    override: str,
    candidates: Iterable[str],
    required: bool = True,
    numeric: bool = False,
) -> Optional[str]:
    if override and override in df.columns:
        return override

    best = None
    best_n = -1

    for cand in candidates:
        c = find_column(df, cand)
        if c is None or c not in df.columns:
            continue

        if numeric:
            vals = pd.to_numeric(df[c], errors="coerce")
            n = int(vals.notna().sum())
        else:
            n = int(df[c].notna().sum())

        if n > best_n:
            best = c
            best_n = n

    if best is None and required:
        raise KeyError(
            f"Could not find required column. Tried: {list(candidates)}\n"
            f"Available columns:\n{list(df.columns)}"
        )

    return best


def to_num(df: pd.DataFrame, col: Optional[str], default: float = np.nan) -> np.ndarray:
    if col is None or col not in df.columns:
        return np.full(len(df), default, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").to_numpy(float)


# -----------------------------
# Prediction-file loading
# -----------------------------

def find_prediction_file(run_dir: Path, cycle: int) -> Path:
    candidates = [
        run_dir / "predictions" / f"pred_cycle{cycle}.csv",
        run_dir / "predictions" / f"all_predictions_cycle{cycle}.csv",
        run_dir / "predictions" / f"pool_predictions_cycle{cycle}.csv",
        run_dir / "predictions" / f"predictions_cycle{cycle}.csv",
        run_dir / f"pred_cycle{cycle}.csv",
        run_dir / f"all_predictions_cycle{cycle}.csv",
    ]

    for p in candidates:
        if p.exists():
            return p

    # Match the cycle number exactly: "*cycle2*" would also match cycle20 to
    # cycle29, and file-name order would then decide.
    exact_cycle = re.compile(rf"cycle{cycle}(?!\d)")
    hits = sorted(run_dir.rglob(f"*cycle{cycle}*.csv"))
    hits = [
        p for p in hits
        if exact_cycle.search(p.name)
        and "pred" in p.name.lower()
        and "val_pred" not in p.name.lower()
        and "test_pred" not in p.name.lower()
    ]

    if len(hits) > 1:
        raise RuntimeError(
            f"Several prediction CSVs for cycle {cycle} under {run_dir}: {hits}"
        )
    if hits:
        return hits[0]

    raise FileNotFoundError(
        f"Could not find prediction CSV for cycle {cycle} under {run_dir}"
    )


def load_prediction_table(run_dir: Path, cycle: int) -> pd.DataFrame:
    fp = find_prediction_file(run_dir, cycle)
    df = pd.read_csv(fp)
    df["source_run"] = run_dir.name
    try:
        df["source_file"] = str(fp.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        df["source_file"] = str(fp)
    print(f"[LOAD] {run_dir.name}: {fp}")
    return df


def load_all_predictions(args: argparse.Namespace) -> pd.DataFrame:
    root = Path(args.root).resolve()

    if args.run_dir:
        run_dir = (root / args.run_dir).resolve()
        return load_prediction_table(run_dir, args.cycle)

    if not args.runs_dir:
        raise ValueError("Provide either --run-dir or --runs-dir")

    runs_base = (root / args.runs_dir).resolve()
    run_dirs = sorted([p for p in runs_base.glob(args.run_glob) if p.is_dir()])

    if not run_dirs:
        raise FileNotFoundError(
            f"No runs found under {runs_base} matching {args.run_glob}"
        )

    frames = []
    for rd in run_dirs:
        frames.append(load_prediction_table(rd, args.cycle))

    return pd.concat(frames, ignore_index=True)


# -----------------------------
# Standardization
# -----------------------------

def standardize_predictions(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    col_id = choose_col(
        df,
        args.col_id,
        ["structure_id", "material_id", "mp_id", "id", "name"],
        required=True,
        numeric=False,
    )

    col_formula = choose_col(
        df,
        args.col_formula,
        ["formula", "pretty_formula", "composition", "reduced_formula"],
        required=False,
        numeric=False,
    )

    col_pbe = choose_col(
        df,
        args.col_pbe_gap,
        [
            "pbe_gap_effective",
            "pbe_gap_eff",
            "pbe_gap",
            "pbe_true",
            "pbe_gap_true",
            "baseline_gap",
            "pbe_baseline",
            "g_b",
        ],
        required=True,
        numeric=True,
    )

    col_hse_pred = choose_col(
        df,
        args.col_hse_pred,
        [
            "hse_pred",
            "hse_gap_pred",
            "pred_hse",
            "hse_pred_end2end",
            "end2end_hse_pred",
            "pred_gap",
            "hse_gap_model",
        ],
        required=False,
        numeric=True,
    )

    col_delta_pred = choose_col(
        df,
        args.col_delta_pred,
        [
            "delta_pred_stageB",
            "delta_pred",
            "pred_delta",
            "delta_hat",
            "predicted_delta",
        ],
        required=False,
        numeric=True,
    )

    col_sigma = choose_col(
        df,
        args.col_sigma,
        [
            "sigma_gpr",
            "sigma_delta",
            "sigma",
            "sigma_eff",
            "uncertainty",
            "std_pred",
            "std_delta",
        ],
        required=True,
        numeric=True,
    )

    col_score = choose_col(
        df,
        args.col_score,
        [
            "score_base",
            "acquisition_score",
            "score",
            "score_delta",
            "S_delta",
        ],
        required=False,
        numeric=True,
    )

    col_labeled = choose_col(
        df,
        args.col_is_labeled,
        ["is_labeled", "labeled", "already_labeled", "in_train", "seen", "is_seen"],
        required=False,
        numeric=True,
    )

    col_selected = choose_col(
        df,
        args.col_is_selected,
        ["is_selected", "selected", "selected_this_cycle", "in_selection"],
        required=False,
        numeric=True,
    )

    col_bank = choose_col(
        df,
        args.col_bank_file,
        ["bank_file", "structure_file", "poscar_file", "cif_file"],
        required=False,
        numeric=False,
    )

    out = pd.DataFrame()
    out["structure_id"] = df[col_id].astype(str)

    if col_formula is not None:
        out["formula"] = df[col_formula].astype(str)
    else:
        out["formula"] = out["structure_id"].str.replace(r"^.*:", "", regex=True)

    out["pbe_gap"] = to_num(df, col_pbe)
    out["sigma_delta"] = to_num(df, col_sigma)

    if col_delta_pred is not None:
        out["delta_pred"] = to_num(df, col_delta_pred)
    else:
        out["delta_pred"] = np.nan

    if col_hse_pred is not None:
        out["hse_gap_pred"] = to_num(df, col_hse_pred)
    elif col_delta_pred is not None:
        out["hse_gap_pred"] = out["pbe_gap"] + out["delta_pred"]
    else:
        raise KeyError(
            "Could not determine predicted HSE gap. Need either hse_pred or delta_pred."
        )

    if col_score is not None:
        out["acquisition_score"] = to_num(df, col_score)
    else:
        out["acquisition_score"] = np.nan

    if col_labeled is not None:
        out["is_labeled"] = to_num(df, col_labeled, default=0.0)
    else:
        out["is_labeled"] = 0.0

    if col_selected is not None:
        out["is_selected"] = to_num(df, col_selected, default=0.0)
    else:
        out["is_selected"] = 0.0

    if col_bank is not None:
        out["bank_file"] = df[col_bank].astype(str)
    else:
        out["bank_file"] = ""

    if "source_run" in df.columns:
        out["source_run"] = df["source_run"].astype(str)
    else:
        out["source_run"] = ""

    if "source_file" in df.columns:
        out["source_file"] = df["source_file"].astype(str)
    else:
        out["source_file"] = ""

    out["baseline_ZBG"] = out["pbe_gap"] <= args.pbe_zero_thr

    out["predicted_class"] = classify_gap(
        out["hse_gap_pred"].to_numpy(float),
        narrow_max=args.narrow_max,
        visible_min=args.visible_min,
        visible_max=args.visible_max,
    )

    return out


def classify_gap(
    hse_pred: np.ndarray,
    narrow_max: float,
    visible_min: float,
    visible_max: float,
) -> List[str]:
    classes = []

    for g in hse_pred:
        if not np.isfinite(g):
            classes.append("invalid")
        elif g <= 0.05:
            classes.append("metal/ZBG-like")
        elif g < narrow_max:
            classes.append("narrow-gap")
        elif visible_min <= g <= visible_max:
            classes.append("visible-range")
        else:
            classes.append("wide-gap")

    return classes


# -----------------------------
# Filtering and ranking
# -----------------------------

def remove_labeled_or_selected(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # If these columns are unavailable, assume all candidates are still eligible.
    if "is_labeled" in out.columns:
        labeled = pd.to_numeric(out["is_labeled"], errors="coerce").fillna(0.0).to_numpy(float)
    else:
        labeled = np.zeros(len(out), dtype=float)

    if "is_selected" in out.columns:
        selected = pd.to_numeric(out["is_selected"], errors="coerce").fillna(0.0).to_numpy(float)
    else:
        selected = np.zeros(len(out), dtype=float)

    keep = (labeled < 0.5) & (selected < 0.5)
    return out.loc[keep].copy()


def apply_filters(df: pd.DataFrame, args: argparse.Namespace) -> Tuple[pd.DataFrame, float]:
    out = remove_labeled_or_selected(df)

    # finite required values
    out = out.loc[
        np.isfinite(out["hse_gap_pred"])
        & np.isfinite(out["sigma_delta"])
        & np.isfinite(out["pbe_gap"])
    ].copy()

    # predicted useful HSE gap window
    out = out.loc[
        (out["hse_gap_pred"] >= args.min_hse_gap)
        & (out["hse_gap_pred"] <= args.max_hse_gap)
    ].copy()

    if out.empty:
        return out, np.nan

    if np.isfinite(args.max_sigma):
        sigma_cutoff = float(args.max_sigma)
    else:
        q = float(args.sigma_quantile)
        q = min(max(q, 0.0), 1.0)
        sigma_cutoff = float(out["sigma_delta"].quantile(q))

    out = out.loc[out["sigma_delta"] <= sigma_cutoff].copy()

    return out, sigma_cutoff


def aggregate_replicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    If predictions from multiple AL replicates are provided, aggregate by structure_id.

    This gives consensus-style values:
        hse_gap_pred_mean
        hse_gap_pred_sd
        sigma_delta_mean
        sigma_delta_min
        pbe_gap_mean

    The publication table then uses the mean values.
    """
    if "source_run" not in df.columns or df["source_run"].nunique() <= 1:
        out = df.copy()
        out["n_replicates"] = 1
        out["hse_gap_pred_sd"] = np.nan
        out["sigma_delta_min"] = out["sigma_delta"]
        return out

    group_cols = ["structure_id"]

    agg = (
        df.groupby(group_cols, as_index=False)
        .agg(
            formula=("formula", "first"),
            pbe_gap=("pbe_gap", "mean"),
            hse_gap_pred=("hse_gap_pred", "mean"),
            hse_gap_pred_sd=("hse_gap_pred", "std"),
            delta_pred=("delta_pred", "mean"),
            sigma_delta=("sigma_delta", "mean"),
            sigma_delta_min=("sigma_delta", "min"),
            acquisition_score=("acquisition_score", "mean"),
            baseline_ZBG=("baseline_ZBG", "max"),
            predicted_class=("predicted_class", lambda x: x.mode().iat[0] if len(x.mode()) else x.iloc[0]),
            bank_file=("bank_file", "first"),
            is_labeled=("is_labeled", "max"),
            is_selected=("is_selected", "max"),
            n_replicates=("source_run", "nunique"),
        )
    )

    return agg


def rank_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rank reliable screening candidates.

    Primary criteria:
        - visible-range predicted class first
        - lower uncertainty
        - lower replicate disagreement, if available
        - predicted HSE gap closer to middle of useful range (~1.6 eV)

    This ranking is meant for triage, not for claiming confirmed discovery.
    """
    out = df.copy()

    target_gap = 1.6
    out["gap_distance_to_target"] = np.abs(out["hse_gap_pred"] - target_gap)

    class_priority = {
        "visible-range": 0,
        "narrow-gap": 1,
        "wide-gap": 2,
        "metal/ZBG-like": 3,
        "invalid": 9,
    }
    out["class_priority"] = out["predicted_class"].map(class_priority).fillna(9).astype(int)

    if "hse_gap_pred_sd" not in out.columns:
        out["hse_gap_pred_sd"] = np.nan

    # NaN replicate SD should not punish single-run mode.
    sd_for_sort = out["hse_gap_pred_sd"].fillna(0.0)

    out = out.assign(_sd_sort=sd_for_sort)

    out = out.sort_values(
        by=[
            "class_priority",
            "sigma_delta",
            "_sd_sort",
            "gap_distance_to_target",
            "hse_gap_pred",
        ],
        ascending=[True, True, True, True, True],
    ).copy()

    out["rank"] = np.arange(1, len(out) + 1)
    out = out.drop(columns=["_sd_sort"])

    return out


def make_reason(row: pd.Series) -> str:
    reasons = []

    if row.get("predicted_class", "") == "visible-range":
        reasons.append("useful predicted gap")
    elif row.get("predicted_class", "") == "narrow-gap":
        reasons.append("low-gap candidate")

    reasons.append("low/moderate uncertainty")

    if bool(row.get("baseline_ZBG", False)):
        reasons.append("baseline-ZBG candidate")

    if row.get("n_replicates", 1) > 1:
        sd = row.get("hse_gap_pred_sd", np.nan)
        if np.isfinite(sd):
            reasons.append("replicate-consensus prediction")

    return "; ".join(reasons)


def format_publication_table(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    top = df.head(top_n).copy()

    top["reason_for_priority"] = top.apply(make_reason, axis=1)

    cols = [
        "rank",
        "structure_id",
        "formula",
        "pbe_gap",
        "hse_gap_pred",
        "sigma_delta",
        "predicted_class",
        "baseline_ZBG",
        "reason_for_priority",
    ]

    if "hse_gap_pred_sd" in top.columns:
        cols.insert(5, "hse_gap_pred_sd")

    if "n_replicates" in top.columns:
        cols.append("n_replicates")

    if "bank_file" in top.columns:
        cols.append("bank_file")

    pub = top[cols].copy()

    rename = {
        "rank": "rank",
        "structure_id": "structure_id",
        "formula": "formula",
        "pbe_gap": "baseline_gap_eV",
        "hse_gap_pred": "predicted_HSE_gap_eV",
        "hse_gap_pred_sd": "predicted_HSE_gap_SD_eV",
        "sigma_delta": "uncertainty_sigma_eV",
        "predicted_class": "predicted_class",
        "baseline_ZBG": "baseline_ZBG",
        "reason_for_priority": "reason_for_priority",
        "n_replicates": "n_replicates",
        "bank_file": "bank_file",
    }

    pub = pub.rename(columns=rename)

    # Round numeric values
    for c in [
        "baseline_gap_eV",
        "predicted_HSE_gap_eV",
        "predicted_HSE_gap_SD_eV",
        "uncertainty_sigma_eV",
    ]:
        if c in pub.columns:
            pub[c] = pd.to_numeric(pub[c], errors="coerce").round(3)

    return pub


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    args = parse_args()

    raw = load_all_predictions(args)
    std = standardize_predictions(raw, args)

    # Aggregate first if multiple replicate predictions are supplied.
    agg = aggregate_replicates(std)

    filtered, sigma_cutoff = apply_filters(agg, args)
    ranked = rank_candidates(filtered)
    pub = format_publication_table(ranked, args.top_n)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / args.out_prefix

    # Full standardized predictions
    std.to_csv(base.with_name(base.name + "_all_standardized_predictions.csv"), index=False)

    # Aggregated predictions if applicable
    agg.to_csv(base.with_name(base.name + "_aggregated_predictions.csv"), index=False)

    # Filtered and ranked candidates
    ranked.to_csv(base.with_name(base.name + "_filtered_ranked_full.csv"), index=False)

    # Publication table
    pub_csv = base.with_name(base.name + "_top_candidates.csv")
    pub_md = base.with_name(base.name + "_top_candidates.md")

    pub.to_csv(pub_csv, index=False)

    with open(pub_md, "w") as f:
        f.write(pub.to_markdown(index=False))

    if args.latex:
        pub_tex = base.with_name(base.name + "_top_candidates.tex")
        with open(pub_tex, "w") as f:
            f.write(pub.to_latex(index=False, escape=False))
        print(f"[OK] wrote: {pub_tex}")

    summary = pd.DataFrame([{
        "cycle": args.cycle,
        "input_rows": len(raw),
        "standardized_rows": len(std),
        "aggregated_rows": len(agg),
        "filtered_rows": len(filtered),
        "top_n_requested": args.top_n,
        "top_n_written": len(pub),
        "min_hse_gap": args.min_hse_gap,
        "max_hse_gap": args.max_hse_gap,
        "sigma_quantile": args.sigma_quantile,
        "sigma_cutoff_used": sigma_cutoff,
    }])

    summary.to_csv(base.with_name(base.name + "_summary.csv"), index=False)

    print("\n[OK] wrote:")
    print(" ", base.with_name(base.name + "_all_standardized_predictions.csv"))
    print(" ", base.with_name(base.name + "_aggregated_predictions.csv"))
    print(" ", base.with_name(base.name + "_filtered_ranked_full.csv"))
    print(" ", pub_csv)
    print(" ", pub_md)
    print(" ", base.with_name(base.name + "_summary.csv"))

    print("\nSummary:")
    print(summary.to_string(index=False))

    print("\nTop candidates:")
    print(pub.to_string(index=False))


if __name__ == "__main__":
    main()
