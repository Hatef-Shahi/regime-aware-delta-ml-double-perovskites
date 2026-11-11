#!/usr/bin/env python3

from pathlib import Path
import re
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, precision_score, recall_score, f1_score, confusion_matrix


REPO = Path(__file__).resolve().parents[2]

DEFAULT_NOGATE_ROOT = REPO / "OFFICIAL-RUNS" / "02_nogate_unified_delta_fullpool"
DEFAULT_GATE_ROOT = REPO / "OFFICIAL-RUNS" / "01_gate_aware_delta_ml_dos_on"
DEFAULT_OUTDIR = REPO / "audits" / "nogate_vs_gate_fullpool" / "test_eval_nogate_cycle24"

PBE_ZERO_THR = 0.05
HSE_ZERO_THR = 0.05


def norm_key(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def find_col(df, candidates, required=False):
    lookup = {norm_key(c): c for c in df.columns}
    for cand in candidates:
        k = norm_key(cand)
        if k in lookup:
            return lookup[k]
    if required:
        raise KeyError(f"Could not find one of {candidates}. Available columns: {list(df.columns)}")
    return None


def read_csv(path):
    return pd.read_csv(path, low_memory=False)


def safe_metrics(y_true, y_pred):
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    m = np.isfinite(yt) & np.isfinite(yp)

    if int(m.sum()) < 2:
        return {"n": int(m.sum()), "mae": np.nan, "rmse": np.nan, "r2": np.nan}

    yt = yt[m]
    yp = yp[m]

    return {
        "n": int(len(yt)),
        "mae": float(mean_absolute_error(yt, yp)),
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "r2": float(r2_score(yt, yp)),
    }


def rescue_metrics(y_true_finite, y_pred_finite):
    y_true = np.asarray(y_true_finite, dtype=int)
    y_pred = np.asarray(y_pred_finite, dtype=int)

    if len(y_true) == 0:
        return {
            "n": 0,
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "tp": 0,
        }

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    return {
        "n": int(len(y_true)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }


def find_truth_csv(gate_root):
    """
    Prefer an existing gate-aware test prediction file as the source of frozen
    test IDs, HSE truth, and PBE baseline gaps.
    """
    candidates = []

    for pattern in [
        "al1/test_eval/test_pred_cycle24.csv",
        "al1/metrics/test_pred_cycle24.csv",
        "al1/**/test_pred_cycle24.csv",
        "al1/**/*test*cycle24*.csv",
        "al1/**/*test*pred*.csv",
    ]:
        candidates.extend(gate_root.glob(pattern))

    candidates = list(dict.fromkeys([p for p in candidates if p.exists()]))

    good = []
    for p in candidates:
        try:
            df = read_csv(p)
        except Exception:
            continue

        id_col = find_col(df, ["structure_id", "id", "bank_file"], required=False)
        hse_col = find_col(df, ["hse_true", "hse_gap", "hse06_gap", "hse_gap_ev"], required=False)
        pbe_col = find_col(df, ["pbe_gap_effective", "pbe_gap", "pbe_gap_ev"], required=False)

        if id_col and hse_col and pbe_col and len(df) >= 20:
            good.append(p)

    if not good:
        raise FileNotFoundError(
            "Could not auto-find a gate-aware test prediction CSV. "
            "Search manually with:\n"
            "find paper-runs/delta_kg2_m6_q035_theta016_dosON/al1 -iname '*test*.csv' -print"
        )

    return good[0]


def load_feature_bank(feature_csvs, id_col="structure_id"):
    frames = []

    for path in feature_csvs:
        path = Path(path)
        if not path.exists():
            continue

        df = read_csv(path)
        found_id = find_col(df, [id_col, "structure_id", "id", "bank_file"], required=False)

        if found_id is None:
            continue

        if found_id != id_col:
            df = df.rename(columns={found_id: id_col})

        df[id_col] = df[id_col].astype(str).str.strip()
        frames.append(df)

    if not frames:
        raise RuntimeError("No usable feature CSVs were found.")

    big = pd.concat(frames, ignore_index=True, sort=False)
    big[id_col] = big[id_col].astype(str).str.strip()

    def first_nonnull(series):
        s = series.dropna()
        if len(s):
            return s.iloc[0]
        return np.nan

    grouped = big.groupby(id_col, as_index=False).agg(first_nonnull)
    return grouped


def transform_features(test_feat, preproc, model_payload):
    num_cols = preproc.get("B_num_cols", [])
    bool_cols = preproc.get("B_bool_cols", [])
    fill_values = preproc.get("B_fill_values", {})
    scaler = model_payload["reg_scaler"]

    missing = [c for c in num_cols + bool_cols if c not in test_feat.columns]
    if missing:
        raise RuntimeError(
            "Missing model features in test feature bank:\n"
            + "\n".join(missing[:100])
        )

    X_num = pd.DataFrame(index=test_feat.index)
    for c in num_cols:
        x = pd.to_numeric(test_feat[c], errors="coerce")
        fill = fill_values.get(c, np.nan)
        if not np.isfinite(fill):
            fill = 0.0
        X_num[c] = x.replace([np.inf, -np.inf], np.nan).fillna(float(fill))

    X_bool = pd.DataFrame(index=test_feat.index)
    for c in bool_cols:
        X_bool[c] = pd.to_numeric(test_feat[c], errors="coerce").fillna(0).astype(int)

    X_num_scaled = scaler.transform(X_num.values) if num_cols else np.zeros((len(test_feat), 0))
    X_bool_arr = X_bool.values.astype(float) if bool_cols else np.zeros((len(test_feat), 0))

    return np.hstack([X_num_scaled, X_bool_arr])


def evaluate_one_run(run_dir, test_df, feature_bank, cycle=24):
    model_path = run_dir / "models" / f"phase2_cycle{cycle}.joblib"
    preproc_path = run_dir / "models" / f"preproc_cycle{cycle}.json"

    if not model_path.exists():
        raise FileNotFoundError(str(model_path))
    if not preproc_path.exists():
        raise FileNotFoundError(str(preproc_path))

    payload = joblib.load(model_path)
    preproc = json.loads(preproc_path.read_text())

    id_col = preproc.get("id_col", "structure_id")
    hse_col = "hse_true_for_eval"
    pbe_col = "pbe_gap_effective_for_eval"
    status_col = "label_status_for_eval"

    model = payload["gpr"]

    merged = test_df.merge(feature_bank, on=id_col, how="left", suffixes=("", "_feature"))

    X = transform_features(merged, preproc, payload)

    mu_delta, sigma = model.predict(X, return_std=True)

    pbe = pd.to_numeric(merged[pbe_col], errors="coerce").to_numpy(dtype=float)
    hse_true = pd.to_numeric(merged[hse_col], errors="coerce").to_numpy(dtype=float)

    status = (
        merged[status_col]
        .fillna("")
        .astype(str)
        .str.upper()
        .to_numpy()
    )

    # METAL_ASSUMED rows are true zero-gap for end-to-end HSE evaluation.
    hse_true = np.where(
        (~np.isfinite(hse_true)) & (status == "METAL_ASSUMED"),
        0.0,
        hse_true,
    )

    hse_pred = pbe + np.asarray(mu_delta, dtype=float)
    hse_pred = np.clip(hse_pred, 0.0, None)

    delta_true = hse_true - pbe
    delta_pred = np.asarray(mu_delta, dtype=float)

    is_baseline_zbg = np.isfinite(pbe) & (pbe <= PBE_ZERO_THR)
    true_metal = (hse_true <= HSE_ZERO_THR) | (status == "METAL_ASSUMED")
    true_finite = (~true_metal) & np.isfinite(hse_true) & (hse_true > HSE_ZERO_THR)

    non_zbg = np.isfinite(pbe) & (pbe > PBE_ZERO_THR)
    non_zbg_delta = non_zbg & true_finite

    pred_finite = hse_pred > HSE_ZERO_THR

    pred = merged[[id_col]].copy()
    pred["run"] = run_dir.name
    pred["pbe_gap_effective"] = pbe
    pred["hse_true"] = hse_true
    pred["hse_pred"] = hse_pred
    pred["delta_true"] = delta_true
    pred["delta_pred"] = delta_pred
    pred["sigma_gpr"] = sigma
    pred["is_baseline_zbg"] = is_baseline_zbg.astype(int)
    pred["true_metal"] = true_metal.astype(int)
    pred["true_finite"] = true_finite.astype(int)
    pred["pred_finite"] = pred_finite.astype(int)
    pred["method"] = "nogate_fullpool_AL"
    pred["model_route"] = "unified_delta_gpr"

    metrics = []

    def add_reg(subset_name, target_name, mask, y_true, y_pred):
        m = np.asarray(mask, dtype=bool)
        met = safe_metrics(np.asarray(y_true)[m], np.asarray(y_pred)[m])
        metrics.append({
            "run": run_dir.name,
            "dataset": "test",
            "subset": subset_name,
            "target": target_name,
            **met,
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
            "tn": np.nan,
            "fp": np.nan,
            "fn": np.nan,
            "tp": np.nan,
        })

    add_reg("all", "HSE06_gap", np.isfinite(hse_true), hse_true, hse_pred)
    add_reg("non_ZBG", "HSE06_gap", non_zbg, hse_true, hse_pred)
    add_reg("non_ZBG", "Delta", non_zbg_delta, delta_true, delta_pred)
    add_reg("baseline_ZBG", "HSE06_gap", is_baseline_zbg, hse_true, hse_pred)
    add_reg("baseline_ZBG_finite_true", "HSE06_gap", is_baseline_zbg & true_finite, hse_true, hse_pred)
    add_reg("baseline_ZBG_true_metal", "HSE06_gap", is_baseline_zbg & true_metal, hse_true, hse_pred)

    zmask = is_baseline_zbg
    cls = rescue_metrics(true_finite[zmask].astype(int), pred_finite[zmask].astype(int))
    metrics.append({
        "run": run_dir.name,
        "dataset": "test",
        "subset": "baseline_ZBG",
        "target": "implicit_rescue_decision",
        "n": cls["n"],
        "mae": np.nan,
        "rmse": np.nan,
        "r2": np.nan,
        "precision": cls["precision"],
        "recall": cls["recall"],
        "f1": cls["f1"],
        "tn": cls["tn"],
        "fp": cls["fp"],
        "fn": cls["fn"],
        "tp": cls["tp"],
    })

    return pred, pd.DataFrame(metrics)


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--nogate-root", default=str(DEFAULT_NOGATE_ROOT))
    ap.add_argument("--gate-root", default=str(DEFAULT_GATE_ROOT))
    ap.add_argument("--cycle", type=int, default=24)
    ap.add_argument("--truth-csv", default="")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUTDIR))
    ap.add_argument("--feature-csv", action="append", default=[])
    args = ap.parse_args()

    nogate_root = Path(args.nogate_root)
    gate_root = Path(args.gate_root)
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    truth_csv = Path(args.truth_csv) if args.truth_csv else find_truth_csv(gate_root)
    print(f"[truth] using: {truth_csv}")

    truth_raw = read_csv(truth_csv)

    id_col_truth = find_col(truth_raw, ["structure_id", "id", "bank_file"], required=True)
    hse_col_truth = find_col(truth_raw, ["hse_true", "hse_gap", "hse06_gap", "hse_gap_ev"], required=True)
    pbe_col_truth = find_col(truth_raw, ["pbe_gap_effective", "pbe_gap", "pbe_gap_ev"], required=True)
    status_col_truth = find_col(truth_raw, ["label_status", "status", "hse_status"], required=False)

    test_df = pd.DataFrame({
        "structure_id": truth_raw[id_col_truth].astype(str).str.strip(),
        "hse_true_for_eval": pd.to_numeric(truth_raw[hse_col_truth], errors="coerce"),
        "pbe_gap_effective_for_eval": pd.to_numeric(truth_raw[pbe_col_truth], errors="coerce"),
        "label_status_for_eval": truth_raw[status_col_truth].astype(str) if status_col_truth else "",
    })

    if "formula" in truth_raw.columns:
        test_df["formula"] = truth_raw["formula"].astype(str)

    test_df = test_df.drop_duplicates(subset=["structure_id"], keep="first").copy()

    print(f"[truth] test rows = {len(test_df)}")

    feature_csvs = []

    # User-supplied feature CSVs.
    feature_csvs.extend([Path(p) for p in args.feature_csv])

    # Default global feature bank.
    feature_csvs.append(REPO / "data" / "trainpool_ml_augmented.csv")

    # Run-local feature banks.
    feature_csvs.extend(sorted(nogate_root.glob("al*/pool_trainval.csv")))
    feature_csvs.extend(sorted(gate_root.glob("al*/pool_trainval.csv")))

    feature_csvs = list(dict.fromkeys(feature_csvs))

    print("[features] sources:")
    for p in feature_csvs:
        print(f"  {p}")

    feature_bank = load_feature_bank(feature_csvs, id_col="structure_id")
    print(f"[features] bank rows = {len(feature_bank)}")

    preds_all = []
    metrics_all = []

    for run_dir in sorted(nogate_root.glob("al*")):
        if not run_dir.is_dir():
            continue

        print(f"[eval] {run_dir.name}")

        pred, met = evaluate_one_run(
            run_dir=run_dir,
            test_df=test_df,
            feature_bank=feature_bank,
            cycle=args.cycle,
        )

        preds_all.append(pred)
        metrics_all.append(met)

    pred_df = pd.concat(preds_all, ignore_index=True)
    met_df = pd.concat(metrics_all, ignore_index=True)

    pred_path = outdir / "nogate_fullpool_test_predictions_cycle24.csv"
    metrics_path = outdir / "nogate_fullpool_test_metrics_by_run_cycle24.csv"
    summary_path = outdir / "nogate_fullpool_test_metrics_summary_cycle24.csv"
    per_structure_path = outdir / "nogate_fullpool_test_per_structure_mean_cycle24.csv"

    pred_df.to_csv(pred_path, index=False)
    met_df.to_csv(metrics_path, index=False)

    metric_cols = ["mae", "rmse", "r2", "precision", "recall", "f1", "tn", "fp", "fn", "tp"]
    summary = (
        met_df
        .groupby(["dataset", "subset", "target"])[metric_cols]
        .agg(["mean", "std", "min", "max"])
    )
    summary.to_csv(summary_path)

    per_structure = (
        pred_df
        .groupby("structure_id", as_index=False)
        .agg(
            hse_true=("hse_true", "first"),
            pbe_gap_effective=("pbe_gap_effective", "first"),
            hse_pred_mean=("hse_pred", "mean"),
            hse_pred_sd=("hse_pred", "std"),
            delta_true=("delta_true", "first"),
            delta_pred_mean=("delta_pred", "mean"),
            delta_pred_sd=("delta_pred", "std"),
            is_baseline_zbg=("is_baseline_zbg", "first"),
            true_metal=("true_metal", "first"),
            true_finite=("true_finite", "first"),
        )
    )
    per_structure.to_csv(per_structure_path, index=False)

    print("\nDONE")
    print(f"Wrote: {pred_path}")
    print(f"Wrote: {metrics_path}")
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {per_structure_path}")

    print("\nKey no-gate full-pool test metrics:")
    show = met_df[
        (met_df["dataset"].eq("test"))
        & (
            (met_df["subset"].eq("all") & met_df["target"].eq("HSE06_gap"))
            | (met_df["subset"].eq("non_ZBG") & met_df["target"].eq("Delta"))
            | (met_df["subset"].eq("baseline_ZBG") & met_df["target"].eq("HSE06_gap"))
            | (met_df["subset"].eq("baseline_ZBG") & met_df["target"].eq("implicit_rescue_decision"))
        )
    ].copy()

    print(show.to_string(index=False))


if __name__ == "__main__":
    main()
