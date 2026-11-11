#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Final TEST evaluation for a finished AL run (NO AL, NO DFT).

This script evaluates one saved AL checkpoint on the held-out test set.

It reports two different quantities:

1. End-to-end HSE test error
   - compares hse_pred vs hse_true after the full routed workflow:
       gate + Stage-B + Stage-C/PBE-ZBG routing

2. Stage-B non-PBE-ZBG Delta error
   - compares delta_pred_stageB vs delta_true only for structures with:
       E_g^PBE > pbe_zero_thr
   - this is the test-set analogue of delta_val_MAE_nonmetal

Important terminology:
- PBE-ZBG means PBE-zero-band-gap, i.e. E_g^PBE <= threshold.
- Do not call this PBE0 in the manuscript, because PBE0 is a hybrid functional.
- The internal code column is kept as is_pbe0 only for compatibility with older files.

Inputs:
    <run-dir>/models/phase2_cycle<CYCLE>.joblib
    <run-dir>/models/preproc_cycle<CYCLE>.json
    test_labels_labeled.csv
    trainpool_ml_augmented.csv

Outputs:
    <run-dir>/test_eval/test_pred_cycle<CYCLE>.csv
    <run-dir>/test_eval/test_metrics_cycle<CYCLE>.json

Example:
    python3 ALL-trials/evaluate_test_cycle.py \
      --run-dir kg2-al-runs/al1 \
      --pool trainpool_ml_augmented.csv \
      --test test_labels_labeled.csv \
      --cycle 24
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error


def last_cycle(models_dir: Path) -> int:
    cycles = []
    for p in models_dir.glob("phase2_cycle*.joblib"):
        try:
            cycles.append(int(p.stem.split("cycle")[1]))
        except Exception:
            pass
    if not cycles:
        raise FileNotFoundError("No phase2_cycle*.joblib found in {}".format(models_dir))
    return max(cycles)


def as_id(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()


def pick_feature(df: pd.DataFrame, name: str) -> pd.Series:
    """
    Prefer pool-side augmented features, then unsuffixed, then test-side.
    """
    pool_name = "{}_pool".format(name)
    test_name = "{}_test".format(name)

    if pool_name in df.columns:
        return df[pool_name]
    if name in df.columns:
        return df[name]
    if test_name in df.columns:
        return df[test_name]

    return pd.Series([np.nan] * len(df), index=df.index)


def pick_test_label(df: pd.DataFrame, name: str) -> pd.Series:
    """
    Prefer test-side truth, then unsuffixed, then pool-side.
    This is used for hse_gap and pbe_gap truth/baseline labels.
    """
    test_name = "{}_test".format(name)
    pool_name = "{}_pool".format(name)

    if test_name in df.columns:
        return df[test_name]
    if name in df.columns:
        return df[name]
    if pool_name in df.columns:
        return df[pool_name]

    return pd.Series([np.nan] * len(df), index=df.index)


def add_dos_features_if_needed(
    df: pd.DataFrame,
    id_col: str,
    need_dos_cols: List[str],
    dos_path: Path,
) -> pd.DataFrame:
    """
    Adds DOS-derived columns if the saved model expects any _dos_* columns
    and they are not already present.

    This keeps the evaluator compatible with older model checkpoints.
    """
    if not need_dos_cols:
        return df

    if all(c in df.columns for c in need_dos_cols):
        return df

    if not dos_path.exists():
        print("[WARN] Model expects DOS features but {} not found. Continuing without DOS merge.".format(dos_path))
        return df

    dos = pd.read_csv(dos_path)

    if id_col not in dos.columns:
        print("[WARN] DOS file does not contain id column '{}'. Continuing without DOS merge.".format(id_col))
        return df

    # Best-effort filters
    if "parsed_ok" in dos.columns:
        dos["parsed_ok"] = pd.to_numeric(dos["parsed_ok"], errors="coerce").fillna(0).astype(int)
        dos = dos.loc[dos["parsed_ok"] == 1].copy()

    if "has_doscar" in dos.columns:
        dos["has_doscar"] = pd.to_numeric(dos["has_doscar"], errors="coerce").fillna(0).astype(int)
        dos = dos.loc[dos["has_doscar"] == 1].copy()

    dos[id_col] = as_id(dos[id_col])
    dos = dos.drop_duplicates(subset=[id_col], keep="last")

    df = df.merge(dos, on=id_col, how="left", validate="many_to_one")

    # Derived columns matching training logic
    dosEf = pd.to_numeric(df.get("dosEf", np.nan), errors="coerce").astype(float)
    df["_dos_log_dosEf"] = np.log1p(np.clip(dosEf.to_numpy(float), 0.0, None))

    maxdos = pd.to_numeric(df.get("maxDOS_between", np.nan), errors="coerce").astype(float)
    df["_dos_log_maxDOS_between"] = np.log1p(np.clip(maxdos.to_numpy(float), 0.0, None))

    gap_dos = pd.to_numeric(df.get("gap", np.nan), errors="coerce").astype(float)
    df["_dos_log_gap"] = np.log1p(np.clip(gap_dos.to_numpy(float), 0.0, None))

    dr = pd.to_numeric(df.get("dist_ratio", np.nan), errors="coerce").astype(float).to_numpy(float)
    dr = np.clip(dr, 1e-6, 1e6)
    df["_dos_log_dist_ratio"] = np.log(dr)
    df["_dos_abs_log_dist_ratio"] = np.abs(df["_dos_log_dist_ratio"].to_numpy(float))

    df["_dos_d_below"] = pd.to_numeric(df.get("d_below", np.nan), errors="coerce").astype(float)
    df["_dos_d_above"] = pd.to_numeric(df.get("d_above", np.nan), errors="coerce").astype(float)
    df["_dos_is_gap"] = pd.to_numeric(df.get("is_gap", 0), errors="coerce").fillna(0).astype(int)

    return df


def build_X(
    df: pd.DataFrame,
    num_cols: List[str],
    bool_cols: List[str],
    fill_values: Dict[str, Any],
    scaler,
) -> np.ndarray:
    """
    Build numeric + boolean feature matrix.

    The scaler is fit only on num_cols, so len(num_cols) must match
    scaler.n_features_in_.
    """
    Xn_list = []

    for c in num_cols:
        s = pd.to_numeric(pick_feature(df, c), errors="coerce").replace([np.inf, -np.inf], np.nan)

        try:
            fill = float(fill_values.get(c, 0.0))
        except Exception:
            fill = 0.0

        Xn_list.append(s.fillna(fill).to_numpy(float))

    Xn = np.vstack(Xn_list).T if Xn_list else np.zeros((len(df), 0), float)

    if Xn.shape[1]:
        exp = getattr(scaler, "n_features_in_", None)
        if exp is not None and int(exp) != Xn.shape[1]:
            raise ValueError(
                "Scaler feature mismatch: scaler expects {} numeric features, built {}. "
                "Check payload fill-value keys and feature construction.".format(exp, Xn.shape[1])
            )
        Xn = scaler.transform(Xn)

    Xb_list = []

    for c in bool_cols:
        s = pd.to_numeric(pick_feature(df, c), errors="coerce").fillna(0).astype(int)
        Xb_list.append(s.to_numpy(float))

    Xb = np.vstack(Xb_list).T if Xb_list else np.zeros((len(df), 0), float)

    return np.hstack([Xn, Xb])


def p_gate_from_model(model, Xa: np.ndarray) -> np.ndarray:
    """
    Reconstruct p_gate used by the AL code.

    Internal convention from previous evaluator:
    gate label = metal-like = 1
    p_gate = probability of finite/rescued class = 1 - p_metal
    """
    if model is None:
        return np.full(Xa.shape[0], 0.5, float)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(Xa)
        classes = getattr(model, "classes_", None)

        if classes is None:
            p_metal = proba[:, 1]
        else:
            classes = np.asarray(classes)
            idx1 = int(np.where(classes == 1)[0][0]) if np.any(classes == 1) else 1
            p_metal = proba[:, idx1]

        return 1.0 - np.clip(p_metal, 0.0, 1.0)

    if hasattr(model, "decision_function"):
        z = np.asarray(model.decision_function(Xa), float)
        p_metal = 1.0 / (1.0 + np.exp(-z))
        return 1.0 - np.clip(p_metal, 0.0, 1.0)

    p_metal = np.asarray(model.predict(Xa), float)
    return 1.0 - np.clip(p_metal, 0.0, 1.0)


def safe_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    return float(mean_absolute_error(y_true[m], y_pred[m])) if m.sum() else float("nan")


def safe_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    return float(np.sqrt(mean_squared_error(y_true[m], y_pred[m]))) if m.sum() else float("nan")


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() < 2:
        return float("nan")

    yt = y_true[m]
    yp = y_pred[m]

    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))

    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="run_v3")
    ap.add_argument("--pool", default="trainpool_ml_augmented.csv")
    ap.add_argument("--test", default="test_labels_labeled.csv")
    ap.add_argument("--dos", default="dos_from_output.csv")
    ap.add_argument("--cycle", type=int, default=None)
    ap.add_argument("--out-dir", default="",
                    help="output folder (default: <run-dir>/test_eval)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    models_dir = run_dir / "models"

    cycle = args.cycle

    if cycle is None and "CYCLE" in os.environ:
        cycle = int(os.environ["CYCLE"])

    if cycle is None:
        cycle = last_cycle(models_dir)

    model_path = models_dir / "phase2_cycle{}.joblib".format(cycle)
    preproc_path = models_dir / "preproc_cycle{}.json".format(cycle)

    if not model_path.exists():
        raise FileNotFoundError("Missing model: {}".format(model_path))
    if not preproc_path.exists():
        raise FileNotFoundError("Missing preproc: {}".format(preproc_path))

    payload = joblib.load(model_path)
    pre = json.loads(preproc_path.read_text())

    id_col = pre["id_col"]
    hse_col = pre["hse_col"]
    pool_pbe_col = pre["pool_pbe_col"]

    pbe_zero_thr = float(pre.get("pbe_zero_thr", 0.05))
    gate_decision_thr = float(pre.get("gate_decision_thr", 0.5))
    target_mode = str(payload.get("target_mode", pre.get("target_mode", "delta")))

    pool = pd.read_csv(args.pool)
    test = pd.read_csv(args.test)

    if id_col not in pool.columns:
        raise KeyError("Pool file missing id column '{}'".format(id_col))
    if id_col not in test.columns:
        raise KeyError("Test file missing id column '{}'".format(id_col))

    pool[id_col] = as_id(pool[id_col])
    test[id_col] = as_id(test[id_col])

    df = test.merge(pool, on=id_col, how="inner", suffixes=("_test", "_pool"))

    if len(df) == 0:
        raise SystemExit("ERROR: merge produced 0 rows; test IDs not found in {}.".format(args.pool))

    # Use true numeric feature list from payload fill dict keys.
    # This matches scaler fit and avoids all-NaN dropped-column mismatch.
    fillA = payload.get("clf_fill_values", pre.get("A_fill_values", {}))
    fillB = payload.get("reg_fill_values", pre.get("B_fill_values", {}))

    A_num = list(fillA.keys())
    B_num = list(fillB.keys())

    A_bool = list(pre.get("A_bool_cols", []))
    B_bool = list(pre.get("B_bool_cols", []))

    # Add DOS-derived features only if the saved model expects them.
    need_dos = [c for c in (A_num + A_bool + B_num + B_bool) if str(c).startswith("_dos_")]
    df = add_dos_features_if_needed(df, id_col, need_dos, Path(args.dos))

    clf_scaler = payload["clf_scaler"]
    reg_scaler = payload["reg_scaler"]

    expA = getattr(clf_scaler, "n_features_in_", None)
    expB = getattr(reg_scaler, "n_features_in_", None)

    print("[INFO] cycle:", cycle)
    print("[INFO] target_mode:", target_mode)
    print("[INFO] A_num used:", len(A_num), "scaler expects:", expA)
    print("[INFO] B_num used:", len(B_num), "scaler expects:", expB)

    Xa = build_X(df, A_num, A_bool, fillA, clf_scaler)
    Xb = build_X(df, B_num, B_bool, fillB, reg_scaler)

    # Gate
    gate_model = payload.get("gate_classifier", payload.get("classifier", None))
    p_gate = p_gate_from_model(gate_model, Xa)

    # Regression models
    gpr = payload["gpr"]
    gpr0 = payload.get("gpr_pbe0", None)

    # Force PBE from test labels, fallback to pool only if test PBE is NaN.
    pbe_test = pd.to_numeric(pick_test_label(df, pool_pbe_col), errors="coerce").to_numpy(float)
    pbe_pool = pd.to_numeric(pick_feature(df, pool_pbe_col), errors="coerce").to_numpy(float)
    pbe = np.where(np.isfinite(pbe_test), pbe_test, pbe_pool)

    mism = np.isfinite(pbe_test) & np.isfinite(pbe_pool) & (np.abs(pbe_test - pbe_pool) > 1e-6)

    print("[INFO] merged test rows:", len(df))
    print("[INFO] pbe mismatches (>1e-6): {} / {} (USING TEST pbe)".format(int(mism.sum()), len(df)))

    is_PBE_ZBG = np.isfinite(pbe) & (pbe <= pbe_zero_thr)

    # Stage-B target prediction.
    # If target_mode = delta, mu_stageB is predicted Δ.
    # If target_mode = direct_hse, mu_stageB is predicted HSE.
    mu_stageB, sig_stageB = gpr.predict(Xb, return_std=True)
    mu_stageB = np.asarray(mu_stageB, float)
    sig_stageB = np.asarray(sig_stageB, float)

    if target_mode == "delta":
        delta_pred_stageB = mu_stageB.copy()
        hse_stageB = pbe + delta_pred_stageB
    elif target_mode == "direct_hse":
        hse_stageB = mu_stageB.copy()
        delta_pred_stageB = hse_stageB - pbe
    else:
        raise ValueError("Unknown target_mode: {}".format(target_mode))

    # Stage-C / PBE-ZBG rescue model.
    # For your delta-mode saved models, gpr_pbe0 predicts finite rescued HSE directly.
    if gpr0 is not None:
        mu0, sig0 = gpr0.predict(Xb, return_std=True)
        hse_stageC = np.asarray(mu0, float)
        sig_stageC = np.asarray(sig0, float)
    else:
        hse_stageC = hse_stageB.copy()
        sig_stageC = sig_stageB.copy()

    # Routed end-to-end HSE prediction:
    # - non-PBE-ZBG: Stage-B HSE
    # - PBE-ZBG + p_gate < threshold: metal-like -> 0
    # - PBE-ZBG + p_gate >= threshold: rescued finite-gap -> Stage-C HSE
    is_metal_route = is_PBE_ZBG & (p_gate < gate_decision_thr)
    is_rescued_route = is_PBE_ZBG & (p_gate >= gate_decision_thr)

    hse_pred = hse_stageB.copy()
    hse_pred[is_metal_route] = 0.0
    hse_pred[is_rescued_route] = hse_stageC[is_rescued_route]
    hse_pred = np.maximum(hse_pred.astype(float), 0.0)

    # Truth
    hse_true = pd.to_numeric(pick_test_label(df, hse_col), errors="coerce").to_numpy(float)
    delta_true = hse_true - pbe

    # End-to-end HSE metrics
    m = np.isfinite(hse_true) & np.isfinite(hse_pred)
    mae = safe_mae(hse_true, hse_pred)
    rmse = safe_rmse(hse_true, hse_pred)
    r2 = safe_r2(hse_true, hse_pred)

    # Meaningful Delta metrics:
    # Stage-B Δ error on non-PBE-ZBG structures only.
    # This matches the validation metric delta_val_MAE_nonmetal.
    non_PBE_ZBG = np.isfinite(pbe) & (pbe > pbe_zero_thr)

    m_delta_nm = (
        np.isfinite(delta_true)
        & np.isfinite(delta_pred_stageB)
        & non_PBE_ZBG
    )

    delta_nm_mae = (
        float(mean_absolute_error(delta_true[m_delta_nm], delta_pred_stageB[m_delta_nm]))
        if m_delta_nm.sum()
        else float("nan")
    )

    delta_nm_rmse = (
        float(np.sqrt(mean_squared_error(delta_true[m_delta_nm], delta_pred_stageB[m_delta_nm])))
        if m_delta_nm.sum()
        else float("nan")
    )

    # Regime-specific routed HSE metrics
    m_zbg = m & is_PBE_ZBG
    m_non_zbg = m & non_PBE_ZBG

    PBE_ZBG_MAE = (
        float(mean_absolute_error(hse_true[m_zbg], hse_pred[m_zbg]))
        if m_zbg.sum()
        else float("nan")
    )

    PBE_ZBG_RMSE = (
        float(np.sqrt(mean_squared_error(hse_true[m_zbg], hse_pred[m_zbg])))
        if m_zbg.sum()
        else float("nan")
    )

    non_PBE_ZBG_MAE = (
        float(mean_absolute_error(hse_true[m_non_zbg], hse_pred[m_non_zbg]))
        if m_non_zbg.sum()
        else float("nan")
    )

    non_PBE_ZBG_RMSE = (
        float(np.sqrt(mean_squared_error(hse_true[m_non_zbg], hse_pred[m_non_zbg])))
        if m_non_zbg.sum()
        else float("nan")
    )

    outdir = Path(args.out_dir) if args.out_dir else run_dir / "test_eval"
    outdir.mkdir(parents=True, exist_ok=True)

    pred = pd.DataFrame({
        id_col: df[id_col].astype(str).values,
        "pbe_gap_test": pbe_test,
        "pbe_gap_pool": pbe_pool,
        "pbe_gap_effective": pbe,

        # Use PBE-ZBG terminology in output.
        # Kept old internal is_pbe0 only if needed by older plotting scripts.
        "is_PBE_ZBG": is_PBE_ZBG.astype(int),
        "is_pbe0": is_PBE_ZBG.astype(int),

        "p_gate": p_gate,
        "is_metal_route": is_metal_route.astype(int),
        "is_rescued_route": is_rescued_route.astype(int),

        "hse_true": hse_true,
        "hse_pred": hse_pred,
        "hse_stageB": hse_stageB,
        "hse_stageC": hse_stageC,

        "delta_true": delta_true,
        "delta_pred_stageB": delta_pred_stageB,
        "abs_delta_error_stageB": np.abs(delta_pred_stageB - delta_true),

        "sigma_stageB": sig_stageB,
        "sigma_stageC": sig_stageC,
    })

    pred_path = outdir / "test_pred_cycle{}.csv".format(cycle)
    pred.to_csv(pred_path, index=False)

    metrics = {
        "cycle": cycle,
        "pool": str(Path(args.pool)),
        "test": str(Path(args.test)),
        "n_test_merged": int(len(df)),
        "n_test_used": int(m.sum()),

        # End-to-end HSE test metrics
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,

        # Test-set analogue of delta_val_MAE_nonmetal
        "Delta_NM_MAE": delta_nm_mae,
        "Delta_NM_RMSE": delta_nm_rmse,
        "Delta_NM_n": int(m_delta_nm.sum()),

        # Routed regime-specific HSE metrics
        "PBE_ZBG_MAE": PBE_ZBG_MAE,
        "PBE_ZBG_RMSE": PBE_ZBG_RMSE,
        "PBE_ZBG_n": int(m_zbg.sum()),
        "non_PBE_ZBG_MAE": non_PBE_ZBG_MAE,
        "non_PBE_ZBG_RMSE": non_PBE_ZBG_RMSE,
        "non_PBE_ZBG_n": int(m_non_zbg.sum()),

        # Routing/settings
        "pbe_zero_thr": pbe_zero_thr,
        "gate_decision_thr": gate_decision_thr,
        "target_mode": target_mode,
        "pbe_source": "TEST (fallback to pool only if NaN)",
        "pbe_mismatches_gt_1e6": int(mism.sum()),

        # Paths/debug
        "model_path": str(model_path),
        "preproc_path": str(preproc_path),
        "A_num_used": len(A_num),
        "B_num_used": len(B_num),
    }

    metrics_path = outdir / "test_metrics_cycle{}.json".format(cycle)
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print("[OK] wrote", pred_path)
    print("[OK] wrote", metrics_path)
    print(
        "[TEST] cycle={}  HSE_MAE={}  HSE_RMSE={}  R2={}  Delta_NM_MAE={}  Delta_NM_RMSE={}".format(
            cycle,
            mae,
            rmse,
            r2,
            delta_nm_mae,
            delta_nm_rmse,
        )
    )


if __name__ == "__main__":
    main()
