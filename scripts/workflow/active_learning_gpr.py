#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: active_learning_gpr.py

"""
AL v3: Δ-learning + PBE≈0 Gate Active Learning (single-phase)
=============================================================

Goal (v3 gate protocol; NO multiphase):
- Start from a fixed labeled seed (trusted labels, typically 100 rows).
- Freeze a stratified fixed VAL set of 30 IDs ONCE at cycle 0
  (outdir/val_ids.csv; legacy outdir/phase2_val_ids.csv is also accepted).
- Keep HSE bin logic for stratification/reporting: <=0.05, 0.05-1, 1-2, >2.
- Keep Δ-learning (Stage B) ONLY on truth nonmetals with finite HSE and finite PBE_eff:
      Δ = HSE - PBE_eff.
- Add a NEW classifier “gate” model focused ONLY on PBE≈0 candidates:
    gate domain: PBE_eff <= pbe_zero_thr
    label: 0 = true metal   (HSE <= metal_thr_ev OR status == METAL_ASSUMED)
           1 = rescued nonmetal (HSE > metal_thr_ev)
  Gate does NOT predict HSE; it only predicts rescued-nonmetal probability.

Routing / prediction (replaces old tau routing):
- If PBE_eff > pbe_zero_thr: predict HSE = PBE_eff + Δ_pred.
- If PBE_eff <= pbe_zero_thr:
    use p_gate = P(rescued nonmetal):
      if p_gate < gate_decision_thr: treat as metal (HSE_pred = 0 for reporting)
      else: treat as nonmetal (HSE_pred = PBE_eff + Δ_pred; with PBE≈0 this ~ Δ_pred)

Acquisition per cycle:
- Keep Δ acquisition logic for non-PBE0 candidates (score_base).
- Add a gate acquisition bucket only from PBE≈0 subset:
    score_gate = entropy(p_gate) + density reward (+ anti-isolation), FPS-diversified.
- Deterministic quota:
    select k_gate from PBE≈0 bucket and k_reg from the rest, FPS within each bucket,
    then merge (fill deterministically if a bucket is short).

Metrics per cycle:
1) Δ regression metrics on VAL truth nonmetals with finite PBE_eff.
2) Gate metrics on VAL restricted to PBE≈0 subset: AUC/AP + F1/prec/rec @ 0.5 + confusion.
3) End-to-end MAE on full VAL (HSE) using gate-based routing.
4) Optional bin breakdown.

Outputs:
- outdir/metrics/metrics.csv includes gate_* fields (auc/ap/f1/prec/rec, n0, confusion counts).
- outdir/metrics/val_pred_cycle*.csv includes p_gate/entropy_gate/is_pbe0 + bins.
- outdir/predictions/pred_cycle*.csv includes score_base/score_gate + p_gate/entropy_gate + sigma_eff + density terms.

Python 3.8 compatible.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Union
import re
import numpy as np
import pandas as pd

try:
    import joblib
except Exception:
    joblib = None

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel as C, WhiteKernel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from sklearn.model_selection import GroupKFold, KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight


# Always-drop features (near-constant / QC-only / unstable under scaling / audit-only)
DEFAULT_SELECT_FEATURE_BLACKLIST: Set[str] = {
    # missingness flags (audit-only by default)
    "ie_missing_A", "ie_missing_B1", "ie_missing_B2",
    # QC flags (audit-only by default)
    "qc_structure_file_exists", "qc_structure_parsed", "qc_structure_loaded",
    "qc_B1_cn_ok", "qc_B2_cn_ok", "qc_A_cn_ok", "qc_is_perovskite_like",
    "qc_gap_present", "qc_bank_exists",
    # near-constant / unstable
    "CN_O_A", "CN_O_B1", "CN_O_B2",
    "is_seed_day0",
    "BOB_B2B2_std", "BOB_B1B1_std",
    "BOB_B2B2_mean", "BOB_B2B2_min", "BOB_B2B2_median",
    "BOB_B1B1_mean", "BOB_B1B1_min", "BOB_B1B1_median",
    # Phase-2 shortcut flags (requested)
    "is_gap0", "is_metal",
}

FORBIDDEN_FEATURE_SUBSTRINGS: Tuple[str, ...] = (
    "labels_store",
    "bank_file",
    "done_dir",
    "fail_reason",
    "where",
    "directory",
    "path",
)


# ----------------------------- small utilities -----------------------------
def _f1_binary(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    denom = (2 * tp + fp + fn)
    return 0.0 if denom == 0 else (2.0 * tp) / denom


def log(msg: str) -> None:
    print(msg, flush=True)


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_csv(p: Path) -> pd.DataFrame:
    if not p.exists():
        raise FileNotFoundError(str(p))
    return pd.read_csv(p)


def dump_state(p: Path, st: dict) -> None:
    safe_mkdir(p.parent)
    p.write_text(json.dumps(st, indent=2))


def load_state(p: Path) -> dict:
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def dump_json(p: Path, obj: dict) -> None:
    safe_mkdir(p.parent)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True))


def parse_blacklist(s: str) -> Set[str]:
    if not s:
        return set()
    return set([x.strip() for x in s.split(",") if x.strip()])


def stable_hash_int(s: str, salt: str = "acq_v1") -> int:
    h = hashlib.md5((salt + "|" + str(s)).encode("utf-8")).hexdigest()
    return int(h, 16)


def percentile_rank(values: np.ndarray, higher_is_better: bool = True) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    n = len(v)
    if n == 0:
        return v
    v2 = np.where(np.isfinite(v), v, -np.inf)
    if not higher_is_better:
        v2 = -v2
    order = np.argsort(v2, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1, dtype=float)
    return ranks / float(n)


def _as_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _norm_status(s: Union[str, float]) -> str:
    return str(s).strip().upper()


def is_trusted_numeric_status(s: str) -> bool:
    return _norm_status(s) in {"SUCCESS", "METAL_ASSUMED"}


def compute_metal_like_hse(
    hse: np.ndarray,
    status: pd.Series,
    thr_ev: float = 0.05,
) -> np.ndarray:
    h = np.asarray(hse, dtype=float)
    s = status.fillna("").astype(str).map(_norm_status).values
    by_hse = np.isfinite(h) & (h <= float(thr_ev))
    by_status = (s == "METAL_ASSUMED")
    return (by_hse | by_status).astype(bool)


def entropy_binary(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def ece_binary(y_true: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(p, dtype=float)
    p = np.clip(p, 0.0, 1.0)
    bins = np.linspace(0.0, 1.0, int(n_bins) + 1)
    ece = 0.0
    n = len(y)
    if n == 0:
        return float("nan")
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        if i == len(bins) - 2:
            m = (p >= lo) & (p <= hi)
        else:
            m = (p >= lo) & (p < hi)
        if not np.any(m):
            continue
        acc = float(np.mean(y[m]))
        conf = float(np.mean(p[m]))
        ece += (np.sum(m) / float(n)) * abs(acc - conf)
    return float(ece)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if np.sum(m) < 2:
        return {"MAE": float("nan"), "RMSE": float("nan"), "R2": float("nan"), "n": int(np.sum(m))}
    yt = y_true[m]
    yp = y_pred[m]
    mae = float(np.mean(np.abs(yt - yp)))
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "n": int(len(yt))}


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if len(y_true) < 1:
        return {"bal_acc": float("nan"), "f1": float("nan"), "prec": float("nan"), "rec": float("nan"), "n": 0}
    return {
        "bal_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "prec": float(precision_score(y_true, y_pred, zero_division=0)),
        "rec": float(recall_score(y_true, y_pred, zero_division=0)),
        "n": int(len(y_true)),
    }


def pbe_bins(pbe: np.ndarray) -> np.ndarray:
    x = np.asarray(pbe, dtype=float)
    out = np.full(len(x), "nan", dtype=object)
    out[np.isfinite(x) & (x <= 0.05)] = "<=0.05"
    out[np.isfinite(x) & (x > 0.05) & (x <= 1.0)] = "0.05-1"
    out[np.isfinite(x) & (x > 1.0) & (x <= 2.0)] = "1-2"
    out[np.isfinite(x) & (x > 2.0)] = ">2"
    return out


def gap_bins(x: np.ndarray) -> np.ndarray:
    """Bin a gap-like scalar using the fixed thresholds used throughout v3."""
    return pbe_bins(x)


def load_or_create_fixed_val_ids(
    outdir: Path,
    seed: pd.DataFrame,
    *,
    id_col: str,
    status_col: str,
    hse_col: str,
    metal_thr_ev: float,
    cycle: int,
    val_size: int = 30,
    salt: str = "val_v3",
) -> List[str]:
    """
    Load a fixed VAL id list from outdir/val_ids.csv (or legacy phase2_val_ids.csv),
    or deterministically create it by stratifying on HSE bins (trusted labels only).

    A new VAL list is created only at cycle 0. At any later cycle a missing file
    raises FileNotFoundError, because a list re-created from the grown ledger
    would differ from the one used so far. A file that exists but holds no IDs
    raises ValueError, because an empty list would exclude nothing from training.

    Determinism: stable_hash_int(id, salt) ordering within each bin.
    """
    val_paths = [outdir / "val_ids.csv", outdir / "phase2_val_ids.csv"]
    for vp in val_paths:
        if vp.exists():
            df = pd.read_csv(vp)
            if id_col in df.columns:
                ids = df[id_col].astype(str).str.strip().tolist()
            else:
                ids = df.iloc[:, 0].astype(str).str.strip().tolist()
            ids = [s for s in ids if s]
            if not ids:
                raise ValueError(f"[val] VAL id file holds no IDs: {vp}")
            log(f"[val] loaded fixed VAL ids: n={len(ids)} from {vp}")
            return ids

    if int(cycle) != 0:
        raise FileNotFoundError(
            f"[val] no VAL id file in {outdir} at cycle {int(cycle)} "
            f"(tried: {', '.join(v.name for v in val_paths)}); "
            "a VAL list is created only at cycle 0"
        )

    safe_mkdir(outdir)

    st = seed[status_col].astype(str).str.strip().str.upper()
    trusted = st.isin(["SUCCESS", "METAL_ASSUMED"])

    hse = pd.to_numeric(seed[hse_col], errors="coerce").astype(float)
    # For METAL_ASSUMED without HSE, treat as 0 for binning/stratification
    hse_for_bin = hse.copy()
    hse_for_bin[(~np.isfinite(hse_for_bin.values)) & (st == "METAL_ASSUMED")] = 0.0

    bins = gap_bins(hse_for_bin.values)
    ok = trusted.values & (bins != "nan")

    df = seed.loc[ok, [id_col]].copy()
    df[id_col] = df[id_col].astype(str).str.strip()
    df["__bin"] = bins[ok].astype(str)

    if len(df) < val_size:
        raise RuntimeError(f"Not enough trusted labeled rows to form VAL={val_size}. Have {len(df)}.")

    bin_order = ["<=0.05", "0.05-1", "1-2", ">2"]
    counts = df["__bin"].value_counts().to_dict()
    total = float(sum(counts.get(b, 0) for b in bin_order))

    alloc = {b: 0 for b in bin_order}
    if total > 0:
        raw = {b: (val_size * counts.get(b, 0) / total) for b in bin_order}
        floor = {b: int(math.floor(raw[b])) for b in bin_order}
        rem = val_size - sum(floor.values())
        frac = sorted(bin_order, key=lambda b: (raw[b] - floor[b]), reverse=True)
        alloc = floor
        for b in frac:
            if rem <= 0:
                break
            if counts.get(b, 0) > alloc[b]:
                alloc[b] += 1
                rem -= 1

    chosen: List[str] = []
    for b in bin_order:
        k = min(alloc.get(b, 0), counts.get(b, 0))
        if k <= 0:
            continue
        sub = df.loc[df["__bin"] == b, [id_col]].copy()
        sub["__h"] = sub[id_col].map(lambda s: stable_hash_int(str(s), salt=salt))
        sub = sub.sort_values("__h", ascending=True)
        chosen += sub[id_col].head(k).astype(str).tolist()

    if len(chosen) < val_size:
        remain = df.loc[~df[id_col].isin(set(chosen)), [id_col]].copy()
        remain["__h"] = remain[id_col].map(lambda s: stable_hash_int(str(s), salt=salt))
        remain = remain.sort_values("__h", ascending=True)
        chosen += remain[id_col].head(val_size - len(chosen)).astype(str).tolist()

    chosen = chosen[:val_size]
    pd.DataFrame({id_col: chosen}).to_csv(outdir / "val_ids.csv", index=False)
    log(f"[val] created fixed VAL ids: n={len(chosen)} -> {outdir / 'val_ids.csv'}")
    return chosen

# ----------------------------- pool hygiene -----------------------------

def apply_safe_mode_filter(pool: pd.DataFrame) -> pd.DataFrame:
    for c in ["qc_structure_loaded", "qc_is_perovskite_like"]:
        if c in pool.columns:
            pool = pool[pool[c] == True].copy()
    return pool


def dedup_pool(pool: pd.DataFrame, id_col: str, pbe_col: str) -> pd.DataFrame:
    df = pool.copy()
    df[id_col] = df[id_col].astype(str)

    qc1 = df["qc_structure_loaded"].astype(int) if "qc_structure_loaded" in df.columns else 0
    qc2 = df["qc_is_perovskite_like"].astype(int) if "qc_is_perovskite_like" in df.columns else 0
    pbe = pd.to_numeric(df[pbe_col], errors="coerce") if pbe_col in df.columns else pd.Series([np.nan] * len(df))
    pbe_pos = (pbe.fillna(0.0) > 0.0).astype(int)

    df["_qc_score"] = qc1 + qc2
    df["_pbe_pos"] = pbe_pos
    df["_pbe_val"] = pbe.fillna(-1e9)

    df = df.sort_values(by=["_qc_score", "_pbe_pos", "_pbe_val"], ascending=[False, False, False])
    df = df.drop_duplicates(subset=[id_col], keep="first").copy()
    df = df.drop(columns=["_qc_score", "_pbe_pos", "_pbe_val"], errors="ignore")
    df = df.reset_index(drop=True)
    return df


def infer_seen_ids(outdir: Path, id_col: str, seed_ids: pd.Series) -> Set[str]:
    seen = set([str(x) for x in seed_ids.astype(str).dropna().unique().tolist()])

    st = load_state(outdir / "state.json")
    for x in st.get("seen_ids", []):
        seen.add(str(x))

    sel_dir = outdir / "selections"
    if sel_dir.exists():
        for p in sorted(sel_dir.glob("selected_cycle*.csv")):
            try:
                df = pd.read_csv(p)
                if id_col in df.columns:
                    seen |= set(df[id_col].astype(str).tolist())
            except Exception:
                pass
    return seen


# ----------------------------- GPR helpers -----------------------------

def build_kernel(n_features: int, kind: str = "matern"):
    if n_features <= 0:
        n_features = 1
    kind = (kind or "matern").lower()
    if kind == "rbf":
        return (
            C(1.0, (1e-3, 1e3))
            * RBF(length_scale=np.ones(n_features), length_scale_bounds=(1e-3, 1e3))
            + WhiteKernel(1e-6)
        )
    return (
        C(1.0, (1e-3, 1e3))
        * Matern(length_scale=np.ones(n_features), nu=2.5, length_scale_bounds=(1e-3, 1e3))
        + WhiteKernel(1e-6)
    )


def qbc_mad_uncertainty(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_cand: np.ndarray,
    kernel_kind: str,
    alpha: float,
    restarts: int,
    committee_size: int,
    random_state: int = 42,
    return_preds: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    rng = np.random.RandomState(random_state)

    base = GaussianProcessRegressor(
        kernel=build_kernel(X_train.shape[1], kind=kernel_kind),
        alpha=alpha,
        n_restarts_optimizer=restarts,
        normalize_y=True,
        random_state=42,
    )
    base.fit(X_train, y_train)
    kernel_fixed = base.kernel_

    M = max(3, int(committee_size))
    preds = np.empty((M, X_cand.shape[0]), dtype=float)

    n = X_train.shape[0]
    for m in range(M):
        idx = rng.randint(0, n, size=n)
        g = GaussianProcessRegressor(
            kernel=kernel_fixed,
            alpha=alpha,
            optimizer=None,
            normalize_y=True,
            random_state=42 + m,
        )
        g.fit(X_train[idx], y_train[idx])
        preds[m] = g.predict(X_cand)

    med = np.median(preds, axis=0)
    mad = np.median(np.abs(preds - med[None, :]), axis=0)

    if return_preds:
        return mad, preds
    return mad


def extract_ard_lengthscales(
    gpr: GaussianProcessRegressor,
    feature_names: List[str],
    tol: float = 1e-8,
) -> Dict[str, object]:
    out: Dict[str, object] = {"feature_names": feature_names, "lengthscales": [], "upper_bound": None, "hit_upper": []}
    try:
        k = getattr(gpr, "kernel_", None)
        if k is None:
            return out

        base = None
        if hasattr(k, "k1") and hasattr(k.k1, "k2") and hasattr(k.k1.k2, "length_scale"):
            base = k.k1.k2
        elif hasattr(k, "k2") and hasattr(k.k2, "length_scale"):
            base = k.k2

        if base is None:
            return out

        ls = np.asarray(getattr(base, "length_scale"), dtype=float).ravel()
        b = getattr(base, "length_scale_bounds", (None, None))
        if isinstance(b, tuple) and len(b) == 2:
            ub = float(b[1])
        else:
            ub = float(np.asarray(b, dtype=float).ravel()[-1])

        hit = (ls >= (ub - tol)).astype(bool)

        out["lengthscales"] = ls.tolist()
        out["upper_bound"] = ub
        out["hit_upper"] = hit.tolist()
        return out
    except Exception:
        return out


# ----------------------------- acquisition helpers -----------------------------

def knn_density(X_pool: np.ndarray, k: int, eps: float = 1e-12) -> np.ndarray:
    k = max(1, int(k))
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto")
    nn.fit(X_pool)
    d, _ = nn.kneighbors(X_pool, return_distance=True)
    dk = d[:, k]
    return 1.0 / (dk + eps)


def anti_isolation_rho(
    X_cand: np.ndarray,
    X_labeled_train: np.ndarray,
    tau_quantile: float = 0.75,
    cutoff_quantile: float = 0.95,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    if X_labeled_train.shape[0] < 1:
        delta = np.full(X_cand.shape[0], 0.0, dtype=float)
        return np.ones_like(delta), delta, 1.0, 0.0

    nn = NearestNeighbors(n_neighbors=1, algorithm="auto")
    nn.fit(X_labeled_train)
    d, _ = nn.kneighbors(X_cand, return_distance=True)
    delta = d[:, 0].astype(float)

    tau = float(np.quantile(delta, float(tau_quantile))) if np.isfinite(delta).any() else 1.0
    if tau <= 0:
        tau = 1.0

    rho = np.exp(-(delta ** 2) / (tau ** 2 + eps))

    cutoff = float(np.quantile(delta, float(cutoff_quantile))) if np.isfinite(delta).any() else float("inf")
    if np.isfinite(cutoff):
        rho = np.where(delta > cutoff, 0.0, rho)

    return rho, delta, tau, cutoff


def fps_with_duplicate_rejection(
    X_short: np.ndarray,
    scores_short: np.ndarray,
    k: int,
    dup_quantile: float = 0.01,
    score_weight: float = 0.0,
) -> List[int]:
    k = max(1, int(k))
    n = X_short.shape[0]
    if n == 0:
        return []

    if n >= 3:
        nn = NearestNeighbors(n_neighbors=2, algorithm="auto").fit(X_short)
        d, _ = nn.kneighbors(X_short, return_distance=True)
        nn_dist = d[:, 1]
        eta = float(np.quantile(nn_dist, float(dup_quantile)))
    else:
        eta = 0.0

    # Normalize scores to [0,1]
    s = np.asarray(scores_short, dtype=float).copy()
    s[~np.isfinite(s)] = -1e9
    smin = float(np.min(s)) if np.isfinite(s).any() else 0.0
    smax = float(np.max(s)) if np.isfinite(s).any() else 1.0
    denom = (smax - smin) if (smax > smin) else 1.0
    s_norm = (s - smin) / denom

    score_weight = float(np.clip(score_weight, 0.0, 1.0))

    # deterministic jitter for tie-breaks
    rng = np.random.RandomState(0)

    # --- START SELECTION ---
    if score_weight <= 1e-12:
        mu = np.mean(X_short, axis=0, keepdims=True)
        start = int(np.argmax(np.linalg.norm(X_short - mu, axis=1)))
    else:
        start = int(np.argmax(s))

    chosen = [start]

    min_dist = np.linalg.norm(X_short - X_short[start:start + 1], axis=1)
    min_dist[start] = -np.inf

    while len(chosen) < k:
        d = min_dist.copy()
        finite = np.isfinite(d)
        if not np.any(finite):
            break

        dmin = float(np.min(d[finite]))
        dmax = float(np.max(d[finite]))
        dden = (dmax - dmin) if (dmax > dmin) else 1.0

        d_norm = np.full_like(d, -np.inf, dtype=float)
        d_norm[finite] = (d[finite] - dmin) / dden

        util = (1.0 - score_weight) * d_norm + score_weight * s_norm

        # --- TIE-BREAK: if distances are effectively tied, add tiny jitter ---
        if (dmax - dmin) <= 1e-12:
            util = util + 1e-12 * rng.rand(util.size)

        for c in chosen:
            util[c] = -np.inf

        j = int(np.argmax(util))
        if not np.isfinite(util[j]):
            break

        if min_dist[j] < eta:
            min_dist[j] = -np.inf
            continue

        chosen.append(j)
        dnew = np.linalg.norm(X_short - X_short[j:j + 1], axis=1)
        min_dist = np.minimum(min_dist, dnew)
        for c in chosen:
            min_dist[c] = -np.inf

    return chosen



def pairwise_distance_summary(X: np.ndarray) -> Dict[str, float]:
    n = int(X.shape[0])
    if n <= 1:
        return {"n": n, "min_dist": float("nan"), "median_dist": float("nan"), "mean_dist": float("nan")}
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            dists.append(float(np.linalg.norm(X[i] - X[j])))
    arr = np.asarray(dists, dtype=float)
    return {
        "n": n,
        "min_dist": float(np.min(arr)) if len(arr) else float("nan"),
        "median_dist": float(np.median(arr)) if len(arr) else float("nan"),
        "mean_dist": float(np.mean(arr)) if len(arr) else float("nan"),
    }


# ----------------------------- feature governance -----------------------------

def _looks_like_qc(c: str) -> bool:
    return c.lower().startswith("qc_")


def _looks_like_missing(c: str) -> bool:
    cl = c.lower()
    return ("missing" in cl) or cl.endswith("_missing") or cl.startswith("missing_")


def _looks_like_bool(c: str) -> bool:
    cl = c.lower()
    return cl.startswith("is_") or cl.startswith("has_") or cl.endswith("_flag") or cl.endswith("_bool")


def _is_forbidden_feature_name(c: str) -> bool:
    cl = c.lower()
    if cl in {"hse_gap", "delta_gap", "band_gap_hse_calc", "band_gap", "band_gap_pbe_calc"}:
        return True
    return any(x in cl for x in FORBIDDEN_FEATURE_SUBSTRINGS)


def _family_heuristic(c: str) -> str:
    cl = c.lower()

    if cl in {"is_metal", "is_gap0"}:
        return "F1b"

    f1_keys = ("pbe", "efermi", "fermi", "is_gap_direct", "direct_gap", "pbe_eff")
    if any(k in cl for k in f1_keys):
        return "F1"

    f2_keys = ("lattice", "volume", "density", "bond", "angle", "tilt", "bav", "bob", "cn_", "coord", "octa", "perov")
    if any(k in cl for k in f2_keys):
        return "F2"

    f3_keys = ("mende", "electroneg", "en_", "ea", "ie", "ioniz", "hardness", "mass", "atomic", "radius")
    if any(k in cl for k in f3_keys):
        return "F3"

    return "F3"


def load_feature_families_json(path: Optional[str]) -> Dict[str, List[str]]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    obj = json.loads(p.read_text())
    out: Dict[str, List[str]] = {}
    for k in ["F1", "F1b", "F2", "F3", "EXCLUDE"]:
        v = obj.get(k, obj.get(k.lower(), []))
        if isinstance(v, list):
            out[k] = [str(x) for x in v]
    return out


def select_feature_columns(
    pool_cols: List[str],
    id_col: str,
    meta_cols: List[str],
    split_col: str,
    feature_set: str,
    blacklist: Set[str],
    include_qc: bool,
    include_missing: bool,
    include_shortcuts: bool,
    families_json: Dict[str, List[str]],
) -> Tuple[List[str], List[str], Dict[str, List[str]]]:
    cols = [c for c in pool_cols if c != id_col]

    fam: Dict[str, List[str]] = {"F1": [], "F1b": [], "F2": [], "F3": [], "EXCLUDE": []}

    def _add(f: str, c: str) -> None:
        fam.setdefault(f, []).append(c)

    # Explicit families override
    explicit = bool(families_json)
    if explicit:
        explicit_set = set()
        for k, lst in families_json.items():
            for c in lst:
                if c in cols:
                    _add(k.upper(), c)
                    explicit_set.add(c)
        for c in cols:
            if c in explicit_set:
                continue
            _add(_family_heuristic(c), c)
    else:
        for c in cols:
            _add(_family_heuristic(c), c)

    # Exclusions (governance + blacklist)
    blocked: Set[str] = set(meta_cols) | {id_col, split_col}
    for c in cols:
        if c in blocked or c in blacklist or _is_forbidden_feature_name(c):
            fam["EXCLUDE"].append(c)

    def _keep(c: str) -> bool:
        if c in fam["EXCLUDE"]:
            return False
        if _looks_like_qc(c) and not include_qc:
            return False
        if _looks_like_missing(c) and not include_missing:
            return False
        if c.lower() in {"is_metal", "is_gap0"} and not include_shortcuts:
            return False
        return True

    # Feature-set selection
    fs = feature_set.strip().upper()
    if fs in {"F2+F3", "F2F3"}:
        selected = [c for c in (fam["F2"] + fam["F3"]) if _keep(c)]
    elif fs in {"F1"}:
        selected = [c for c in fam["F1"] if _keep(c)]
        if include_shortcuts:
            selected += [c for c in fam["F1b"] if _keep(c)]
    elif fs in {"F1+F2+F3", "F123"}:
        selected = [c for c in (fam["F1"] + fam["F2"] + fam["F3"]) if _keep(c)]
        if include_shortcuts:
            selected += [c for c in fam["F1b"] if _keep(c)]
    elif fs in {"AUTO"}:
        selected = [c for c in cols if _keep(c)]
    else:
        raise ValueError(f"Unknown feature-set '{feature_set}'. Use F2F3, F1, F123, or AUTO.")

    # Split into numeric/bool columns (bools are not scaled)
    bool_cols = [c for c in selected if _looks_like_bool(c)]
    num_cols = [c for c in selected if c not in bool_cols]

    # Stable unique
    def _uniq(xs: List[str]) -> List[str]:
        seen: Set[str] = set()
        out: List[str] = []
        for x in xs:
            if x not in seen:
                out.append(x)
                seen.add(x)
        return out

    return _uniq(num_cols), _uniq(bool_cols), fam


def preprocess_fit(X_seed: pd.DataFrame, num_cols: List[str], bool_cols: List[str]) -> Dict[str, object]:
    scaler = StandardScaler()
    if num_cols:
        scaler.fit(X_seed[num_cols].values)
    return {"scaler": scaler, "num_cols": num_cols, "bool_cols": bool_cols}


def preprocess_transform(X: pd.DataFrame, artifacts: Dict[str, object]) -> np.ndarray:
    scaler = artifacts["scaler"]
    num_cols = artifacts["num_cols"]
    bool_cols = artifacts["bool_cols"]

    X_num = scaler.transform(X[num_cols].values) if num_cols else np.zeros((len(X), 0))
    X_bool = X[bool_cols].values.astype(float) if bool_cols else np.zeros((len(X), 0))
    return np.hstack([X_num, X_bool])


def _num_frame(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    if not cols:
        return pd.DataFrame(index=df.index)
    out = df[cols].apply(pd.to_numeric, errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)   # <-- ADD THIS
    return out


def _bool_frame(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    if not cols:
        return pd.DataFrame(index=df.index)
    out = df[cols].copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(int)
    return out


def fit_transform_block(
    seed_train: pd.DataFrame,
    seed_val: pd.DataFrame,
    pool_trainval: pd.DataFrame,
    pool_cand: pd.DataFrame,
    num_cols: List[str],
    bool_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, object], Dict[str, float]]:
    Xtr_num = _num_frame(seed_train, num_cols)
    Xva_num = _num_frame(seed_val, num_cols)
    Xp_num = _num_frame(pool_cand, num_cols)

    fill_values: Dict[str, float] = {}

    if num_cols:
        Xpool_num_full = _num_frame(pool_trainval, num_cols)

        all_nan_num = [
            c for c in num_cols
            if pd.concat([Xtr_num[c], Xpool_num_full[c]], ignore_index=True).isna().all()
        ]
        if all_nan_num:
            log(f"[features] dropped all-NaN numeric cols ({len(all_nan_num)}): {all_nan_num}")
            num_cols = [c for c in num_cols if c not in all_nan_num]
            Xtr_num = Xtr_num[num_cols]
            Xva_num = Xva_num[num_cols] if len(seed_val) else Xva_num
            Xp_num = Xp_num[num_cols]
            Xpool_num_full = Xpool_num_full[num_cols]

        med_pool = Xpool_num_full.median(numeric_only=True)
        med_comb = pd.concat([Xtr_num, Xpool_num_full], ignore_index=True).median(numeric_only=True)

        for c in num_cols:
            fill = med_pool.get(c, np.nan)
            if pd.isna(fill):
                fill = med_comb.get(c, np.nan)
            if pd.isna(fill):
                fill = 0.0
            fill = float(fill)
            fill_values[c] = fill
            Xtr_num[c] = Xtr_num[c].fillna(fill)
            if len(seed_val):
                Xva_num[c] = Xva_num[c].fillna(fill)
            Xp_num[c] = Xp_num[c].fillna(fill)

    Xtr_bool = _bool_frame(seed_train, bool_cols)
    Xva_bool = _bool_frame(seed_val, bool_cols)
    Xp_bool = _bool_frame(pool_cand, bool_cols)

    Xtr = pd.concat([Xtr_num, Xtr_bool], axis=1)
    Xva = pd.concat([Xva_num, Xva_bool], axis=1) if len(seed_val) else pd.DataFrame(columns=Xtr.columns)
    Xp = pd.concat([Xp_num, Xp_bool], axis=1)

    artifacts = preprocess_fit(Xtr, num_cols, bool_cols)
    Xs_train = preprocess_transform(Xtr, artifacts)
    Xs_val = preprocess_transform(Xva, artifacts) if len(seed_val) else np.zeros((0, Xs_train.shape[1]))
    Xs_pool = preprocess_transform(Xp, artifacts)

    return Xs_train, Xs_val, Xs_pool, Xs_train, artifacts, fill_values


# ----------------------------- metrics I/O -----------------------------

def save_metrics_csv(outdir: Path, cycle: int, row: Dict[str, float]) -> None:
    safe_mkdir(outdir / "metrics")
    r = {"cycle": int(cycle), **row}
    mf = outdir / "metrics" / "metrics.csv"
    if mf.exists():
        df = pd.read_csv(mf)
        df = df[df["cycle"] != int(cycle)].copy()
        df = pd.concat([df, pd.DataFrame([r])], ignore_index=True)
    else:
        df = pd.DataFrame([r])
    df = df.sort_values("cycle")
    df.to_csv(mf, index=False)


def write_selection(outdir: Path, cycle: int, sel: pd.DataFrame) -> None:
    safe_mkdir(outdir / "selections")
    sel.to_csv(outdir / "selections" / f"selected_cycle{int(cycle)}.csv", index=False)


# ----------------------------- core command: select -----------------------------

def cmd_select(args) -> None:
    outdir = Path(args.outdir)
    safe_mkdir(outdir)
    # Optional: runtime PBE overrides (DFT truth ONLY)
    # Priority (truth): seed pbe_gap (parsed from DFT outputs) > labels_store pbe_gap (DFT cache) > Output/ parsed on the fly > missing (NaN)
    pbe_override: Dict[str, float] = {}
    pbe_source: Dict[str, str] = {}
    
    store_path = outdir / "labels_store.csv"
    store_df = None
    
    def _norm_id(x) -> str:
        return str(x).strip()
    
    def _parse_pbe_from_log(fp: Path):
        """
        the DFT workflow's band_GGA/Band_gap.log patterns:
          - "This system is metallic." => 0.0
          - "Band gap: X eV" => X
        """
        try:
            txt = fp.read_text(errors="ignore")
        except Exception:
            return (np.nan, "")
        low = txt.lower()
        if "metallic" in low:
            return (0.0, f"pbe:{fp.as_posix()}")
        m = re.search(r"Band gap:\s*([0-9]*\.?[0-9]+)\s*eV", txt)
        if m:
            return (float(m.group(1)), f"pbe:{fp.as_posix()}")
        return (np.nan, "")
    
    # --- 1) Load cached DFT-PBE overrides if present (optional cache) ---
    if store_path.exists():
        try:
            store_df = pd.read_csv(store_path)
            if args.id_col in store_df.columns and args.pool_pbe_col in store_df.columns:
                tmp = store_df[[args.id_col, args.pool_pbe_col]].copy()
                tmp[args.id_col] = tmp[args.id_col].astype(str).map(_norm_id)
                tmp[args.pool_pbe_col] = pd.to_numeric(tmp[args.pool_pbe_col], errors="coerce")
                tmp = tmp[np.isfinite(tmp[args.pool_pbe_col])].drop_duplicates(subset=[args.id_col], keep="last")
                pbe_override.update(dict(zip(tmp[args.id_col].astype(str), tmp[args.pool_pbe_col].astype(float))))
                if "pbe_source" in store_df.columns:
                    t2 = store_df[[args.id_col, "pbe_source"]].copy()
                    t2[args.id_col] = t2[args.id_col].astype(str).map(_norm_id)
                    t2["pbe_source"] = t2["pbe_source"].fillna("").astype(str)
                    pbe_source.update(dict(zip(t2[args.id_col].astype(str), t2["pbe_source"])))
                log(f"[pbe] loaded DFT PBE overrides from labels_store.csv: {len(pbe_override)} ids")
        except Exception as e:
            log(f"[pbe] WARNING: failed to load pbe overrides from {store_path}: {e!r}")
    
    # --- 2) Load pool + seed ---
    pool = read_csv(Path(args.pool))
    if args.safe_mode_filter:
        pool = apply_safe_mode_filter(pool)
    
    seed_lab = read_csv(Path(args.seed))
    
    pool[args.id_col] = pool[args.id_col].astype(str).map(_norm_id)
    seed_lab[args.id_col] = seed_lab[args.id_col].astype(str).map(_norm_id)
    
    # Seed must contain id, hse, label_status (Phase-2)
    for req in [args.id_col, args.hse_col, args.status_col]:
        if req not in seed_lab.columns:
            if req == args.status_col and store_path.exists():
                # fallback: use labels_store to supply status
                try:
                    st = store_df if store_df is not None else pd.read_csv(store_path)
                    if args.id_col in st.columns and args.status_col in st.columns:
                        tmp = st[[args.id_col, args.status_col]].copy()
                        tmp[args.id_col] = tmp[args.id_col].astype(str).map(_norm_id)
                        seed_lab[args.id_col] = seed_lab[args.id_col].astype(str).map(_norm_id)
                        seed_lab = seed_lab.merge(tmp, on=args.id_col, how="left", suffixes=("", "_store"))
                        if args.status_col not in seed_lab.columns and f"{args.status_col}_store" in seed_lab.columns:
                            seed_lab[args.status_col] = seed_lab[f"{args.status_col}_store"]
                        log("[seed] label_status sourced from labels_store.csv")
                        continue
                except Exception:
                    pass
            raise KeyError(
                f"Seed must contain '{args.id_col}', '{args.hse_col}', and '{args.status_col}' for Phase-2."
            )
    
    # --- 3) Merge seed PBE (DFT truth) into overrides (SEED WINS) ---
    if args.pool_pbe_col in seed_lab.columns:
        try:
            tmp = seed_lab[[args.id_col, args.pool_pbe_col]].copy()
            tmp[args.id_col] = tmp[args.id_col].astype(str).map(_norm_id)
            tmp[args.pool_pbe_col] = pd.to_numeric(tmp[args.pool_pbe_col], errors="coerce")
            tmp = tmp[np.isfinite(tmp[args.pool_pbe_col])]
            for sid, val in zip(tmp[args.id_col].astype(str), tmp[args.pool_pbe_col].astype(float)):
                pbe_override[sid] = float(val)  # seed wins
                pbe_source.setdefault(sid, "seed:pbe_gap")
            log(f"[pbe] merged seed PBE into overrides (seed wins): now {len(pbe_override)} ids")
        except Exception as e:
            log(f"[pbe] WARNING: failed to merge seed PBE overrides: {e!r}")
    
    # --- 4) Fresh-run support: auto-parse Output/ for missing pool IDs (no manual scripts) ---
    # This is what makes "delete run_v3 and start fresh" safe again.
    output_root = Path("Output")
    need_ids = set(pool[args.id_col].astype(str).tolist()) - set(pbe_override.keys())
    
    # choose best directory name column (if present)
    dir_col = None
    for c in ["dir_name", "bank_file"]:
        if c in pool.columns and pool[c].fillna("").astype(str).str.strip().ne("").any():
            dir_col = c
            break
    
    pool_lookup = pool.set_index(args.id_col)
    parsed = 0
    missed = 0
    new_rows = []
    
    if len(need_ids) > 0:
        for sid in sorted(list(need_ids)):
            dname = sid
            if dir_col is not None and sid in pool_lookup.index:
                v = str(pool_lookup.loc[sid, dir_col]).strip()
                if v and v.lower() != "nan":
                    dname = v
    
            fp = output_root / dname / "band_GGA" / "Band_gap.log"
            if not fp.exists():
                missed += 1
                continue
    
            val, src = _parse_pbe_from_log(fp)
            if np.isfinite(val):
                pbe_override[sid] = float(val)
                pbe_source[sid] = src
                parsed += 1
                new_rows.append({
                    args.id_col: sid,
                    args.pool_pbe_col: float(val),
                    "pbe_source": src,
                    # keep these columns so update/upsert never breaks if you reuse labels_store later
                    args.hse_col: np.nan,
                    args.status_col: "",
                    "fail_reason": "",
                    "attempt_count": 0,
                    "label_source": "from_output_pbe",
                })
            else:
                missed += 1
    
        log(f"[pbe] auto-parsed Output/ PBE: added={parsed} missed={missed} overrides_total={len(pbe_override)}")
    
        # cache (optional but recommended): recreates run_v3/labels_store.csv after you delete run_v3/
        if parsed > 0:
            safe_mkdir(outdir)
            df_new = pd.DataFrame(new_rows)
            if store_path.exists():
                try:
                    df_old = pd.read_csv(store_path)
                    df_all = pd.concat([df_old, df_new], ignore_index=True)
                except Exception:
                    df_all = df_new
            else:
                df_all = df_new
    
            df_all[args.id_col] = df_all[args.id_col].astype(str).map(_norm_id)
            df_all[args.pool_pbe_col] = pd.to_numeric(df_all[args.pool_pbe_col], errors="coerce")
            df_all = df_all[np.isfinite(df_all[args.pool_pbe_col])].drop_duplicates(subset=[args.id_col], keep="last")
            df_all.to_csv(store_path, index=False)
            log(f"[pbe] cached overrides -> {store_path} rows={len(df_all)}")
    
    log("[pbe] PBE policy: labeled truth uses DFT (seed/labels_store/Output); candidates fall back to pool PBE if DFT missing.")
    
    def pbe_truth(df: pd.DataFrame) -> pd.Series:
        """
        For labeled rows (seed_train/seed_val): DFT truth only.
        Prefer df[pbe_gap] if it is already DFT-filled, else override map, else NaN.
        NEVER fall back to pool/MP.
        """
        base = pd.to_numeric(df.get(args.pool_pbe_col, np.nan), errors="coerce")
        ov = df[args.id_col].astype(str).map(_norm_id).map(pbe_override)
        ov = pd.to_numeric(ov, errors="coerce")
        return base.where(np.isfinite(base), ov)
    
    def pbe_baseline(df: pd.DataFrame) -> pd.Series:
        """
        For unlabeled candidates: use DFT if known, else fall back to pool/MP pbe_gap.
        """
        base = pd.to_numeric(df.get(args.pool_pbe_col, np.nan), errors="coerce")
        ov = df[args.id_col].astype(str).map(_norm_id).map(pbe_override)
        ov = pd.to_numeric(ov, errors="coerce")
        return ov.where(np.isfinite(ov), base)

    
    split_col = args.split_col
    has_split = split_col in pool.columns

    
    # Keep this check for schema sanity/dedup (even though pool pbe is not trusted for truth)
    if args.pool_pbe_col not in pool.columns:
        raise KeyError(
            f"Pool missing '{args.pool_pbe_col}'. Needed for pool schema/dedup, "
            "even though DFT-only PBE truth comes from seed/labels_store."
        )
    
    pool = dedup_pool(pool, args.id_col, args.pool_pbe_col)
    
    if has_split:
        pool_trainval = pool[pool[split_col].astype(str).isin(["train", "val"])].copy()
        pool_cand = pool[pool[split_col].astype(str).eq("train")].copy()
    else:
        pool_trainval = pool.copy()
        pool_cand = pool.copy()
    
    pool_trainval = pool_trainval.reset_index(drop=True)
    pool_cand = pool_cand.reset_index(drop=True)

    # report coverage only (do NOT drop)
    cand_dft = pd.to_numeric(pool_cand[args.id_col].astype(str).map(_norm_id).map(pbe_override), errors="coerce")
    cand_pool = pd.to_numeric(pool_cand.get(args.pool_pbe_col, np.nan), errors="coerce")
    log(f"[pbe] cand coverage: n={len(pool_cand)}  dft_known={int(np.isfinite(cand_dft).sum())}  pool_pbe_finite={int(np.isfinite(cand_pool).sum())}")

    # Join labels onto pool (train+val only)
    # IMPORTANT (DFT-only truth):
    #   - Pool pbe_gap is NOT trusted as truth.
    #   - Keep seed pbe_gap (from DFT outputs) if present, otherwise leave NaN here.
    #   - Truth PBE used downstream is always seed_pbe = pbe_effective(...).
    
    seed_left_cols = [args.id_col, args.hse_col, args.status_col]
    if args.pool_pbe_col in seed_lab.columns:
        seed_left_cols.append(args.pool_pbe_col)
    
    seed_left = seed_lab[seed_left_cols].copy()
    seed_left[args.id_col] = seed_left[args.id_col].astype(str).str.strip()
    seed_left[args.hse_col] = pd.to_numeric(seed_left[args.hse_col], errors="coerce")
    seed_left[args.status_col] = (
        seed_left[args.status_col]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
        .replace({"NAN": ""})
    )
    
    if args.pool_pbe_col in seed_left.columns:
        seed_left[args.pool_pbe_col] = pd.to_numeric(seed_left[args.pool_pbe_col], errors="coerce")
    
    # Merge seed labels into pool features (pool contributes features + its pbe_gap column)
    seed = pd.merge(
        seed_left,
        pool_trainval,
        on=args.id_col,
        how="inner",
        validate="many_to_one",
        suffixes=("_seed", ""),  # seed pbe -> pbe_gap_seed if pool also has pbe_gap
    )
    
    if len(seed) == 0:
        raise RuntimeError(
            "After joining seed labels to pool by id, seed is empty. "
            "Your seed IDs must exist in the pool CSV you are using."
        )
    
    # ---- DFT-only fuse rule for the *seed dataframe* ----
    # If both exist:
    #   - keep pool pbe ONLY for mismatch warnings/debug
    #   - set seed[pbe_gap] = seed_pbe when finite, else NaN (NO fallback to pool)
    seed_pbe_seed_col = f"{args.pool_pbe_col}_seed"
    
    # pool pbe exists in merged seed (from pool_trainval)
    seed_pool_pbe = pd.to_numeric(seed.get(args.pool_pbe_col, np.nan), errors="coerce")
    
    if seed_pbe_seed_col in seed.columns:
        seed_seed_pbe = pd.to_numeric(seed[seed_pbe_seed_col], errors="coerce")
    
        # Warn if seed and pool disagree where both are finite
        both = np.isfinite(seed_seed_pbe.values) & np.isfinite(seed_pool_pbe.values)
        if both.any():
            diff = np.abs(seed_seed_pbe.values - seed_pool_pbe.values)
            n_bad = int(np.sum(both & (diff > 1e-6)))
            if n_bad > 0:
                try:
                    ex = seed.loc[both & (diff > 1e-6), [args.id_col, seed_pbe_seed_col, args.pool_pbe_col]].head(5)
                    log(
                        f"[seed] WARNING: {n_bad} rows have seed vs pool PBE mismatch; "
                        f"DFT-only truth uses SEED/Output values. Examples:\n{ex.to_string(index=False)}"
                    )
                except Exception:
                    log(f"[seed] WARNING: {n_bad} rows have seed vs pool PBE mismatch; DFT-only truth uses SEED/Output values.")
    
        # DFT-only: keep seed pbe if present, otherwise NaN here
        seed[args.pool_pbe_col] = seed_seed_pbe.where(np.isfinite(seed_seed_pbe), np.nan)
    
        # drop the suffixed seed column after fusing
        seed = seed.drop(columns=[seed_pbe_seed_col])
    else:
        # no seed-side PBE provided -> do NOT use pool PBE as truth
        seed[args.pool_pbe_col] = np.nan
    
    # Compute effective PBE + targets (DFT-only truth: seed/labels_store/Output overrides)
    # Compute effective PBE + targets (TRUTH policy: DFT-only via seed/overrides; no MP fallback)
    seed_pbe = pd.to_numeric(pbe_truth(seed), errors="coerce")
    
    # Sanity: how much usable PBE do we have after merge + overrides?
    _p = pd.to_numeric(seed_pbe, errors="coerce").to_numpy(dtype=float)
    _p0 = np.isfinite(_p) & (_p <= float(args.pbe_zero_thr))
    log(
        f"[seed] merged_seed n={len(seed)} finite_pbe={int(np.isfinite(_p).sum())} "
        f"pbe0={int(_p0.sum())} pbe_zero_thr={float(args.pbe_zero_thr):.3f}"
    )
    
    seed_hse = pd.to_numeric(seed[args.hse_col], errors="coerce")
    y_delta_all = (seed_hse - seed_pbe).values

    # Target-mode ablation for Gap 3:
    #   delta      : Stage-B GPR learns HSE - PBE_eff, then reconstructs HSE = PBE_eff + target_pred.
    #   direct_hse : Stage-B GPR learns HSE directly, then target_pred is already HSE.
    # All routing, features, splits, and acquisition machinery are otherwise kept fixed.
    target_mode = str(getattr(args, "target_mode", "delta")).strip().lower()
    if target_mode not in {"delta", "direct_hse"}:
        raise ValueError(f"Unknown --target-mode {target_mode!r}; use 'delta' or 'direct_hse'.")
    log(f"[target] Stage-B regression target mode = {target_mode}")




    # Phase-2 metal-like truth label
    metal_like_truth = compute_metal_like_hse(
        hse=seed_hse.values,
        status=seed[args.status_col],
        thr_ev=args.metal_thr_ev,
    )

    # Keep finite Δ rows for regression; classifier can use rows with status-defined label even if HSE NaN?
    # Here we only treat rows as usable if metal-like truth can be evaluated.
    # For METAL_ASSUMED with NaN HSE, metal_like_truth is still True.
    seed = seed.copy()
    seed["_pbe_eff"] = seed_pbe.values
    seed["_hse"] = seed_hse.values
    seed["_delta"] = y_delta_all
    seed["_metal_like_truth"] = metal_like_truth.astype(int)

    
    # ---------------- Fixed VAL split (v3 gate protocol) ----------------
    val_ids = load_or_create_fixed_val_ids(
        outdir=outdir,
        seed=seed,
        id_col=args.id_col,
        status_col=args.status_col,
        hse_col=args.hse_col,
        metal_thr_ev=float(args.metal_thr_ev),
        cycle=int(args.cycle),
        val_size=30,
        salt="val_v3",
    )
    val_ids_set = set([str(s).strip() for s in val_ids])

    seed_id_series = seed[args.id_col].astype(str).str.strip()
    va_mask = seed_id_series.isin(val_ids_set).values
    tr_mask = ~va_mask

    seed_train = seed.loc[tr_mask].copy()
    seed_val = seed.loc[va_mask].copy()

    if len(seed_val) < 3:
        log(f"[val] WARNING: only {len(seed_val)} VAL rows present after merge; metrics may be noisy.")
    if len(seed_train) < 3:
        log(f"[train] WARNING: only {len(seed_train)} TRAIN rows present after VAL split; training may fallback.")

    meta_cols = [args.id_col, "formula", "source", args.pool_pbe_col]  # add pbe_gap here
    if has_split:
        meta_cols.append(split_col)
    # Feature governance
    blacklist_cli = parse_blacklist(args.feature_blacklist or "")
    blacklist = set(DEFAULT_SELECT_FEATURE_BLACKLIST) | set(blacklist_cli)
    # ---- HARD EXCLUDE PBE columns as FEATURES ----
    # Truth PBE is handled via pbe_effective() and used only to form Δ and reconstruct HSE.
    # Pool pbe_gap can be MP-derived and may mismatch DFT-PBE; do not let it enter the models.
    blacklist.add(str(args.pool_pbe_col).strip())              # "pbe_gap"
    blacklist.add(f"{str(args.pool_pbe_col).strip()}_seed")    # if it ever exists
    blacklist.add("_pbe_eff")                                  # cached truth PBE in seed (never a feature)
    blacklist.add("pbe_gap_effective")
    blacklist.add("_pbe_gap_pool_orig")
    blacklist.add("pbe_source")
    families_json = load_feature_families_json(args.feature_families_json)
    # Extra hard blacklist (control/leakage columns)
    # - split + split_* must never be features
    if split_col:
        blacklist.add(str(split_col).strip())
    blacklist.add("split")

    # Stage-A feature set
    num_cols_A, bool_cols_A, famA = select_feature_columns(
        pool_cols=list(pool_trainval.columns),
        id_col=args.id_col,
        meta_cols=meta_cols,
        split_col=split_col,
        feature_set=args.clf_feature_set,
        blacklist=blacklist,
        include_qc=args.include_qc_flags,
        include_missing=args.include_missingness_flags,
        include_shortcuts=args.include_shortcut_flags,
        families_json=families_json,
    )

    # Stage-B feature set
    num_cols_B, bool_cols_B, famB = select_feature_columns(
        pool_cols=list(pool_trainval.columns),
        id_col=args.id_col,
        meta_cols=meta_cols,
        split_col=split_col,
        feature_set=args.reg_feature_set,
        blacklist=blacklist,
        include_qc=args.include_qc_flags,
        include_missing=args.include_missingness_flags,
        include_shortcuts=args.include_shortcut_flags,
        families_json=families_json,
    )
    def _assert_no_pbe(cols, name):
        bad = [c for c in cols if str(c).strip() in {
            str(args.pool_pbe_col).strip(),
            f"{str(args.pool_pbe_col).strip()}_seed",
            "_pbe_eff",
            "pbe_gap_effective",
        }]
        if bad:
            raise RuntimeError(f"[features] BUG: PBE leaked into {name}: {bad}")
    
    _assert_no_pbe(num_cols_A + bool_cols_A, "Stage-A features")
    _assert_no_pbe(num_cols_B + bool_cols_B, "Stage-B features")

    # ---- HARD DROP LEAKY/CONTROL COLUMNS (guarantee) ----
    def _drop_leaky(cols):
        out = []
        sc = str(split_col).strip().lower() if split_col else ""
        for c in cols:
            if c is None:
                continue
            lc = str(c).strip().lower()
            # drop exact split col + any split* columns (split_day0, split_*, etc.)
            if sc and lc == sc:
                continue
            if lc.startswith("split"):
                continue
            out.append(c)
        return out

    num_cols_A  = _drop_leaky(num_cols_A)
    bool_cols_A = _drop_leaky(bool_cols_A)
    num_cols_B  = _drop_leaky(num_cols_B)
    bool_cols_B = _drop_leaky(bool_cols_B)

    log(f"[features] A: num={len(num_cols_A)} bool={len(bool_cols_A)}   B: num={len(num_cols_B)} bool={len(bool_cols_B)}")
    if args.log_feature_families:
        dump_json(outdir / "metrics" / f"feature_families_cycle{int(args.cycle)}.json", {"A": famA, "B": famB})


    # ---------------- Metalness classifier (ML) ----------------
    # We train a classifier for metal-like under HSE on ALL labeled data (not PBE0-only),
    # then convert to p_gate = P(rescued nonmetal) and only apply it in PBE0 regime.
    pbe_zero_thr = float(args.pbe_zero_thr)
    gate_decision_thr = float(getattr(args, "gate_decision_thr", 0.5))


    # ---- Optional: merge DOS-derived PBE electronic features for the gate (Stage A only) ----
    # This is intentionally *read-only* and uses an existing CSV (no re-parsing Output/ every run).
    # It survives outdir deletes because the CSV is external to the run directory.
    dos_path = Path(getattr(args, "dos_csv", "dos_from_output.csv") or "")
    dos_df = None
    if str(dos_path).strip() != "" and dos_path.exists():
        try:
            dos_raw = read_csv(dos_path)
            # Keep only successfully parsed DOS rows
            if "parsed_ok" in dos_raw.columns:
                dos_raw["parsed_ok"] = pd.to_numeric(dos_raw["parsed_ok"], errors="coerce").fillna(0).astype(int)
                dos_raw = dos_raw.loc[dos_raw["parsed_ok"].values.astype(int) == 1].copy()
            if "has_doscar" in dos_raw.columns:
                dos_raw["has_doscar"] = pd.to_numeric(dos_raw["has_doscar"], errors="coerce").fillna(0).astype(int)
                dos_raw = dos_raw.loc[dos_raw["has_doscar"].values.astype(int) == 1].copy()
            # Minimal columns needed (avoid pulling paths into the run)
            keep = [args.id_col, "dosEf", "d_below", "d_above", "dist_ratio", "gap", "n_gap_pts", "maxDOS_between", "is_gap", "metal_like"]
            keep = [c for c in keep if c in dos_raw.columns]
            dos_df = dos_raw[keep].copy()
            dos_df[args.id_col] = dos_df[args.id_col].astype(str).str.strip()
            dos_df = dos_df.drop_duplicates(subset=[args.id_col], keep="last")
            log(f"[dos] loaded {len(dos_df)}/{len(dos_raw)} usable DOS rows from {dos_path}")
        except Exception as e:
            dos_df = None
            log(f"[dos] WARNING: failed to read/merge DOS CSV {dos_path}: {e!r}")

    def _merge_dos(df: pd.DataFrame) -> pd.DataFrame:
        if dos_df is None or len(df) == 0 or args.id_col not in df.columns:
            return df
        out = df.merge(dos_df, on=args.id_col, how="left", validate="many_to_one")
        return out

    if dos_df is not None:
        seed_train = _merge_dos(seed_train)
        seed_val   = _merge_dos(seed_val) if len(seed_val) else seed_val
        pool_cand  = _merge_dos(pool_cand)

        # Derived, numerically stable features (do NOT use absolute energies Ef/vbm/cbm as features)
        for _df in (seed_train, seed_val, pool_cand):
            if len(_df) == 0:
                continue
            dosEf = pd.to_numeric(_df.get("dosEf", np.nan), errors="coerce").astype(float)
            _df["_dos_log_dosEf"] = np.log1p(np.clip(dosEf.to_numpy(dtype=float), 0.0, None))

            maxdos = pd.to_numeric(_df.get("maxDOS_between", np.nan), errors="coerce").astype(float)
            _df["_dos_log_maxDOS_between"] = np.log1p(np.clip(maxdos.to_numpy(dtype=float), 0.0, None))

            gap_dos = pd.to_numeric(_df.get("gap", np.nan), errors="coerce").astype(float)
            _df["_dos_log_gap"] = np.log1p(np.clip(gap_dos.to_numpy(dtype=float), 0.0, None))

            dr = pd.to_numeric(_df.get("dist_ratio", np.nan), errors="coerce").astype(float).to_numpy(dtype=float)
            dr = np.clip(dr, 1e-6, 1e6)
            _df["_dos_log_dist_ratio"] = np.log(dr)
            _df["_dos_abs_log_dist_ratio"] = np.abs(_df["_dos_log_dist_ratio"].to_numpy(dtype=float))

            _df["_dos_d_below"] = pd.to_numeric(_df.get("d_below", np.nan), errors="coerce").astype(float)
            _df["_dos_d_above"] = pd.to_numeric(_df.get("d_above", np.nan), errors="coerce").astype(float)

            _df["_dos_is_gap"] = pd.to_numeric(_df.get("is_gap", 0), errors="coerce").fillna(0).astype(int)

        # Extend gate feature lists (Stage A only)
        dos_num_cols = ["_dos_log_dosEf", "_dos_log_maxDOS_between", "_dos_abs_log_dist_ratio", "_dos_log_gap", "_dos_d_below", "_dos_d_above"]
        dos_bool_cols = ["_dos_is_gap"]
        num_cols_A = list(dict.fromkeys(list(num_cols_A) + [c for c in dos_num_cols if c not in blacklist]))
        bool_cols_A = list(dict.fromkeys(list(bool_cols_A) + [c for c in dos_bool_cols if c not in blacklist]))
        log(f"[dos] gate feature add: +{len(dos_num_cols)} num +{len(dos_bool_cols)} bool (now A_num={len(num_cols_A)} A_bool={len(bool_cols_A)})")

    # Prepare matrices for the classifier (reuse "A" feature block)
    Xa_tr, Xa_va, Xa_pool, _, artA, fillA = fit_transform_block(
        seed_train=seed_train,
        seed_val=seed_val,
        pool_trainval=pool_cand,  # TRAIN-only to avoid unsupervised VAL leakage via medians
        pool_cand=pool_cand,
        num_cols=num_cols_A,
        bool_cols=bool_cols_A,
    )

    # PBE≈0 indicators (only for *application*, not for classifier training)
    pbe_tr_series = pd.to_numeric(seed_train["_pbe_eff"], errors="coerce")
    pbe_va_series = pd.to_numeric(seed_val["_pbe_eff"], errors="coerce") if len(seed_val) else pd.Series([], dtype=float)
    is_pbe0_tr = np.isfinite(pbe_tr_series.values) & (pbe_tr_series.values <= pbe_zero_thr)
    is_pbe0_va = np.isfinite(pbe_va_series.values) & (pbe_va_series.values <= pbe_zero_thr) if len(seed_val) else np.zeros(0, dtype=bool)

    pbe_pool_series = pd.to_numeric(pbe_baseline(pool_cand), errors="coerce")
    is_pbe0_pool = np.isfinite(pbe_pool_series.values) & (pbe_pool_series.values <= pbe_zero_thr)


    # Optional: DOS-derived boundary target (used for gate acquisition only)
    _dos_gate_target = float("nan")
    if "_dos_log_dosEf" in seed_train.columns and len(seed_train):
        try:
            # Use TRAIN PBE0 medians per class when available (more stable than a pool-only target)
            dos_all = pd.to_numeric(seed_train["_dos_log_dosEf"], errors="coerce").to_numpy(dtype=float)
            m_pbe0 = is_pbe0_tr
            # Note: y_metal_all defined below; we can only compute target later.
            # We'll fill _dos_gate_target once y_metal_tr exists.
        except Exception:
            _dos_gate_target = float("nan")
    # Trusted label mask
    st_tr = seed_train[args.status_col].astype(str).str.strip().str.upper()
    trusted_tr = st_tr.isin(["SUCCESS", "METAL_ASSUMED"]).values
    hse_tr_series = pd.to_numeric(seed_train["_hse"], errors="coerce").astype(float)
    hse_ok_tr = np.isfinite(hse_tr_series.values) | (st_tr.values == "METAL_ASSUMED")

    # Train classifier on ALL trusted rows (metal vs nonmetal), not only PBE0
    gate_train_mask = trusted_tr & hse_ok_tr
    gate_idx_tr = np.where(gate_train_mask)[0]

    def _entropy01(p: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(p, dtype=float), 1e-12, 1.0 - 1e-12)
        return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))

    def _predict_p1(model, X: np.ndarray) -> np.ndarray:
        # robust class-1 probability
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            classes = getattr(model, "classes_", None)
            if classes is None and hasattr(model, "estimator") and hasattr(model.estimator, "classes_"):
                classes = model.estimator.classes_
            if classes is None:
                idx1 = 1
            else:
                idx1 = int(np.where(np.asarray(classes) == 1)[0][0])
            return np.asarray(proba[:, idx1], dtype=float)

        if hasattr(model, "decision_function"):
            z = np.asarray(model.decision_function(X), dtype=float)
            return 1.0 / (1.0 + np.exp(-z))

        return np.asarray(model.predict(X), dtype=float)

    # y_metal: 1 = metal-like under HSE, 0 = nonmetal-like
    y_metal_all = seed_train["_metal_like_truth"].astype(int).values
    y_metal_tr = y_metal_all[gate_idx_tr] if gate_idx_tr.size else np.array([], dtype=int)

    n_gate = int(gate_idx_tr.size)
    prior_metal = float(np.mean(y_metal_tr)) if y_metal_tr.size else 0.5
    pos = int(np.sum(y_metal_tr == 1))
    neg = int(np.sum(y_metal_tr == 0))

    gate_clf = None
    log(f"[gate] metal_clf train_n={n_gate} metal_pos={pos} nonmetal_neg={neg} prior_metal={prior_metal:.3f} pbe_zero_thr={pbe_zero_thr:.3f}")

    # Fallback if insufficient/one-class
    if n_gate < 2 or pos == 0 or neg == 0:
        log(f"[gate] fallback: insufficient/one-class data. Using constant p_metal={prior_metal:.3f} (p_gate={1.0-prior_metal:.3f}) in PBE0.")
    else:
        min_leaf = max(2, min(5, n_gate // 8 if n_gate >= 16 else 2))
    
        
        # sample weights = class-balance * PBE-focus (so gate learns global boundary but prioritizes PBE≈0 regime)
        lam = float(getattr(args, "gate_weight_lam", 0.20))
        lam = float(np.clip(lam, 1e-6, 1.0))
        scale = float(getattr(args, "gate_weight_scale", 0.10))
        # focus weight uses distance ABOVE pbe_zero_thr; all pbe<=thr get weight=1
        pbe_gate = pd.to_numeric(pbe_tr_series.iloc[gate_idx_tr], errors="coerce").to_numpy(dtype=float)
        d_above_thr = np.maximum(np.nan_to_num(pbe_gate, nan=np.inf) - pbe_zero_thr, 0.0)
        if not np.isfinite(scale) or scale <= 0.0:
            w_focus = np.where(d_above_thr <= 0.0, 1.0, lam).astype(float)
        else:
            w_focus = (lam + (1.0 - lam) * np.exp(-d_above_thr / scale)).astype(float)
            w_focus = np.where(np.isfinite(pbe_gate), w_focus, lam).astype(float)

        w_bal = compute_sample_weight(class_weight="balanced", y=np.asarray(y_metal_tr, dtype=int)).astype(float)
        sw = (w_bal * w_focus).astype(float)
        # normalize for numerical stability across cycles
        sw = sw / float(np.mean(sw)) if np.isfinite(np.mean(sw)) and np.mean(sw) > 0 else sw

        try:
            log(f"[gate] sample_weight stats: lam={lam:.3f} scale={scale:.3f}  sw_min/med/max="
                f"{float(np.min(sw)):.3f}/{float(np.median(sw)):.3f}/{float(np.max(sw)):.3f}")
        except Exception:
            pass

        if args.clf_model == "logreg":
            base = LogisticRegression(
                max_iter=2000,
                class_weight=None,   # handled by sample_weight
                solver="lbfgs",
            )
            base.fit(Xa_tr[gate_idx_tr], y_metal_tr, sample_weight=sw)
            gate_clf = base
            log("[gate] fitted metal_clf logreg (balanced*focus via sample_weight).")
        else:
            base = HistGradientBoostingClassifier(
                max_depth=args.hgb_max_depth,
                learning_rate=args.hgb_lr,
                max_iter=args.hgb_max_iter,
                min_samples_leaf=min_leaf,
                early_stopping=False,
                random_state=int(getattr(args, "random_seed", 42)),
            )

            base.fit(Xa_tr[gate_idx_tr], y_metal_tr, sample_weight=sw)
            gate_clf = base
            log(f"[gate] fitted metal_clf HistGB: min_samples_leaf={min_leaf}.")
    
        # ---- Optional calibration (safe CV for tiny/imbalanced data) ----
        if args.clf_calibrate:
            try:
                y = np.asarray(y_metal_tr, dtype=int)  # 1=metal, 0=rescued/nonmetal
                pos_m = int(np.sum(y == 1))
                neg_m = int(np.sum(y == 0))
                cv_req = int(getattr(args, "clf_calibration_cv", 3))
                method = str(getattr(args, "clf_calibration_method", "sigmoid"))
    
                if pos_m < 2 or neg_m < 2:
                    log(f"[gate] skip calibration: too few samples per class (metal={pos_m}, nonmetal={neg_m}).")
                else:
                    cv = min(cv_req, pos_m, neg_m)
                    cv = max(2, cv)
    
                    cal_kwargs = dict(method=method, cv=cv)
    
                    # sklearn compatibility: "estimator" (newer) vs "base_estimator" (older)
                    try:
                        cal = CalibratedClassifierCV(estimator=gate_clf, **cal_kwargs)
                    except TypeError:
                        cal = CalibratedClassifierCV(base_estimator=gate_clf, **cal_kwargs)
    
                    Xg = Xa_tr[gate_idx_tr]
    
                    # Try passing sample_weight (some sklearn versions support it)
                    try:
                        if sw is not None:
                            cal.fit(Xg, y, sample_weight=sw)
                        else:
                            cal.fit(Xg, y)
                    except TypeError:
                        cal.fit(Xg, y)
    
                    gate_clf = cal
                    log(f"[gate] calibrated metal_clf (method={method} cv={cv} metal={pos_m} nonmetal={neg_m}).")
    
            except Exception as e:
                log(f"[gate] WARNING: calibration failed: {e!r}. Using uncalibrated metal_clf.")



    # Predict p_metal for all pool/val rows (then convert to p_gate only in PBE0)
    if gate_clf is None:
        p_metal_pool = np.full(len(pool_cand), prior_metal, dtype=float)
        p_metal_val  = np.full(len(seed_val), prior_metal, dtype=float) if len(seed_val) else np.array([], dtype=float)
    else:
        p_metal_pool = np.clip(_predict_p1(gate_clf, Xa_pool), 0.0, 1.0)
        p_metal_val  = np.clip(_predict_p1(gate_clf, Xa_va),   0.0, 1.0) if len(seed_val) else np.array([], dtype=float)

    # Auto-flip sanity on TRAIN: metals should have higher p_metal than nonmetals
    gate_flipped = False
    if gate_clf is not None and y_metal_tr.size >= 6 and pos > 0 and neg > 0:
        p_tr = np.clip(_predict_p1(gate_clf, Xa_tr[gate_idx_tr]), 0.0, 1.0)
        m_pos = float(np.mean(p_tr[y_metal_tr == 1])) if np.any(y_metal_tr == 1) else float("nan")
        m_neg = float(np.mean(p_tr[y_metal_tr == 0])) if np.any(y_metal_tr == 0) else float("nan")
        auc_tr = float(roc_auc_score(y_metal_tr, p_tr)) if len(np.unique(y_metal_tr)) >= 2 else float("nan")

        if (np.isfinite(m_pos) and np.isfinite(m_neg) and m_pos < m_neg) or (np.isfinite(auc_tr) and auc_tr < 0.5):
            gate_flipped = True
            log(f"[gate] WARNING: inverted metal_clf on TRAIN (mean_metal={m_pos:.3f} < mean_nonmetal={m_neg:.3f}, auc_tr={auc_tr:.3f}). Flipping p_metal <- 1-p_metal.")
            p_metal_pool = 1.0 - p_metal_pool
            p_metal_val  = 1.0 - p_metal_val
            prior_metal  = 1.0 - prior_metal

    log(f"[gate] flipped={int(gate_flipped)}")

    # Convert to p_gate = P(rescued nonmetal) only in PBE0; non-PBE0 treated as 1.0
    p_gate_pool = np.ones(len(pool_cand), dtype=float)
    p_gate_val  = np.ones(len(seed_val), dtype=float) if len(seed_val) else np.array([], dtype=float)
    
    if np.any(is_pbe0_pool):
        p_gate_pool[is_pbe0_pool] = np.clip(1.0 - p_metal_pool[is_pbe0_pool], 0.0, 1.0)
    if len(seed_val) and np.any(is_pbe0_va):
        p_gate_val[is_pbe0_va] = np.clip(1.0 - p_metal_val[is_pbe0_va], 0.0, 1.0)
    
    # --- NEW: de-saturate probabilities (prevents ~0/1 collapse on tiny/imbalanced data) ---
    def _desaturate(p: np.ndarray, eps: float = 0.02) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        p = np.clip(p, 0.0, 1.0)
        # map [0,1] -> [eps, 1-eps]
        return eps + (1.0 - 2.0 * eps) * p
    
    if np.any(is_pbe0_pool):
        p_gate_pool[is_pbe0_pool] = _desaturate(p_gate_pool[is_pbe0_pool], eps=0.02)
    if len(seed_val) and np.any(is_pbe0_va):
        p_gate_val[is_pbe0_va] = _desaturate(p_gate_val[is_pbe0_va], eps=0.02)
    
    # entropy after desaturation
    entropy_gate = np.zeros(len(pool_cand), dtype=float)
    entropy_gate_val = np.zeros(len(seed_val), dtype=float) if len(seed_val) else np.array([], dtype=float)
    if np.any(is_pbe0_pool):
        entropy_gate[is_pbe0_pool] = _entropy01(p_gate_pool[is_pbe0_pool])
    if len(seed_val) and np.any(is_pbe0_va):
        entropy_gate_val[is_pbe0_va] = _entropy01(p_gate_val[is_pbe0_va])
    
    if np.any(is_pbe0_pool):
        pg = p_gate_pool[is_pbe0_pool]
        log(f"[gate] p_gate_pool(PBE0) stats: n={int(pg.size)} "
            f"min/med/max={float(np.min(pg)):.3f}/{float(np.median(pg)):.3f}/{float(np.max(pg)):.3f}")




    # ---------------- Gate VAL metrics (restricted to PBE0) ----------------
    gate_auc = float("nan")
    gate_ap = float("nan")
    gate_rank = {"auc": float("nan"), "ap": float("nan")}
    gate_best = {
        "best_f1": float("nan"),
        "best_thr": float("nan"),
        "best_prec": float("nan"),
        "best_rec": float("nan"),
        "best_pred_pos": 0,
    }
    gate_met = {
        "thr": float(getattr(args, "gate_decision_thr", 0.5)),
        "f1": float("nan"),
        "prec": float("nan"),
        "rec": float("nan"),
        "n": 0,
        "pred_pos": 0,
        "pred_neg": 0,
    }
    gate_cm = {"tn": 0, "fp": 0, "fn": 0, "tp": 0}

    def _safe_auc_ap(y, p):
        y = np.asarray(y, int)
        p = np.asarray(p, float)
        m = np.isfinite(p)
        y = y[m]; p = p[m]
        if y.size < 2 or len(np.unique(y)) < 2:
            return float("nan"), float("nan")
        return float(roc_auc_score(y, p)), float(average_precision_score(y, p))

    def _best_f1(y, p):
        """Return best_f1, best_thr, best_prec, best_rec, best_pred_pos."""
        y = np.asarray(y, int)
        p = np.asarray(p, float)
        m = np.isfinite(p)
        y = y[m]; p = p[m]
        if y.size < 2 or len(np.unique(y)) < 2:
            return float("nan"), float("nan"), float("nan"), float("nan"), 0

        # Candidate thresholds: unique probs + endpoints
        thr_list = np.unique(np.clip(p, 0.0, 1.0))
        thr_list = np.unique(np.concatenate(([0.0], thr_list, [1.0])))

        best = (-1.0, 0.5, 0.0, 0.0, 0)  # f1, thr, prec, rec, pred_pos
        for thr in thr_list:
            yh = (p >= thr).astype(int)
            pred_pos = int(yh.sum())
            f1 = float(f1_score(y, yh, zero_division=0))
            if f1 > best[0]:
                pr = float(precision_score(y, yh, zero_division=0))
                rc = float(recall_score(y, yh, zero_division=0))
                best = (f1, float(thr), pr, rc, pred_pos)
        return best

    if len(seed_val):
        st_va = seed_val[args.status_col].astype(str).str.strip().str.upper()
        trusted_va = st_va.isin(["SUCCESS", "METAL_ASSUMED"]).values
        hse_va = pd.to_numeric(seed_val["_hse"], errors="coerce").astype(float).values
        hse_ok_va = np.isfinite(hse_va) | (st_va.values == "METAL_ASSUMED")

        # rescued label within PBE0: 1 if nonmetal-like, 0 if metal-like
        # (_metal_like_truth already encodes METAL_ASSUMED as metal-like)
        y_gate_va = (seed_val["_metal_like_truth"].values.astype(int) == 0).astype(int)

        gate_val_mask = is_pbe0_va & trusted_va & hse_ok_va
        if int(np.sum(gate_val_mask)) >= 1:
            yv = y_gate_va[gate_val_mask]
            pv = np.asarray(p_gate_val[gate_val_mask], dtype=float)

            # Ranking metrics (threshold-free)
            auc, ap = _safe_auc_ap(yv, pv)
            gate_auc = float(auc)
            gate_ap = float(ap)
            gate_rank = {"auc": float(auc), "ap": float(ap)}

            # ---- Metrics at DEPLOYED threshold (NOT fixed 0.5) ----
            gate_thr = float(getattr(args, "gate_decision_thr", 0.5))  # use the same thr you route with
            y_hat = (pv >= gate_thr).astype(int)

            gate_met = {
                "thr": gate_thr,
                "f1": float(f1_score(yv, y_hat, zero_division=0)),
                "prec": float(precision_score(yv, y_hat, zero_division=0)),
                "rec": float(recall_score(yv, y_hat, zero_division=0)),
                "n": int(len(yv)),
                "pred_pos": int(y_hat.sum()),
                "pred_neg": int((y_hat == 0).sum()),
            }

            cm = confusion_matrix(yv, y_hat, labels=[0, 1])
            gate_cm = {
                "tn": int(cm[0, 0]),
                "fp": int(cm[0, 1]),
                "fn": int(cm[1, 0]),
                "tp": int(cm[1, 1]),
            }

            # Best-F1 diagnostic (tells you if model can work, and your thr is just off)
            bf1, bthr, bpr, brc, bpos = _best_f1(yv, pv)
            gate_best = {
                "best_f1": bf1,
                "best_thr": bthr,
                "best_prec": bpr,
                "best_rec": brc,
                "best_pred_pos": bpos,
            }

# ---------------- Stage B: regression on non-metals ----------------
    # IMPORTANT: do NOT trust precomputed seed_train["_delta"] unless you 100% guarantee
    # PBE is present/finite for all labeled rows. Recompute Δ = HSE - PBE_effective here.

    # Pull truth columns
    hse_tr = pd.to_numeric(seed_train[args.hse_col], errors="coerce")
    pbe_tr = pd.to_numeric(pbe_truth(seed_train), errors="coerce")

    # Truth metal-like mask (already computed upstream)
    nonmetal_tr_mask = (seed_train["_metal_like_truth"].values.astype(int) == 0)

    # Recomputed delta (do NOT fill missing PBE with 0 for training)
    delta_tr = (hse_tr - pbe_tr).astype(float)

    # Train ONLY on nonmetals with finite HSE & finite PBE & finite Δ AND PBE>thr
    okB = (
        nonmetal_tr_mask
        & np.isfinite(hse_tr.values)
        & np.isfinite(pbe_tr.values)
        & np.isfinite(delta_tr.values)
        & (pbe_tr.values > pbe_zero_thr)
    )
    
    seed_train_B = seed_train.loc[okB].copy()
    if target_mode == "direct_hse":
        yB_train = hse_tr.loc[okB].values.astype(float)
        reg_target_name = "hse"
    else:
        yB_train = delta_tr.loc[okB].values.astype(float)
        reg_target_name = "delta"
    
    reg_fallback = f"nonmetal_pbe_gt_thr_{reg_target_name}"
    if len(seed_train_B) < 5:
        log(f"[reg] WARNING: small Δ-train set (n={len(seed_train_B)}) for PBE>thr nonmetals; continuing.")
        reg_fallback = f"small_train_nonmetal_pbe_gt_thr_{reg_target_name}"
    
    if len(seed_train_B) < 2:
        log("[select] Not enough labels to fit Δ regression (need ≥2 nonmetal rows with PBE>thr). Exiting.")
        return


    # Optional but often stabilizes GP as AL adds hard/outlier points:
    # clip Δ to reduce hyperparameter thrash from a few extreme points
    if target_mode == "delta" and getattr(args, "clip_delta", False):
        lo, hi = np.nanpercentile(yB_train, [1, 99])
        yB_train = np.clip(yB_train, lo, hi).astype(float)
        log(f"[reg] clipped Δ train to [{lo:.3f}, {hi:.3f}]")

    # Debug (keep this at least for next rerun)
    try:
        q = np.nanpercentile(yB_train, [0, 5, 50, 95, 100])
        log(f"[reg] StageB train n={len(yB_train)}  target={reg_target_name} pct=[{q[0]:.3f},{q[1]:.3f},{q[2]:.3f},{q[3]:.3f},{q[4]:.3f}]  mode={reg_fallback}")
    except Exception:
        pass

    # Prepare matrices for B (train_B, val, pool)
    Xb_tr, Xb_va, Xb_pool, Xb_tr_for_iso, artB, fillB = fit_transform_block(
        seed_train=seed_train_B,
        seed_val=seed_val,
        pool_trainval=pool_cand,  # TRAIN-only to avoid unsupervised VAL leakage via medians
        pool_cand=pool_cand,
        num_cols=num_cols_B,
        bool_cols=bool_cols_B,
    )

    gpr = GaussianProcessRegressor(
        kernel=build_kernel(Xb_tr.shape[1], kind=args.kernel),
        alpha=args.alpha,
        n_restarts_optimizer=args.restarts,
        normalize_y=True,
        random_state=42,
    )
    gpr.fit(Xb_tr, yB_train)
    # ---------------- Stage C: regression on rescued PBE0 nonmetals ----------------
    # Train on truth rescued in TRAIN: (PBE<=thr) and (HSE>thr) and not METAL_ASSUMED.
    # Predict HSE directly for PBE0 points.
    
    pbe_zero_thr = float(args.pbe_zero_thr)
    
    # Build Stage-C training subset from seed_train (NOT seed_train_B)
    st_tr_all = seed_train[args.status_col].astype(str).str.strip().str.upper()
    hse_tr_all = pd.to_numeric(seed_train[args.hse_col], errors="coerce").astype(float)
    pbe_tr_all = pd.to_numeric(pbe_truth(seed_train), errors="coerce").astype(float)
    
    okC = (
        (st_tr_all.values != "METAL_ASSUMED")
        & np.isfinite(hse_tr_all.values)
        & np.isfinite(pbe_tr_all.values)
        & (pbe_tr_all.values <= pbe_zero_thr)          # PBE0 domain
        & (hse_tr_all.values > float(args.metal_thr_ev))  # rescued truth (nonmetal)
    )
    
    seed_train_C = seed_train.loc[okC].copy()
    yC_train = pd.to_numeric(seed_train_C[args.hse_col], errors="coerce").values.astype(float)
    
    n_train_rescued_pbe0 = int(len(seed_train_C))
    MIN_C = 4  # <-- put it HERE
    
    gpr0 = None
    Xc_tr = Xc_va = Xc_pool = None  # defined for safety
    
    if n_train_rescued_pbe0 < MIN_C:
        log(f"[reg0] StageC skipped: only n={n_train_rescued_pbe0} train rows (need >={MIN_C})")
    else:
        # Use the same feature set as Stage-B (num_cols_B/bool_cols_B) to avoid new governance complexity
        Xc_tr, Xc_va, Xc_pool, _, artC, fillC = fit_transform_block(
            seed_train=seed_train_C,
            seed_val=seed_val,
            pool_trainval=pool_cand,   # TRAIN-only to avoid VAL leakage
            pool_cand=pool_cand,
            num_cols=num_cols_B,
            bool_cols=bool_cols_B,
        )
    
        kernel0 = build_kernel(Xc_tr.shape[1], kind=args.kernel)
        gpr0 = GaussianProcessRegressor(
            kernel=kernel0,
            alpha=max(float(args.alpha), 0.10),  # noise floor helps tiny n
            optimizer=None,                     # IMPORTANT: avoid hyperparam opt on tiny n
            normalize_y=True,
            random_state=42,
        )
        gpr0.fit(Xc_tr, yC_train)
        log(f"[reg0] StageC fitted: n={n_train_rescued_pbe0} (optimizer=None alpha={max(float(args.alpha),0.10):.3f})")

    # Predict Stage-B target and sigma for candidate pool (train split only)
    # In delta mode, target = Δ and HSE = PBE + target.
    # In direct_hse mode, target = HSE directly and implied Δ = HSE_pred - PBE.
    mu_target_pool, sig_target_pool = gpr.predict(Xb_pool, return_std=True)
    
    # PBE for candidates
    pbe_pool_series = pd.to_numeric(pbe_baseline(pool_cand), errors="coerce")
    pbe_pool_vals = pbe_pool_series.values.astype(float)
    is_pbe0_pool = np.isfinite(pbe_pool_vals) & (pbe_pool_vals <= pbe_zero_thr)
    if target_mode == "direct_hse":
        hse_pred_nm_B = np.asarray(mu_target_pool, dtype=float)
        mu_delta_pool = (hse_pred_nm_B - pbe_pool_vals).astype(float)
    else:
        mu_delta_pool = np.asarray(mu_target_pool, dtype=float)
        hse_pred_nm_B = pbe_pool_vals + mu_delta_pool
    sig_delta_pool = np.asarray(sig_target_pool, dtype=float)
    
    # p_gate_pool is P(rescued nonmetal) only in PBE0; non-PBE0 treated as 1.0 upstream
    p_gate_arr = np.asarray(p_gate_pool, dtype=float)
    
    # ---------------- Routing / mixture (pool) ----------------
    # Ensure threshold is defined in this scope
    gate_decision_thr = float(getattr(args, "gate_decision_thr", 0.5))
    
    # Stage B nonmetal predictor (PBE>thr): already constructed according to target_mode above
    sig_nm_B = np.asarray(sig_delta_pool, dtype=float)
    
    # Stage C predictor (PBE0 nonmetals): HSE_nm = HSE0_pred (if available), else fallback to Stage B
    if gpr0 is not None and Xc_pool is not None:
        mu_hse0_pool, sig_hse0_pool = gpr0.predict(Xc_pool, return_std=True)
        hse_pred_nm_C = np.asarray(mu_hse0_pool, dtype=float)
        sig_nm_C = np.asarray(sig_hse0_pool, dtype=float)
    else:
        hse_pred_nm_C = hse_pred_nm_B
        sig_nm_C = sig_nm_B
    
    # Choose nonmetal model by regime (PBE0 uses StageC reg0 if available)
    mu_nm_pool  = np.where(is_pbe0_pool, hse_pred_nm_C, hse_pred_nm_B).astype(float)
    sig_nm_pool = np.where(is_pbe0_pool, sig_nm_C, sig_nm_B).astype(float)
    
    # ---------------- HARD ROUTING (PBE0 only) ----------------
    # If PBE0 and p_gate < thr -> treat as metal -> HSE_pred = 0
    # Else -> treat as nonmetal -> HSE_pred = mu_nm_pool
    is_metal_route = is_pbe0_pool & (p_gate_arr < gate_decision_thr)
    hse_pred_used = np.where(is_metal_route, 0.0, mu_nm_pool).astype(float)
    
    # (optional for debugging; keep commented to avoid new columns/files)
    # hse_pred_soft = np.where(is_pbe0_pool, p_gate_arr * mu_nm_pool, mu_nm_pool).astype(float)
    
    # For reporting, define delta_pred_used consistently as HSE_pred - PBE
    delta_pred_used = (hse_pred_used - pbe_pool_vals).astype(float)
    
    # ---------------- Uncertainty ----------------
    # Keep your "mixture" sigma (captures gate uncertainty even though prediction is hard)
    # Var(HSE) = p*sigma^2 + p(1-p)*mu^2   in PBE0, else sigma^2
    var_mix = np.where(
        is_pbe0_pool,
        p_gate_arr * (sig_nm_pool ** 2) + p_gate_arr * (1.0 - p_gate_arr) * (mu_nm_pool ** 2),
        (sig_nm_pool ** 2),
    ).astype(float)
    sigma_mix_pool = np.sqrt(np.maximum(var_mix, 1e-12)).astype(float)
    
    # Keep variable names expected by downstream acquisition code:
    sigma_pool = np.asarray(sig_nm_pool, dtype=float)      # "raw" nonmetal sigma
    sigma_eff  = np.asarray(sigma_mix_pool, dtype=float)   # sigma used by scoring
    
    # Gate entropy only meaningful in PBE0 bucket
    entropy_gate = np.zeros(len(pool_cand), dtype=float)
    entropy_gate[is_pbe0_pool] = entropy_binary(p_gate_arr[is_pbe0_pool])

    
    # ---------------- Validation eval (v3 gate) ----------------
    val_metrics_B = {"MAE": float("nan"), "RMSE": float("nan"), "R2": float("nan"), "n": 0}
    end2end_metrics = {"MAE": float("nan"), "RMSE": float("nan"), "R2": float("nan"), "n": 0}
    strat: Dict[str, object] = {"cycle": int(args.cycle), "bins": {}}
    
    gate_decision_thr = float(getattr(args, "gate_decision_thr", 0.5))
    
    if len(seed_val) > 0:
        pbe_val_series = pd.to_numeric(pbe_truth(seed_val), errors="coerce")
        pbe_val = pbe_val_series.values.astype(float)   # DO NOT fill NaN with 0
        hse_val_true = pd.to_numeric(seed_val[args.hse_col], errors="coerce").values.astype(float)
    
        mu_target_val, sig_target_val = gpr.predict(Xb_va, return_std=True)
    
        is_pbe0_val = np.isfinite(pbe_val_series.values) & (pbe_val_series.values <= pbe_zero_thr)
        p_gate_val_arr = np.asarray(p_gate_val, dtype=float)
    
        # Stage B nonmetal prediction (PBE>thr), constructed according to target_mode
        if target_mode == "direct_hse":
            hse_nm_B_val = np.asarray(mu_target_val, dtype=float)
            mu_delta_val = (hse_nm_B_val - pbe_val).astype(float)
        else:
            mu_delta_val = np.asarray(mu_target_val, dtype=float)
            hse_nm_B_val = pbe_val + mu_delta_val
        sig_delta_val = np.asarray(sig_target_val, dtype=float)
        sig_nm_B_val = np.asarray(sig_delta_val, dtype=float)
    
        # Stage C nonmetal prediction (PBE0 nonmetals)
        if 'gpr0' in locals() and gpr0 is not None:
            mu_hse0_val, sig_hse0_val = gpr0.predict(Xc_va, return_std=True)
            hse_nm_C_val = np.asarray(mu_hse0_val, dtype=float)
            sig_nm_C_val = np.asarray(sig_hse0_val, dtype=float)
        else:
            hse_nm_C_val = hse_nm_B_val
            sig_nm_C_val = sig_nm_B_val
    
        mu_nm_val = np.where(is_pbe0_val, hse_nm_C_val, hse_nm_B_val).astype(float)
        sig_nm_val = np.where(is_pbe0_val, sig_nm_C_val, sig_nm_B_val).astype(float)
    
        # ---------------- HARD ROUTING (VAL) ----------------
        is_metal_val = is_pbe0_val & (p_gate_val_arr < gate_decision_thr)
        hse_val_pred = np.where(is_metal_val, 0.0, mu_nm_val).astype(float)
    
        # Delta fields (for plotting/debug)
        delta_val_true = (hse_val_true - pbe_val).astype(float)
        delta_val_pred = (hse_val_pred - pbe_val).astype(float)
    
        # Δ regression metrics: only where Δ is meaningful (truth nonmetals AND PBE>thr)
        metal_val_truth = seed_val["_metal_like_truth"].values.astype(int)
        nonmetal_val_mask = (
            (metal_val_truth == 0)
            & np.isfinite(hse_val_true)
            & np.isfinite(pbe_val_series.values)
            & (pbe_val_series.values > pbe_zero_thr)
        )
        if np.sum(nonmetal_val_mask) >= 2:
            val_metrics_B = regression_metrics(
                delta_val_true[nonmetal_val_mask],
                (np.asarray(mu_delta_val, dtype=float))[nonmetal_val_mask],  # Δ_pred from Stage B
            )
    
        ok_end = np.isfinite(hse_val_true) & np.isfinite(hse_val_pred)
        end2end_metrics = regression_metrics(hse_val_true[ok_end], hse_val_pred[ok_end])
    
        # Optional dump: per-cycle validation predictions/residuals
        if args.dump_val_preds:
            safe_mkdir(outdir / "metrics")
    
            # Keep mixture sigma for auditing (captures gate uncertainty)
            sigma_mix_val = np.where(
                is_pbe0_val,
                np.sqrt(np.maximum(
                    p_gate_val_arr * (sig_nm_val ** 2) + p_gate_val_arr * (1.0 - p_gate_val_arr) * (mu_nm_val ** 2),
                    1e-12
                )),
                sig_nm_val
            ).astype(float)
    
            df_val = pd.DataFrame({
                args.id_col: seed_val[args.id_col].astype(str).values,
                "formula": seed_val["formula"].astype(str).values if "formula" in seed_val.columns else [""] * len(seed_val),
                "source": seed_val["source"].astype(str).values if "source" in seed_val.columns else [""] * len(seed_val),
                "label_status": seed_val[args.status_col].astype(str).values,
                "pbe_gap_effective": pbe_val,
                "hse_true": hse_val_true,
                "hse_pred": hse_val_pred,
                "delta_true": delta_val_true,
                "delta_pred": delta_val_pred,
                "delta_pred_stageB": np.asarray(mu_delta_val, dtype=float),
                "stageB_target_pred": np.asarray(mu_target_val, dtype=float),
                "target_mode": target_mode,
                "sigma_gpr": sig_nm_val,       # regime-specific nonmetal sigma
                "sigma_mix": sigma_mix_val,    # mixture sigma (audit)
                "p_gate": p_gate_val_arr,
                "entropy_gate": entropy_binary(p_gate_val_arr),
                "is_pbe0": is_pbe0_val.astype(int),
            })
    
            df_val["true_bin"] = gap_bins(df_val["hse_true"].values.astype(float))
            df_val["pred_bin"] = gap_bins(df_val["hse_pred"].values.astype(float))
            df_val.to_csv(outdir / "metrics" / f"val_pred_cycle{int(args.cycle)}.csv", index=False)

        # Stratified reporting by HSE bins (VAL only)
        bins = gap_bins(hse_val_true)
        for b in ["<=0.05", "0.05-1", "1-2", ">2"]:
            m = (bins == b) & np.isfinite(hse_val_true)
            if np.sum(m) == 0:
                continue
            yb_true = hse_val_true[m]
            yb_pred = hse_val_pred[m]
            met_bin = regression_metrics(yb_true, yb_pred)
            strat["bins"][b] = {
                "n": int(np.sum(m)),
                "end2end_hse": met_bin,
            }
# ---------------- ARD dump (optional) ----------------
    if args.dump_ard:
        ard = extract_ard_lengthscales(gpr, feature_names=list(num_cols_B) + list(bool_cols_B))
        ard["cycle"] = int(args.cycle)
        ard["reg_train_mode"] = reg_fallback
        dump_json(outdir / "metrics" / f"ard_cycle{int(args.cycle)}.json", ard)

    # ---------------- Optional: dump preprocessing metadata and/or model ----------------
    if args.dump_preproc:
        scA: StandardScaler = artA["scaler"]  # type: ignore
        scB: StandardScaler = artB["scaler"]  # type: ignore
        pre = {
            "cycle": int(args.cycle),
            "id_col": str(args.id_col),
            "hse_col": str(args.hse_col),
            "status_col": str(args.status_col),
            "pool_pbe_col": str(args.pool_pbe_col),
            "split_col": str(split_col),
            "metal_thr_ev": float(args.metal_thr_ev),
            "clf_feature_set": str(args.clf_feature_set),
            "reg_feature_set": str(args.reg_feature_set),
            "include_qc_flags": bool(args.include_qc_flags),
            "include_missingness_flags": bool(args.include_missingness_flags),
            "include_shortcut_flags": bool(args.include_shortcut_flags),
            "A_num_cols": list(num_cols_A),
            "A_bool_cols": list(bool_cols_A),
            "B_num_cols": list(num_cols_B),
            "B_bool_cols": list(bool_cols_B),
            "A_fill_values": {k: float(v) for k, v in fillA.items()},
            "B_fill_values": {k: float(v) for k, v in fillB.items()},
            "A_scaler_mean": scA.mean_.tolist() if hasattr(scA, "mean_") else [],
            "A_scaler_scale": scA.scale_.tolist() if hasattr(scA, "scale_") else [],
            "B_scaler_mean": scB.mean_.tolist() if hasattr(scB, "mean_") else [],
            "B_scaler_scale": scB.scale_.tolist() if hasattr(scB, "scale_") else [],
            "kernel_kind": str(args.kernel),
            "alpha": float(args.alpha),
            "restarts": int(args.restarts),
            "pbe_zero_thr": float(args.pbe_zero_thr),
            "gate_decision_thr": float(getattr(args, "gate_decision_thr", 0.5)),
            "target_mode": target_mode,
            "reg_target_name": reg_target_name,
            "k_gate": int(getattr(args, "k_gate", 2)) if getattr(args, "gate_frac", None) is None else int(round(float(args.gate_frac) * int(args.topk))),
            "reg_train_mode": reg_fallback,
            "blacklist": sorted(list(blacklist)),
        }
        dump_json(outdir / "models" / f"preproc_cycle{int(args.cycle)}.json", pre)

    if args.dump_model:
        if joblib is None:
            log("[model] WARNING: joblib not available; skipping --dump-model.")
        else:
            safe_mkdir(outdir / "models")
            
            # Gate classifier replaces the old Stage-A "clf". Keep backward-compatible key names.
            gate_model = locals().get("gate_clf", None)
            if gate_model is None:
                gate_model = locals().get("gate", None)
            if gate_model is None:
                gate_model = locals().get("clf", None)  # fallback if something still defines it
            
            payload = {
                "cycle": int(args.cycle),
                "gate_classifier": gate_model,
                "classifier": gate_model,  # backward compat for old readers
                "gpr": gpr,
                "gpr_pbe0": locals().get("gpr0", None),
                "reg0_mode": locals().get("reg0_mode", ""),
                "target_mode": target_mode,
                "reg_target_name": reg_target_name,
                "kernel_": str(getattr(gpr, "kernel_", "")),
                "clf_feature_names": list(num_cols_A) + list(bool_cols_A),
                "reg_feature_names": list(num_cols_B) + list(bool_cols_B),
                "clf_scaler": artA["scaler"],
                "reg_scaler": artB["scaler"],
                "clf_fill_values": {k: float(v) for k, v in fillA.items()},
                "reg_fill_values": {k: float(v) for k, v in fillB.items()},
            }
            joblib.dump(payload, outdir / "models" / f"phase2_cycle{int(args.cycle)}.joblib")


    # ---------------- Acquisition (AL vs random) ----------------
    
    # Candidate pool features in reg feature space
    X_cand = Xb_pool
    X_numtrain = Xb_tr_for_iso
    
    # QBC uncertainty u_mad(x) = MAD of committee predictions (regressor)
    if args.acquisition == "random":
        rng = np.random.RandomState(int(args.random_seed))
        u_mad = rng.rand(X_cand.shape[0]).astype(float)
    else:
        if args.dump_committee_preds:
            u_mad, preds = qbc_mad_uncertainty(
                X_train=Xb_tr,
                y_train=yB_train,
                X_cand=X_cand,
                kernel_kind=args.kernel,
                alpha=args.alpha,
                restarts=args.restarts,
                committee_size=args.committee_size,
                random_state=args.qbc_random_state,
                return_preds=True,
            )
            safe_mkdir(outdir / "predictions")
            fmt = args.committee_preds_format
            if fmt == "npy":
                np.save(outdir / "predictions" / f"committee_preds_cycle{int(args.cycle)}.npy", preds)
            else:
                pd.DataFrame(preds).to_csv(
                    outdir / "predictions" / f"committee_preds_cycle{int(args.cycle)}.csv",
                    index=False,
                )
        else:
            u_mad = qbc_mad_uncertainty(
                X_train=Xb_tr,
                y_train=yB_train,
                X_cand=X_cand,
                kernel_kind=args.kernel,
                alpha=args.alpha,
                restarts=args.restarts,
                committee_size=args.committee_size,
                random_state=args.qbc_random_state,
                return_preds=False,
            )
    
    # Representativeness and anti-isolation
    rho = knn_density(X_cand, k=args.knn_k, eps=1e-12)  # higher = denser
    rho_iso, delta_iso, tau_iso, cutoff_iso = anti_isolation_rho(
        X_cand=X_cand,
        X_labeled_train=X_numtrain,
        tau_quantile=args.tau_quantile,
        cutoff_quantile=args.iso_cutoff_quantile,
        eps=1e-12,
    )
    
    # Gate uncertainty (only meaningful for PBE0); keep for Stage-C only
    u_gate = np.asarray(entropy_gate, dtype=float)
    
    # Composite reg uncertainty (kept for auditing)
    u_reg_eff = np.asarray(u_mad, dtype=float)
    
    ids_all = pool_cand[args.id_col].astype(str).tolist()
    tie_all = np.array([(stable_hash_int(s, salt=args.acq_salt) % 1_000_003) for s in ids_all], dtype=float)
    micro = 1e-12 * tie_all
    
    # Keep ranks for auditing (not used for score_base)
    u_cls = np.asarray(u_gate, dtype=float)
    ru_cls = percentile_rank(u_cls + micro, higher_is_better=True)
    ru_reg = percentile_rank(u_reg_eff + micro, higher_is_better=True)
    rrho   = percentile_rank(rho + micro, higher_is_better=True)       # higher = better (dense)
    riso   = percentile_rank(rho_iso + micro, higher_is_better=True)   # higher = better (representative)
    
    # ---- z-scored acquisition terms ----
    def _zscore(a: np.ndarray) -> np.ndarray:
        x = np.asarray(a, dtype=float)
        mu = np.nanmean(x)
        sd = np.nanstd(x)
        if not np.isfinite(sd) or sd < 1e-12:
            sd = 1.0
        z = (x - mu) / sd
        z[~np.isfinite(z)] = 0.0
        return np.clip(z, -3.0, 3.0)
    
    # Uncertainty terms (want HIGH)
    sigma_raw = np.asarray(sigma_pool, dtype=float)  # raw nonmetal GP std
    SIGMA_LAMBDA = 0.70
    sigma_mix = (1.0 - SIGMA_LAMBDA) * sigma_raw + SIGMA_LAMBDA * np.asarray(sigma_eff, dtype=float)
    
    z_sig  = _zscore(sigma_mix)
    z_mad  = _zscore(u_mad)
    z_ureg = _zscore(u_reg_eff)  # optional, kept
    
    # Structure terms (Stage-B relevant)
    z_rho  = _zscore(rho)        # want HIGH (dense) -> reward
    z_riso = _zscore(rho_iso)    # want HIGH (representative) -> reward
    z_del  = _zscore(delta_iso)  # want LOW -> penalize
    
    # weights (Stage-B)
    w_sig  = 2.00
    w_mad  = 0.40
    w_ureg = 0.05
    w_del  = 0.25
    w_rho  = float(getattr(args, "alpha_rho", 0.0))
    w_riso = float(getattr(args, "beta_iso", 0.0))
    
    # ---------------- Stage-B score (score_base) ----------------
    # IMPORTANT: Stage-B acquisition must NOT use any gate/Stage-C terms.
    score_base = (
        w_sig  * z_sig +
        w_mad  * z_mad +
        w_ureg * z_ureg
        - w_del  * z_del
        + w_rho  * z_rho
        + w_riso * z_riso
        + micro
    ).astype(float)
    
    # ---------------- Stage-C score (score_gate) ----------------
    # Keep Stage-C fully separate: uses gate entropy + boundary + optional DOS/ratio + structure terms.
    z_ent = _zscore(u_gate)  # gate entropy (PBE0 meaningful, non-PBE0 ~0)
    
    # --- DOS-based ambiguity terms for gate acquisition (PBE0 only) ---
    w_bdry  = float(getattr(args, "w_gate_bdry", 1.0))
    w_dos   = float(getattr(args, "w_gate_dos", 0.0))
    w_ratio = float(getattr(args, "w_gate_ratio", 0.0))
    
    z_dos   = np.zeros(len(pool_cand), dtype=float)
    z_ratio = np.zeros(len(pool_cand), dtype=float)
    
    if (w_dos != 0.0 or w_ratio != 0.0) and np.any(is_pbe0_pool):
        try:
            # --- DOS(Ef) ambiguity term ---
            if "_dos_log_dosEf" in pool_cand.columns:
                dos_log = pd.to_numeric(pool_cand["_dos_log_dosEf"], errors="coerce") \
                            .replace([np.inf, -np.inf], np.nan) \
                            .to_numpy(dtype=float)
    
                dos_target = float(locals().get("_dos_gate_target", float("nan")))
                if not np.isfinite(dos_target):
                    m = is_pbe0_pool & np.isfinite(dos_log)
                    if np.any(m):
                        dos_target = float(np.nanmedian(dos_log[m]))
    
                u_dos = np.full(len(pool_cand), np.nan, dtype=float)
                if np.isfinite(dos_target):
                    m = is_pbe0_pool & np.isfinite(dos_log)
                    u_dos[m] = -np.abs(dos_log[m] - dos_target)
                z_dos = _zscore(u_dos)
    
            # --- dist_ratio ambiguity term ---
            ratio_abs = None
            if "dist_ratio" in pool_cand.columns:
                ratio = pd.to_numeric(pool_cand["dist_ratio"], errors="coerce") \
                         .replace([np.inf, -np.inf], np.nan) \
                         .to_numpy(dtype=float)
                ratio = np.clip(ratio, 1e-6, 1e6)
                ratio_abs = np.abs(np.log(ratio))
            elif "_dos_abs_log_dist_ratio" in pool_cand.columns:
                ratio_abs = pd.to_numeric(pool_cand["_dos_abs_log_dist_ratio"], errors="coerce") \
                            .replace([np.inf, -np.inf], np.nan) \
                            .to_numpy(dtype=float)
    
            if ratio_abs is not None:
                u_ratio = np.full(len(pool_cand), np.nan, dtype=float)
                m = is_pbe0_pool & np.isfinite(ratio_abs)
                u_ratio[m] = -ratio_abs[m]
                z_ratio = _zscore(u_ratio)
    
        except Exception:
            z_dos[:] = 0.0
            z_ratio[:] = 0.0
    
    # --- Gate boundary proximity term (safe) ---
    z_bdry = np.zeros(len(pool_cand), dtype=float)
    if w_bdry != 0.0 and "p_gate_arr" in locals() and np.any(is_pbe0_pool):
        thr_dec = float(getattr(args, "gate_decision_thr", 0.5))
        u_bdry = np.full(len(pool_cand), np.nan, dtype=float)
        m = is_pbe0_pool & np.isfinite(p_gate_arr)
        u_bdry[m] = -np.abs(p_gate_arr[m] - thr_dec)
        z_bdry = _zscore(u_bdry)
    
    score_gate = (
        z_ent
        + w_bdry  * z_bdry
        + w_dos   * z_dos
        + w_ratio * z_ratio
        + w_rho   * z_rho
        + w_riso  * z_riso
        - w_del   * z_del
        + micro
    ).astype(float)
    
    # If acquisition=='random', force true random scores for BOTH base & gate buckets
    if args.acquisition == "random":
        rng = np.random.RandomState(int(args.random_seed))
        r = rng.rand(X_cand.shape[0]).astype(float)
        score_base = (r + micro).astype(float)
        score_gate = (r + micro).astype(float)
    
    # Guardrail: verify directions
    def _corr(a, b) -> float:
        a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 10:
            return float("nan")
        return float(np.corrcoef(a[m], b[m])[0, 1])
    
    # Make sure these exist for logging
    if "rho" not in locals() or rho is None:
        rho = np.full_like(score_base, np.nan, dtype=float)
    if "rho_iso" not in locals() or rho_iso is None:
        rho_iso = np.full_like(score_base, np.nan, dtype=float)
    if "delta_iso" not in locals() or delta_iso is None:
        delta_iso = np.full_like(score_base, np.nan, dtype=float)
    
    log(
        "[acq-check][StageB] "
        f"corr(score_base,rho)={_corr(score_base, rho):+.3f}  "
        f"corr(score_base,rho_iso)={_corr(score_base, rho_iso):+.3f}  "
        f"corr(score_base,delta_iso)={_corr(score_base, delta_iso):+.3f}  "
        f"alpha_rho={w_rho:.3f} beta_iso={w_riso:.3f}"
    )


    
    # Exclude seen IDs (resume-safe)
    seen = infer_seen_ids(outdir, args.id_col, seed_lab[args.id_col])
    pool_ids = pool_cand[args.id_col].astype(str).reset_index(drop=True)
    mask_new = ~pool_ids.isin(seen)
    log(f"[seen] seed_ids={seed_lab[args.id_col].astype(str).nunique()}  seen_total={len(seen)}  pool_cand={len(pool_ids)}  unseen={int(mask_new.sum())}")
    if int(mask_new.sum()) == 0:
        ex = list(sorted(list(seen)))[:10]
        log(f"[seen] first 10 seen ids: {ex}")
    
    if mask_new.sum() == 0:
        log("[select] No unseen items left after filtering. Exiting.")
        return
    
    unseen_idx = np.where(mask_new.values)[0]
    
    pbe_base_all = pd.to_numeric(pbe_baseline(pool_cand), errors="coerce")
    thr0 = float(getattr(args, "pbe_zero_thr", 0.05))
    gate_thr = float(getattr(args, "gate_decision_thr", 0.5))
    df_acq = pd.DataFrame({
        args.id_col: pool_ids.iloc[unseen_idx].values.astype(str),
    
        "formula": (pool_cand["formula"].astype(str).iloc[unseen_idx].values
                    if "formula" in pool_cand.columns else [""] * len(unseen_idx)),
    
        "source": (pool_cand["source"].astype(str).iloc[unseen_idx].values
                   if "source" in pool_cand.columns else [""] * len(unseen_idx)),
    
        "pbe_gap_effective": pbe_base_all.iloc[unseen_idx].values,
        "is_pbe0": (pbe_base_all.iloc[unseen_idx].values <= thr0).astype(int),
    
        "p_gate": np.asarray(p_gate_pool, dtype=float)[unseen_idx],
        "entropy_gate": np.asarray(entropy_gate, dtype=float)[unseen_idx],
    
        "u_mad": np.asarray(u_mad, dtype=float)[unseen_idx],
        "u_reg_eff": np.asarray(u_reg_eff, dtype=float)[unseen_idx],
    
        "sigma_gpr": np.asarray(sigma_pool, dtype=float)[unseen_idx],
        "sigma_eff": np.asarray(sigma_eff, dtype=float)[unseen_idx],
    
        "mu_delta_gpr": np.asarray(mu_delta_pool, dtype=float)[unseen_idx],
        "stageB_target_pred": np.asarray(mu_target_pool, dtype=float)[unseen_idx],
        "target_mode": target_mode,
        "hse_pred": np.asarray(hse_pred_used, dtype=float)[unseen_idx],
    
        "rho_knn": np.asarray(rho, dtype=float)[unseen_idx],
        "delta_iso": np.asarray(delta_iso, dtype=float)[unseen_idx],
        "rho_iso": np.asarray(rho_iso, dtype=float)[unseen_idx],
    
        # ranks kept for auditing (NOT weighted contributions)
        "rrho": np.asarray(rrho, dtype=float)[unseen_idx],
        "riso": np.asarray(riso, dtype=float)[unseen_idx],
    
        "score_base": np.asarray(score_base, dtype=float)[unseen_idx],   # Stage-B score only
        "score_gate": np.asarray(score_gate, dtype=float)[unseen_idx],   # Stage-C score only
    
        "pool_row": unseen_idx,
    })
    
    # ---------- NEW: Stage-B term contributions (audit-only; does not affect selection) ----------
    # These must exist from the acquisition block; if not, fail loudly (better than silent wrong plots).
    need = ["z_sig","z_mad","z_ureg","z_rho","z_riso","z_del","w_sig","w_mad","w_ureg","w_rho","w_riso","w_del"]
    missing = [k for k in need if k not in locals()]
    if missing:
        log(f"[WARN] missing Stage-B term vars for df_acq terms: {missing} (term_* cols will be NaN)")
        df_acq["term_sig"]  = np.nan
        df_acq["term_mad"]  = np.nan
        df_acq["term_ureg"] = np.nan
        df_acq["term_rho"]  = np.nan
        df_acq["term_riso"] = np.nan
        df_acq["term_del"]  = np.nan
    else:
        # weighted terms consistent with score_base definition
        df_acq["term_sig"]  = (np.asarray(w_sig  * z_sig,  dtype=float)[unseen_idx]).astype(float)
        df_acq["term_mad"]  = (np.asarray(w_mad  * z_mad,  dtype=float)[unseen_idx]).astype(float)
        df_acq["term_ureg"] = (np.asarray(w_ureg * z_ureg, dtype=float)[unseen_idx]).astype(float)
        df_acq["term_rho"]  = (np.asarray(w_rho  * z_rho,  dtype=float)[unseen_idx]).astype(float)
        df_acq["term_riso"] = (np.asarray(w_riso * z_riso, dtype=float)[unseen_idx]).astype(float)
        df_acq["term_del"]  = (np.asarray(-w_del * z_del, dtype=float)[unseen_idx]).astype(float)
    
    # ---------- Derived columns AFTER construction ----------
    df_acq["gate_margin"] = np.where(
        df_acq["is_pbe0"].astype(int).values == 1,
        np.abs(pd.to_numeric(df_acq["p_gate"], errors="coerce").to_numpy(float) - gate_thr),
        np.nan,
    )
    
    # IMPORTANT: for any generic ranking/inspection, use regime-appropriate score
    df_acq["score_used"] = np.where(
        df_acq["is_pbe0"].astype(int).values == 1,
        pd.to_numeric(df_acq["score_gate"], errors="coerce").to_numpy(float),
        pd.to_numeric(df_acq["score_base"], errors="coerce").to_numpy(float),
    )
    
    df_acq["_tie"] = df_acq[args.id_col].map(lambda s: stable_hash_int(s, salt=args.acq_salt))
    
    # Sort by the score actually used for that regime (does NOT change Stage-C logic; only prevents accidental mixing)
    df_acq = df_acq.sort_values(["score_used", "_tie"], ascending=[False, True]).reset_index(drop=True)
    # Save per-cycle prediction table (auditable)
    safe_mkdir(outdir / "predictions")
    
    pred_cols = [
        args.id_col, "formula", "source",
        "pbe_gap_effective", "is_pbe0",
    
        # Gate (Stage-C audit)
        "p_gate", "entropy_gate", "gate_margin",
    
        # Uncertainty / acquisition signals (Stage-B audit)
        "u_mad", "u_reg_eff",
        "sigma_gpr", "sigma_eff", "mu_delta_gpr", "stageB_target_pred", "target_mode",
    
        # Structure terms
        "rho_knn", "rho_iso", "delta_iso",
    
        # ranks (keep for compatibility)
        "rrho", "riso",
    
        # NEW: true weighted contributions (if present)
        "term_sig", "term_mad", "term_ureg", "term_rho", "term_riso", "term_del",
    
        # Scores
        "score_base", "score_gate", "score_used",
    
        # Final prediction
        "hse_pred",
    
        # Resume/debug mapping
        "pool_row",
    ]
    
    pred_cols = [c for c in pred_cols if c in df_acq.columns]
    df_acq[pred_cols].to_csv(outdir / "predictions" / f"pred_cycle{int(args.cycle)}.csv", index=False)

    # ---------------- TRUE RANDOM benchmark selection (no acquisition scoring, no FPS) ----------------
    # Note: this is a baseline for comparison. It preserves all file names/columns but chooses
    # candidates uniformly at random with the same (k_gate, k_reg) quota as AL.
    if args.acquisition == "random":
        topk = int(args.topk)
        # quota (match AL: k_gate from PBE≈0 bucket, remaining from non-PBE0)
        if getattr(args, "gate_frac", None) is not None:
            k_gate = int(round(float(args.gate_frac) * topk))
        else:
            k_gate = int(getattr(args, "k_gate", 2))
        k_gate = max(0, min(k_gate, topk))
        k_reg = topk - k_gate

        log(
            f"[select][random] quota: topk={topk} k_gate={k_gate} k_reg={k_reg} "
            f"pbe_zero_thr={float(args.pbe_zero_thr):.3f} random_seed={int(args.random_seed)} cycle={int(args.cycle)}"
        )

        rng = np.random.RandomState(int(args.random_seed) + 1000003 * int(args.cycle))

        gate_df_all = df_acq[df_acq["is_pbe0"].astype(int) == 1].copy()
        reg_df_all  = df_acq[df_acq["is_pbe0"].astype(int) == 0].copy()

        chosen_gate_idx: List[int] = []
        chosen_reg_idx: List[int] = []

        if k_gate > 0 and len(gate_df_all) > 0:
            perm = rng.permutation(gate_df_all.index.to_numpy())
            chosen_gate_idx = perm[: min(k_gate, perm.size)].tolist()

        if k_reg > 0 and len(reg_df_all) > 0:
            perm = rng.permutation(reg_df_all.index.to_numpy())
            chosen_reg_idx = perm[: min(k_reg, perm.size)].tolist()

        chosen_idx: List[int] = list(chosen_gate_idx) + list(chosen_reg_idx)

        # Backfill if any bucket is short (prefer the other bucket, then global)
        if len(chosen_gate_idx) < k_gate:
            need = k_gate - len(chosen_gate_idx)
            src = reg_df_all.loc[~reg_df_all.index.isin(set(chosen_idx))].index.to_numpy()
            if src.size > 0:
                perm = rng.permutation(src)
                extra = perm[: min(need, perm.size)].tolist()
                chosen_gate_idx += extra
                chosen_idx += extra

        if len(chosen_reg_idx) < k_reg:
            need = k_reg - len(chosen_reg_idx)
            src = gate_df_all.loc[~gate_df_all.index.isin(set(chosen_idx))].index.to_numpy()
            if src.size > 0:
                perm = rng.permutation(src)
                extra = perm[: min(need, perm.size)].tolist()
                chosen_reg_idx += extra
                chosen_idx += extra

        if len(chosen_idx) < topk:
            need = topk - len(chosen_idx)
            src = df_acq.loc[~df_acq.index.isin(set(chosen_idx))].index.to_numpy()
            if src.size > 0:
                perm = rng.permutation(src)
                chosen_idx += perm[: min(need, perm.size)].tolist()

        chosen_idx = chosen_idx[:topk]
        sel = df_acq.loc[chosen_idx].copy()

        log(
            f"[select][random] picked: gate={len(chosen_gate_idx)}/{k_gate} (PBE0)  "
            f"reg={len(chosen_reg_idx)}/{k_reg} (non-PBE0)  total={len(sel)}/{topk}"
        )

        keep_cols = [
            args.id_col, "formula", "source",
            "score_base", "score_gate",
            "pbe_gap_effective", "is_pbe0",
            "p_gate", "entropy_gate",
            "u_mad", "u_reg_eff", "sigma_gpr", "sigma_eff",
            "rho_knn", "rho_iso", "delta_iso",
        ]
        keep_cols = [c for c in keep_cols if c in sel.columns]
        sel_out = sel[keep_cols].copy()

        write_selection(outdir, args.cycle, sel_out)

        if args.dump_diversity:
            try:
                X_sel = X_cand[sel["pool_row"].values.astype(int), :]
                summ = pairwise_distance_summary(X_sel)
            except Exception:
                summ = {"mean": float("nan"), "p10": float("nan"), "p50": float("nan"), "p90": float("nan")}
            summ.update({
                "cycle": int(args.cycle),
                "n_selected": int(len(sel_out)),
                "k_gate": int(k_gate),
                "k_reg": int(k_reg),
            })
            safe_mkdir(outdir / "metrics")
            dump_json(outdir / "metrics" / f"diversity_cycle{int(args.cycle)}.json", summ)

        # Update state.json (seen ids)
        st = load_state(outdir / "state.json")
        st.setdefault("selected", {})[str(args.cycle)] = sel_out[args.id_col].astype(str).tolist()
        st["seen_ids"] = list(sorted(set(st.get("seen_ids", [])) | set(sel_out[args.id_col].astype(str))))
        dump_state(outdir / "state.json", st)




        # ---------------- Metrics outputs ----------------
        # Flat CSV row
        flat = {
            # Gate (VAL PBE0 only)
            "gate_thr": float(gate_met.get("thr", float(getattr(args, "gate_decision_thr", 0.5)))),
            "gate_auc": float(gate_rank.get("auc", float(gate_auc))),
            "gate_ap": float(gate_rank.get("ap", float(gate_ap))),
            "gate_f1": float(gate_met.get("f1", float("nan"))),
            "gate_prec": float(gate_met.get("prec", float("nan"))),
            "gate_rec": float(gate_met.get("rec", float("nan"))),
            "gate_val_n0": int(gate_met.get("n", 0)),
            "gate_pred_pos": int(gate_met.get("pred_pos", 0)),
            "gate_pred_neg": int(gate_met.get("pred_neg", 0)),
            "gate_tn": int(gate_cm.get("tn", 0)),
            "gate_fp": int(gate_cm.get("fp", 0)),
            "gate_fn": int(gate_cm.get("fn", 0)),
            "gate_tp": int(gate_cm.get("tp", 0)),
            "gate_best_f1": float(gate_best.get("best_f1", float("nan"))),
            "gate_best_thr": float(gate_best.get("best_thr", float("nan"))),
            "gate_best_prec": float(gate_best.get("best_prec", float("nan"))),
            "gate_best_rec": float(gate_best.get("best_rec", float("nan"))),
            "gate_best_pred_pos": int(gate_best.get("best_pred_pos", 0)),

            # Δ reg (VAL truth nonmetals with finite PBE_eff)
            "delta_val_MAE_nonmetal": float(val_metrics_B.get("MAE", float("nan"))),
            "delta_val_RMSE_nonmetal": float(val_metrics_B.get("RMSE", float("nan"))),
            "delta_val_R2_nonmetal": float(val_metrics_B.get("R2", float("nan"))),
            "delta_val_n_nonmetal": int(val_metrics_B.get("n", 0)),

            # End-to-end (VAL all)
            "end2end_hse_MAE": float(end2end_metrics.get("MAE", float("nan"))),
            "end2end_hse_RMSE": float(end2end_metrics.get("RMSE", float("nan"))),
            "end2end_hse_R2": float(end2end_metrics.get("R2", float("nan"))),
            "end2end_hse_n": int(end2end_metrics.get("n", 0)),

            # bookkeeping
            "cycle": int(args.cycle),
            "target_mode": target_mode,
            "reg_target_name": reg_target_name,
            "topk": int(args.topk),
            "k_gate": int(getattr(args, "k_gate", 2)) if getattr(args, "gate_frac", None) is None else int(round(float(args.gate_frac) * int(args.topk))),
            "pbe_zero_thr": float(args.pbe_zero_thr),
            "gate_decision_thr": float(getattr(args, "gate_decision_thr", 0.5)),
            "train_n_total": int(len(seed_train)),
            "train_n_nonmetal_used": int(len(seed_train_B)),
            "reg_train_mode_fallback": 1 if reg_fallback != "nonmetal_only_finite_pbe" else 0,
        }
        save_metrics_csv(outdir, args.cycle, flat)

        # Rich JSON (stratified)
        strat.update({
            "gate": {
                "thr": float(gate_met.get("thr", float(getattr(args, "gate_decision_thr", 0.5)))),
                "auc": float(gate_rank.get("auc", float(gate_auc))),
                "ap": float(gate_rank.get("ap", float(gate_ap))),
                "f1": float(gate_met.get("f1", float("nan"))),
                "prec": float(gate_met.get("prec", float("nan"))),
                "rec": float(gate_met.get("rec", float("nan"))),
                "val_n0": int(gate_met.get("n", 0)),
                "pred_pos": int(gate_met.get("pred_pos", 0)),
                "pred_neg": int(gate_met.get("pred_neg", 0)),
                "confusion": gate_cm,
                "best": gate_best,
            },
            "stageB": {"val_nonmetal_delta": val_metrics_B, "reg_train_mode": reg_fallback, "target_mode": target_mode, "reg_target_name": reg_target_name},
            "end2end": {"val_hse": end2end_metrics},
            "acquisition": {
                "w_sig": float(w_sig),
                "w_reg": float(args.w_reg),
                "alpha_rho": float(args.alpha_rho),
                "beta_iso": float(args.beta_iso),
            },
        })

        dump_json(outdir / "metrics" / f"metrics_cycle{int(args.cycle)}.json", strat)

        log(f"[select] cycle {args.cycle}: selected={len(sel_out)} k_gate={k_gate} k_reg={k_reg}")
        log(f"[select] wrote selection -> {outdir / 'selections' / f'selected_cycle{int(args.cycle)}.csv'}")


        return


    # ---------------- Quota selection + FPS diversify (deterministic) ----------------
    topk = int(args.topk)
    shortlist_mult = int(args.shortlist_mult)
    
    # Deterministic quota
    if getattr(args, "gate_frac", None) is not None:
        k_gate = int(round(float(args.gate_frac) * topk))
    else:
        k_gate = int(getattr(args, "k_gate", 2))
    k_gate = max(0, min(k_gate, topk))
    k_reg = topk - k_gate
    
    log(
        f"[select] quota (deterministic): topk={topk} k_gate={k_gate} k_reg={k_reg} "
        f"pbe_zero_thr={float(args.pbe_zero_thr):.3f}"
    )
    
    # Build per-bucket pools
    gate_df_all = df_acq[df_acq["is_pbe0"].astype(int) == 1].copy()   # Stage-C (gate rescue)
    reg_df_all  = df_acq[df_acq["is_pbe0"].astype(int) == 0].copy()   # Stage-B (delta reg)
    
    # Helper: safe numeric
    def _num(s, default=-1e9):
        return pd.to_numeric(s, errors="coerce").fillna(default).astype(float)
    
    def _log_gate_pick(tag: str, df_pick: pd.DataFrame):
        if df_pick is None or len(df_pick) == 0:
            log(f"[gate-pick] {tag}: NONE")
            return
    
        r = df_pick.iloc[0]
        sid = str(r.get(args.id_col, ""))
    
        pg  = float(pd.to_numeric(r.get("p_gate", np.nan), errors="coerce"))
        ent = float(pd.to_numeric(r.get("entropy_gate", np.nan), errors="coerce"))
    
        # Look up DOS values in pool_cand using pool_row (df_acq always has pool_row)
        pr = pd.to_numeric(r.get("pool_row", np.nan), errors="coerce")
        if np.isfinite(pr):
            pr = int(pr)
            if 0 <= pr < len(pool_cand):
                rr = pool_cand.iloc[pr]
                dosEf_raw = pd.to_numeric(rr.get("dosEf", np.nan), errors="coerce")
                gap_raw   = pd.to_numeric(rr.get("gap", np.nan), errors="coerce")
                is_gap_raw = pd.to_numeric(rr.get("is_gap", 0), errors="coerce")
            else:
                dosEf_raw = np.nan; gap_raw = np.nan; is_gap_raw = 0
        else:
            dosEf_raw = np.nan; gap_raw = np.nan; is_gap_raw = 0
    
        dos = float(np.log1p(max(float(dosEf_raw), 0.0))) if np.isfinite(dosEf_raw) else float("nan")
        dosgap = float(np.log1p(max(float(gap_raw), 0.0))) if np.isfinite(gap_raw) else float("nan")
        isg = int(is_gap_raw) if np.isfinite(is_gap_raw) else 0
    
        log(f"[gate-pick] {tag}: id={sid} p_gate={pg:.3f} ent={ent:.3f} dosEf_log={dos:.3f} dosgap_log={dosgap:.3f} is_gap={isg}")
    
    # ---- Stage-C: gate boundary filter (targets calibrated boundary) ----
    q_gate = float(getattr(args, "gate_entropy_quantile", 0.0) or 0.0)
    
    gate_df_boundary = gate_df_all
    gate_thr = None
    
    if 0.0 < q_gate < 1.0 and k_gate > 0 and len(gate_df_all) > 0:
        thr_dec = float(getattr(args, "gate_decision_thr", 0.5))
        pg = _num(gate_df_all["p_gate"], default=np.nan)
        margin = (pg - thr_dec).abs()
        m_f = margin[np.isfinite(margin.to_numpy(dtype=float))].to_numpy(dtype=float)
    
        if m_f.size >= max(10, k_gate):
            q_keep = 1.0 - q_gate
            gate_thr = float(np.quantile(m_f, q_keep))
            gate_df_boundary = gate_df_all[margin <= gate_thr].copy()
            log(
                f"[gate] boundary filter: q={q_gate:.2f} keep={q_keep:.2f} margin_thr={gate_thr:.6f} "
                f"kept={len(gate_df_boundary)}/{len(gate_df_all)} thr_dec={thr_dec:.3f}"
            )
        else:
            log(f"[gate] boundary filter skipped (gate_n={m_f.size} < {max(10, k_gate)}).")
            gate_df_boundary = gate_df_all
    
    # ---- Stage-C: build sub-policy gate scores (keep stage-C workflow) ----
    if len(gate_df_all) > 0:
        gate_df_all = gate_df_all.copy()
    
        # existing: model-based metalness and boundary uncertainty
        gate_df_all["score_gate_metal"] = 1.0 - _num(gate_df_all["p_gate"], default=0.5)
        gate_df_all["score_gate_boundary"] = _num(gate_df_all.get("entropy_gate", np.nan), default=-1e9)
    
        # existing: "rescued-like" heuristic (kept as fallback; not primary policy)
        if "mu_nm" in gate_df_all.columns:
            gate_df_all["score_gate_rescued"] = _num(gate_df_all["mu_nm"], default=-1e9)
        elif "hse_pred_nm" in gate_df_all.columns:
            gate_df_all["score_gate_rescued"] = _num(gate_df_all["hse_pred_nm"], default=-1e9)
        elif "hse_pred" in gate_df_all.columns:
            gate_df_all["score_gate_rescued"] = _num(gate_df_all["hse_pred"], default=-1e9)
        else:
            gate_df_all["score_gate_rescued"] = gate_df_all["score_gate_boundary"]
    
        # ---- NEW: DOS-driven extremes (coverage) ----
        # dos_hi: maximize DOS(Ef)
        # dos_gap: prefer is_gap, then minimize DOS(Ef)
        if "_dos_log_dosEf" in gate_df_all.columns:
            dos = _num(gate_df_all["_dos_log_dosEf"], default=np.nan)
            isgap = _num(gate_df_all.get("_dos_is_gap", np.nan), default=0.0)
    
            gate_df_all["score_gate_dos_hi"] = dos
            gate_df_all["score_gate_dos_gap"] = (5.0 * isgap) - dos
        else:
            # No DOS available -> fall back to model-based extremes
            gate_df_all["score_gate_dos_hi"] = gate_df_all["score_gate_metal"]
            gate_df_all["score_gate_dos_gap"] = gate_df_all["score_gate_rescued"]
    
        if gate_df_boundary is not gate_df_all:
            gate_df_boundary = gate_df_boundary.copy()
            for c in [
                "score_gate_metal", "score_gate_boundary", "score_gate_rescued",
                "score_gate_dos_hi", "score_gate_dos_gap",
            ]:
                if c not in gate_df_boundary.columns and c in gate_df_all.columns:
                    gate_df_boundary[c] = gate_df_all.loc[gate_df_boundary.index, c]
    
    # ---- shortlist sizes ----
    M_gate = min(len(gate_df_all), max(k_gate, shortlist_mult * max(1, k_gate))) if k_gate > 0 else 0
    
    # Stage-B: keep shortlist tight enough that diversity cannot dominate,
    # but not so tight it collapses to pure top-k
    M_reg_mult = 6
    M_reg = min(len(reg_df_all), max(k_reg, M_reg_mult * max(1, k_reg))) if k_reg > 0 else 0
    
    # shortlist builders (use _tie secondary for determinism)
    def _shortlist(df: pd.DataFrame, score_col: str, M: int) -> pd.DataFrame:
        if M <= 0 or len(df) == 0:
            return df.iloc[0:0].copy()
        if "_tie" in df.columns:
            return df.sort_values([score_col, "_tie"], ascending=[False, True]).head(M).copy()
        return df.sort_values(score_col, ascending=False).head(M).copy()
    
    # ---- Stage-C: gate shortlists per sub-policy ----
    gate_short_boundhi  = _shortlist(gate_df_boundary, "score_gate_boundary", min(M_gate, len(gate_df_boundary))) if M_gate > 0 else gate_df_all.iloc[0:0].copy()
    gate_short_bound    = _shortlist(gate_df_all,      "score_gate_boundary", M_gate) if M_gate > 0 else gate_df_all.iloc[0:0].copy()
    gate_short_dos_hi   = _shortlist(gate_df_all,      "score_gate_dos_hi",   M_gate) if M_gate > 0 else gate_df_all.iloc[0:0].copy()
    gate_short_dos_gap  = _shortlist(gate_df_all,      "score_gate_dos_gap",  M_gate) if M_gate > 0 else gate_df_all.iloc[0:0].copy()
    
    # keep these as fallbacks
    gate_short_metal    = _shortlist(gate_df_all,      "score_gate_metal",    M_gate) if M_gate > 0 else gate_df_all.iloc[0:0].copy()
    gate_short_rescued  = _shortlist(gate_df_all,      "score_gate_rescued",  M_gate) if M_gate > 0 else gate_df_all.iloc[0:0].copy()
    
    # ---- Stage-B: reg shortlist uses the unified acquisition score_base (already built upstream) ----
    reg_short = _shortlist(reg_df_all, "score_base", M_reg) if M_reg > 0 else reg_df_all.iloc[0:0].copy()
    
    # ---- Stage-B safety: score floor BEFORE any diversification ----
    q_floor = 0.35
    if k_reg > 0 and len(reg_short) > 0:
        sb = _num(reg_short["score_base"], default=np.nan).to_numpy(dtype=float)
        sb_f = sb[np.isfinite(sb)]
        if sb_f.size >= max(10, k_reg):
            thr = float(np.quantile(sb_f, q_floor))
            reg_short_floor = reg_short[_num(reg_short["score_base"], default=-1e9) >= thr].copy()
            if len(reg_short_floor) >= k_reg:
                log(
                    f"[reg] score floor applied: q_floor={q_floor:.2f} thr={thr:.6f} "
                    f"kept={len(reg_short_floor)}/{len(reg_short)}"
                )
                reg_short = reg_short_floor
    
    def pick_fps_from_df(df_short: pd.DataFrame, k: int, score_col: str, score_weight: float = 0.0):
        if k <= 0 or len(df_short) == 0:
            return []
        rows = df_short["pool_row"].values.astype(int)
        X_sub = X_cand[rows, :]
        scores_sub = _num(df_short[score_col], default=-1e9).values.astype(float)
        chosen_local = fps_with_duplicate_rejection(
            X_short=X_sub,
            scores_short=scores_sub,
            k=min(k, len(df_short)),
            dup_quantile=args.dup_quantile,
            score_weight=score_weight,
        )
        return df_short.iloc[np.array(chosen_local, dtype=int)].index.tolist()
    
    chosen_idx = []
    
    # ---- Stage-C: Gate picks (coverage policy: boundary + DOS extremes) ----
    chosen_gate_idx = []
    if k_gate > 0 and len(gate_df_all) > 0:
        taken_gate = set()
    
        # Policy:
        #  - always 1 boundary if k_gate>=1
        #  - if k_gate>=2: 1 DOS extreme (alternate by cycle parity)
        #  - if k_gate>=3: take the opposite extreme too
        k_bound = 1 if k_gate >= 1 else 0
        k_ext   = 1 if k_gate >= 2 else 0
        k_third = 1 if k_gate >= 3 else 0
        k_back  = max(0, k_gate - (k_bound + k_ext + k_third))
    
        # (1) boundary pick (entropy) within boundary-filtered shortlist if available
        if k_bound:
            bound_hi = gate_short_boundhi.loc[~gate_short_boundhi.index.isin(taken_gate)].copy()
            pick = pick_fps_from_df(bound_hi, 1, "score_gate_boundary", score_weight=0.0)
            if len(pick) < 1:
                bound_all = gate_short_bound.loc[~gate_short_bound.index.isin(taken_gate)].copy()
                pick = pick_fps_from_df(bound_all, 1, "score_gate_boundary", score_weight=0.0)
            chosen_gate_idx += pick
            taken_gate |= set(pick)
            if len(pick) > 0:
                _log_gate_pick("boundary", df_acq.loc[pick])
    
        # (2) DOS extreme pick (alternate even/odd cycle for coverage)
        if k_ext:
            want_hi = (int(args.cycle) % 2 == 0)
            if want_hi:
                cand = gate_short_dos_hi.loc[~gate_short_dos_hi.index.isin(taken_gate)].copy()
                pick = pick_fps_from_df(cand, 1, "score_gate_dos_hi", score_weight=0.0)
                log("[gate] policy: extreme=dos_hi (cycle parity even)")
                chosen_gate_idx += pick
                taken_gate |= set(pick)
                if len(pick) > 0:
                    _log_gate_pick("dos_hi", df_acq.loc[pick])
            else:
                cand = gate_short_dos_gap.loc[~gate_short_dos_gap.index.isin(taken_gate)].copy()
                pick = pick_fps_from_df(cand, 1, "score_gate_dos_gap", score_weight=0.0)
                log("[gate] policy: extreme=dos_gap (cycle parity odd)")
                chosen_gate_idx += pick
                taken_gate |= set(pick)
                if len(pick) > 0:
                    _log_gate_pick("dos_gap", df_acq.loc[pick])
    
        # (3) third pick: opposite extreme if k_gate>=3
        if k_third:
            want_hi = (int(args.cycle) % 2 == 0)
            if want_hi:
                cand = gate_short_dos_gap.loc[~gate_short_dos_gap.index.isin(taken_gate)].copy()
                pick = pick_fps_from_df(cand, 1, "score_gate_dos_gap", score_weight=0.0)
                log("[gate] policy: third=dos_gap")
                chosen_gate_idx += pick
                taken_gate |= set(pick)
                if len(pick) > 0:
                    _log_gate_pick("third_dos_gap", df_acq.loc[pick])
            else:
                cand = gate_short_dos_hi.loc[~gate_short_dos_hi.index.isin(taken_gate)].copy()
                pick = pick_fps_from_df(cand, 1, "score_gate_dos_hi", score_weight=0.0)
                log("[gate] policy: third=dos_hi")
                chosen_gate_idx += pick
                taken_gate |= set(pick)
                if len(pick) > 0:
                    _log_gate_pick("third_dos_hi", df_acq.loc[pick])
    
        # backfill from boundary shortlist if needed
        if k_back > 0:
            back = gate_short_bound.loc[~gate_short_bound.index.isin(taken_gate)].copy()
            pick = pick_fps_from_df(back, k_back, "score_gate_boundary", score_weight=0.0)
            chosen_gate_idx += pick
            taken_gate |= set(pick)
            if len(pick) > 0:
                _log_gate_pick("backfill_boundary", df_acq.loc[pick])
    
        # Final fallback (keep your existing behavior)
        if len(chosen_gate_idx) < k_gate:
            need = k_gate - len(chosen_gate_idx)
            fallback = _shortlist(gate_df_all, "score_gate", M_gate) if "score_gate" in gate_df_all.columns else gate_short_bound
            fallback = fallback.loc[~fallback.index.isin(taken_gate)].copy()
            pick = pick_fps_from_df(
                fallback,
                need,
                "score_gate" if "score_gate" in fallback.columns else "score_gate_boundary",
                score_weight=0.0,
            )
            chosen_gate_idx += pick
            taken_gate |= set(pick)
            if len(pick) > 0:
                _log_gate_pick("fallback", df_acq.loc[pick])
    
    # de-dup preserve order (prevents duplicated selections)
    chosen_gate_idx = list(dict.fromkeys(chosen_gate_idx))
    
    # ---- Stage-B: Reg picks (score-dominant + small diversify) ----
    chosen_reg_idx = []
    if k_reg > 0 and len(reg_short) > 0:
        reg_short_remaining = reg_short.loc[~reg_short.index.isin(set(chosen_gate_idx))].copy()
        if "_tie" in reg_short_remaining.columns:
            reg_sorted = reg_short_remaining.sort_values(["score_base", "_tie"], ascending=[False, True]).copy()
        else:
            reg_sorted = reg_short_remaining.sort_values("score_base", ascending=False).copy()
    
        if k_reg <= 2:
            k_reg_div = 0
        else:
            k_reg_div = 1  # keep diversification small and visible
        k_reg_score = max(0, k_reg - k_reg_div)
    
        # 1) Exploit: take top by score_base
        if k_reg_score > 0 and len(reg_sorted) > 0:
            chosen_reg_idx += reg_sorted.head(min(k_reg_score, len(reg_sorted))).index.tolist()
    
        # 2) Diversify: pick ONE farthest point among top-N remaining by score_base
        # IMPORTANT: this does NOT call FPS(k=1), because that degenerates to argmax(score) in your FPS implementation.
        if k_reg_div > 0:
            taken2 = set(chosen_gate_idx) | set(chosen_reg_idx)
            reg_div_pool = reg_sorted.loc[~reg_sorted.index.isin(taken2)].copy()
    
            log(f"[reg-debug] k_reg={k_reg} k_reg_score={k_reg_score} k_reg_div={k_reg_div} "
                f"reg_sorted={len(reg_sorted)} reg_div_pool={len(reg_div_pool)}")
    
            if len(reg_div_pool) > 0:
                # Keep scoring dominant: restrict diversification candidates to top-N by score_base
                M_div = int(min(len(reg_div_pool), max(10, 6 * max(1, k_reg))))  # e.g., k_reg=5 -> top 30
                reg_div_pool_top = reg_div_pool.head(M_div).copy()
    
                # Optional: skip the immediate next-best (prevents always selecting rank 5)
                skip_div_top = 1
                if len(reg_div_pool_top) > skip_div_top:
                    reg_div_pool_top2 = reg_div_pool_top.iloc[skip_div_top:].copy()
                else:
                    reg_div_pool_top2 = reg_div_pool_top
    
                rows = reg_div_pool_top2["pool_row"].values.astype(int)
                X_pool = X_cand[rows, :]
    
                # distance to already-selected (gate + top-score reg)
                if len(taken2) > 0:
                    sel_rows = df_acq.loc[list(taken2), "pool_row"].values.astype(int)
                    X_sel = X_cand[sel_rows, :]
                    min_dist = np.full(X_pool.shape[0], np.inf, dtype=float)
                    for j in range(X_sel.shape[0]):
                        dd = np.linalg.norm(X_pool - X_sel[j:j+1], axis=1)
                        min_dist = np.minimum(min_dist, dd)
                else:
                    min_dist = np.zeros(X_pool.shape[0], dtype=float)
    
                dmin = float(np.min(min_dist)) if min_dist.size else 0.0
                dmax = float(np.max(min_dist)) if min_dist.size else 0.0
                log(f"[reg-div] pool_top={len(reg_div_pool_top2)} dist_spread(min,max)=({dmin:.6g},{dmax:.6g}) "
                    f"skip_div_top={skip_div_top} M_div={M_div}")
    
                # If distances are tied (or near-tied), you literally can't diversify in this feature space
                if (dmax - dmin) <= 1e-12:
                    log("[reg-div] distances tied -> fallback to best remaining score in reg_div_pool_top2")
                    i0 = 0
                else:
                    i0 = int(np.argmax(min_dist))
    
                picked_name = reg_div_pool_top2.iloc[i0].name
                chosen_reg_idx += [picked_name]
    
                # Debug: show where this sits in score rank order within reg_sorted
                try:
                    picked_id = str(reg_div_pool_top2.iloc[i0][args.id_col])
                    # rank in reg_sorted (1-based)
                    reg_ids = reg_sorted[args.id_col].astype(str).tolist()
                    picked_rank = reg_ids.index(picked_id) + 1 if picked_id in reg_ids else None
                    log(f"[reg-div] picked_id={picked_id} rank_in_reg_sorted={picked_rank}")
                except Exception:
                    pass
    
    # de-dup preserve order (prevents duplicated selections)
    chosen_reg_idx = list(dict.fromkeys(chosen_reg_idx))
    
    # ---- Enforce quotas explicitly (prevents fill from turning everything into gate) ----
    chosen_gate_idx = chosen_gate_idx[:k_gate]
    chosen_reg_idx  = chosen_reg_idx[:k_reg]
    
    # If we fell short in either bucket, fill THAT bucket by score (deterministic, score-dominant)
    if len(chosen_gate_idx) < k_gate and len(gate_df_all) > 0:
        need = k_gate - len(chosen_gate_idx)
        gate_fill = gate_df_all.loc[~gate_df_all.index.isin(set(chosen_gate_idx))].copy()
        score_col = "score_gate_boundary" if "score_gate_boundary" in gate_fill.columns else ("score_gate" if "score_gate" in gate_fill.columns else None)
        if score_col is not None and len(gate_fill) > 0:
            gate_fill = _shortlist(gate_fill, score_col, min(len(gate_fill), need))
            chosen_gate_idx += gate_fill.index.tolist()
    
    if len(chosen_reg_idx) < k_reg and len(reg_df_all) > 0:
        need = k_reg - len(chosen_reg_idx)
        reg_fill = reg_df_all.loc[~reg_df_all.index.isin(set(chosen_reg_idx))].copy()
        reg_fill = reg_fill.loc[~reg_fill.index.isin(set(chosen_gate_idx))].copy()
        if len(reg_fill) > 0:
            reg_fill = _shortlist(reg_fill, "score_base", min(len(reg_fill), need))
            chosen_reg_idx += reg_fill.index.tolist()
    
    # de-dup preserve order again after fills
    chosen_gate_idx = list(dict.fromkeys(chosen_gate_idx))[:k_gate]
    chosen_reg_idx  = list(dict.fromkeys(chosen_reg_idx))[:k_reg]
    
    # combine (unique)
    chosen_idx = list(dict.fromkeys(list(chosen_gate_idx) + list(chosen_reg_idx)))
    
    # If still short (bucket shortages), fill globally using unified score (score-dominant)
    if len(chosen_idx) < topk:
        taken = set(chosen_idx)
        remaining = df_acq.loc[~df_acq.index.isin(taken)].copy()
        remaining["score_fill"] = np.where(
            remaining["is_pbe0"].astype(int).values == 1,
            _num(remaining["score_gate"], default=-1e9).values,
            _num(remaining["score_base"], default=-1e9).values,
        ).astype(float)
    
        M_fill = min(len(remaining), shortlist_mult * max(1, (topk - len(chosen_idx))))
        fill_short = remaining.sort_values("score_fill", ascending=False).head(M_fill).copy()
    
        k_need = min(topk - len(chosen_idx), len(fill_short))
    
        rows = fill_short["pool_row"].values.astype(int)
        X_sub = X_cand[rows, :]
        scores_sub = fill_short["score_fill"].values.astype(float)
    
        if len(fill_short) > 0:
            d0 = np.linalg.norm(X_sub - X_sub[0:1], axis=1)
            log(f"[fill] dist spread vs first: min={float(np.min(d0)):.6g} max={float(np.max(d0)):.6g}")
    
        top_ids = fill_short[args.id_col].astype(str).head(k_need).tolist()
        top_scores = np.round(fill_short["score_fill"].astype(float).head(k_need).to_numpy(), 6).tolist()
    
        chosen_local = fps_with_duplicate_rejection(
            X_short=X_sub,
            scores_short=scores_sub,
            k=k_need,
            dup_quantile=args.dup_quantile,
            score_weight=0.20,
        )
    
        picked = fill_short.iloc[np.array(chosen_local, dtype=int)].copy()
        picked_ids = picked[args.id_col].astype(str).tolist()
        picked_scores = np.round(picked["score_fill"].astype(float).to_numpy(), 6).tolist()
    
        log(f"[fill-debug] k_need={k_need} chosen_local={chosen_local}")
        log(f"[fill-debug] top_ids={top_ids}")
        log(f"[fill-debug] picked_ids={picked_ids}")
        log(f"[fill-debug] top_scores={top_scores}")
        log(f"[fill-debug] picked_scores={picked_scores}")
        log(f"[fill-debug] picked == top ? {picked_ids == top_ids}")
    
        chosen_idx += picked.index.tolist()
    
    # final de-dup + cap
    chosen_idx = list(dict.fromkeys(chosen_idx))[:topk]
    sel = df_acq.loc[chosen_idx].copy()
    
    log(
        f"[select] picked: gate={len(chosen_gate_idx)}/{k_gate} (PBE0)  "
        f"reg={len(chosen_reg_idx)}/{k_reg} (non-PBE0)  total={len(sel)}/{topk}"
    )
    
    keep_cols = [
        args.id_col, "formula", "source",
        "score_base", "score_gate",
        "pbe_gap_effective", "is_pbe0",
        "p_gate", "entropy_gate",
        "u_mad", "u_reg_eff", "sigma_gpr", "sigma_eff",
        "rho_knn", "rho_iso", "delta_iso",
    ]
    keep_cols = [c for c in keep_cols if c in sel.columns]
    sel_out = sel[keep_cols].copy()
    
    write_selection(outdir, args.cycle, sel_out)
    
    if args.dump_diversity:
        try:
            X_sel = X_cand[sel["pool_row"].values.astype(int), :]
            summ = pairwise_distance_summary(X_sel)
        except Exception:
            summ = {"mean": float("nan"), "p10": float("nan"), "p50": float("nan"), "p90": float("nan")}
        summ.update({
            "cycle": int(args.cycle),
            "n_selected": int(len(sel_out)),
            "k_gate": int(k_gate),
            "k_reg": int(k_reg),
        })
        safe_mkdir(outdir / "metrics")
        dump_json(outdir / "metrics" / f"diversity_cycle{int(args.cycle)}.json", summ)
    
    # Update state.json (seen ids)
    st = load_state(outdir / "state.json")
    st.setdefault("selected", {})[str(args.cycle)] = sel_out[args.id_col].astype(str).tolist()
    st["seen_ids"] = list(sorted(set(st.get("seen_ids", [])) | set(sel_out[args.id_col].astype(str))))
    dump_state(outdir / "state.json", st)


    # ---------------- Metrics outputs ----------------
    # Flat CSV row
    flat = {
        # Gate (VAL PBE0 only)
        "gate_thr": float(gate_met.get("thr", float(getattr(args, "gate_decision_thr", 0.5)))),
        "gate_auc": float(gate_rank.get("auc", float(gate_auc))),
        "gate_ap": float(gate_rank.get("ap", float(gate_ap))),
        "gate_f1": float(gate_met.get("f1", float("nan"))),
        "gate_prec": float(gate_met.get("prec", float("nan"))),
        "gate_rec": float(gate_met.get("rec", float("nan"))),
        "gate_val_n0": int(gate_met.get("n", 0)),
        "gate_pred_pos": int(gate_met.get("pred_pos", 0)),
        "gate_pred_neg": int(gate_met.get("pred_neg", 0)),
        "gate_tn": int(gate_cm.get("tn", 0)),
        "gate_fp": int(gate_cm.get("fp", 0)),
        "gate_fn": int(gate_cm.get("fn", 0)),
        "gate_tp": int(gate_cm.get("tp", 0)),
        "gate_best_f1": float(gate_best.get("best_f1", float("nan"))),
        "gate_best_thr": float(gate_best.get("best_thr", float("nan"))),
        "gate_best_prec": float(gate_best.get("best_prec", float("nan"))),
        "gate_best_rec": float(gate_best.get("best_rec", float("nan"))),
        "gate_best_pred_pos": int(gate_best.get("best_pred_pos", 0)),

        # Δ reg (VAL truth nonmetals with finite PBE_eff)
        "delta_val_MAE_nonmetal": float(val_metrics_B.get("MAE", float("nan"))),
        "delta_val_RMSE_nonmetal": float(val_metrics_B.get("RMSE", float("nan"))),
        "delta_val_R2_nonmetal": float(val_metrics_B.get("R2", float("nan"))),
        "delta_val_n_nonmetal": int(val_metrics_B.get("n", 0)),

        # End-to-end (VAL all)
        "end2end_hse_MAE": float(end2end_metrics.get("MAE", float("nan"))),
        "end2end_hse_RMSE": float(end2end_metrics.get("RMSE", float("nan"))),
        "end2end_hse_R2": float(end2end_metrics.get("R2", float("nan"))),
        "end2end_hse_n": int(end2end_metrics.get("n", 0)),

        # bookkeeping
        "cycle": int(args.cycle),
        "topk": int(args.topk),
        "k_gate": int(getattr(args, "k_gate", 2)) if getattr(args, "gate_frac", None) is None else int(round(float(args.gate_frac) * int(args.topk))),
        "pbe_zero_thr": float(args.pbe_zero_thr),
        "gate_decision_thr": float(getattr(args, "gate_decision_thr", 0.5)),
        "train_n_total": int(len(seed_train)),
        "train_n_nonmetal_used": int(len(seed_train_B)),
        "reg_train_mode_fallback": 1 if reg_fallback != "nonmetal_only_finite_pbe" else 0,
    }
    save_metrics_csv(outdir, args.cycle, flat)

    # Rich JSON (stratified)
    strat.update({
        "gate": {
            "thr": float(gate_met.get("thr", float(getattr(args, "gate_decision_thr", 0.5)))),
            "auc": float(gate_rank.get("auc", float(gate_auc))),
            "ap": float(gate_rank.get("ap", float(gate_ap))),
            "f1": float(gate_met.get("f1", float("nan"))),
            "prec": float(gate_met.get("prec", float("nan"))),
            "rec": float(gate_met.get("rec", float("nan"))),
            "val_n0": int(gate_met.get("n", 0)),
            "pred_pos": int(gate_met.get("pred_pos", 0)),
            "pred_neg": int(gate_met.get("pred_neg", 0)),
            "confusion": gate_cm,
            "best": gate_best,
        },
        "stageB": {"val_nonmetal_delta": val_metrics_B, "reg_train_mode": reg_fallback},
        "end2end": {"val_hse": end2end_metrics},
        "acquisition": {
            "w_sig": float(w_sig),
            "w_reg": float(args.w_reg),
            "alpha_rho": float(args.alpha_rho),
            "beta_iso": float(args.beta_iso),
        },
    })

    dump_json(outdir / "metrics" / f"metrics_cycle{int(args.cycle)}.json", strat)

    log(f"[select] cycle {args.cycle}: selected={len(sel_out)} k_gate={k_gate} k_reg={k_reg}")
    log(f"[select] wrote selection -> {outdir / 'selections' / f'selected_cycle{int(args.cycle)}.csv'}")

# ----------------------------- command: update -----------------------------

def cmd_update(args) -> None:
    """
    Update labels_store.csv and update seed labels:
      - upsert this cycle's results into labels_store.csv (attempt_count accumulates)
      - fill existing seed rows where hse_gap and/or pbe_gap is missing (trusted numeric statuses)
      - append brand-new trusted numeric labels (default)

    Phase-2 requirement:
      - seed.csv MUST carry label_status for all finite numeric labels.
        Any finite hse_gap with missing/blank label_status will be set to SUCCESS.
    """
    outdir = Path(args.outdir)
    safe_mkdir(outdir)

    def _norm_status(s: pd.Series) -> pd.Series:
        s = s.fillna("").astype(str).str.strip()
        s = s.mask(s.str.lower().eq("nan"), "")
        return s.str.upper()

    # ---- load seed ----
    seed_path = Path(args.seed)
    seed = read_csv(seed_path) if seed_path.exists() else pd.DataFrame(
        columns=[args.id_col, args.hse_col, args.pool_pbe_col, args.status_col]
    )

    # Ensure required columns exist in seed
    if args.id_col not in seed.columns:
        seed[args.id_col] = ""
    if args.hse_col not in seed.columns:
        seed[args.hse_col] = np.nan
    if args.pool_pbe_col not in seed.columns:
        seed[args.pool_pbe_col] = np.nan
    if args.status_col not in seed.columns:
        seed[args.status_col] = ""

    seed[args.id_col] = seed[args.id_col].astype(str).str.strip()
    seed[args.hse_col] = pd.to_numeric(seed[args.hse_col], errors="coerce")
    seed[args.pool_pbe_col] = pd.to_numeric(seed[args.pool_pbe_col], errors="coerce")
    seed[args.status_col] = _norm_status(seed[args.status_col])

    # ---- load results ----
    res = read_csv(Path(args.results_csv))
    required = [args.id_col, args.hse_col, args.status_col]
    for c in required:
        if c not in res.columns:
            raise KeyError(f"Results CSV missing required column: {c}")

    # Build normalized results frame (1 row per id)
    res_norm = pd.DataFrame(
        {
            args.id_col: res[args.id_col].astype(str).str.strip(),
            args.hse_col: pd.to_numeric(res[args.hse_col], errors="coerce"),
            args.pool_pbe_col: pd.to_numeric(res[args.pool_pbe_col], errors="coerce")
            if args.pool_pbe_col in res.columns
            else np.nan,
            "pbe_source": res["pbe_source"].astype(str) if "pbe_source" in res.columns else "",
            args.status_col: res[args.status_col],
            "fail_reason": res["fail_reason"].astype(str) if "fail_reason" in res.columns else "",
            "label_source": res["label_source"].astype(str) if "label_source" in res.columns else "",
        }
    ).drop_duplicates(subset=[args.id_col], keep="last").copy()

    res_norm[args.id_col] = res_norm[args.id_col].astype(str).str.strip()
    res_norm[args.status_col] = _norm_status(res_norm[args.status_col])

    # ---- label store ----
    store_path = outdir / "labels_store.csv"
    base_cols = [
        args.id_col,
        args.hse_col,
        args.pool_pbe_col,
        "pbe_source",
        args.status_col,
        "fail_reason",
        "attempt_count",
        "label_source",
    ]

    store = pd.DataFrame(columns=base_cols)
    if store_path.exists():
        try:
            store = pd.read_csv(store_path)
        except Exception as e:
            log(f"[update] WARNING: failed to read {store_path}: {e!r}. Rebuilding labels_store.")

    # Ensure required columns in store
    for c in base_cols:
        if c not in store.columns:
            store[c] = np.nan if c in {args.hse_col, args.pool_pbe_col} else ""
    store[args.id_col] = store[args.id_col].astype(str).str.strip()
    store["attempt_count"] = pd.to_numeric(store["attempt_count"], errors="coerce").fillna(0).astype(int)
    store[args.hse_col] = pd.to_numeric(store[args.hse_col], errors="coerce")
    store[args.pool_pbe_col] = pd.to_numeric(store[args.pool_pbe_col], errors="coerce")
    store[args.status_col] = _norm_status(store[args.status_col])
    store["pbe_source"] = store["pbe_source"].fillna("").astype(str).str.strip()

    res_ids = set(res_norm[args.id_col].astype(str).tolist())

    # Increment attempt_count for existing ids in this batch
    if len(store):
        mask_existing = store[args.id_col].isin(res_ids)
        store.loc[mask_existing, "attempt_count"] = store.loc[mask_existing, "attempt_count"] + 1

    # Add new ids to store
    existing_ids = set(store[args.id_col].astype(str).tolist())
    new_ids = res_ids - existing_ids
    if new_ids:
        new_rows = pd.DataFrame({args.id_col: list(new_ids), "attempt_count": 1})
        for c in store.columns:
            if c not in new_rows.columns:
                new_rows[c] = np.nan if c in {args.hse_col, args.pool_pbe_col} else ""
        new_rows = new_rows[store.columns]
        store = pd.concat([store, new_rows], ignore_index=True)

    # Upsert store values
    res_lookup = res_norm.set_index(args.id_col)
    upd_mask = store[args.id_col].isin(res_ids)

    new_hse = store.loc[upd_mask, args.id_col].map(res_lookup[args.hse_col])
    new_hse = pd.to_numeric(new_hse, errors="coerce").values
    old_hse = pd.to_numeric(store.loc[upd_mask, args.hse_col], errors="coerce").values
    store.loc[upd_mask, args.hse_col] = np.where(np.isfinite(new_hse), new_hse, old_hse)

    new_pbe = store.loc[upd_mask, args.id_col].map(res_lookup[args.pool_pbe_col])
    new_pbe = pd.to_numeric(new_pbe, errors="coerce").values
    old_pbe = pd.to_numeric(store.loc[upd_mask, args.pool_pbe_col], errors="coerce").values
    store.loc[upd_mask, args.pool_pbe_col] = np.where(np.isfinite(new_pbe), new_pbe, old_pbe)

    new_src = store.loc[upd_mask, args.id_col].map(res_lookup["pbe_source"])
    new_src = new_src.fillna("").astype(str).str.strip()
    valid_src = new_src.ne("") & new_src.str.lower().ne("nan")
    if valid_src.any():
        store.loc[upd_mask & valid_src, "pbe_source"] = new_src[valid_src].values

    # Always update status/fail_reason/label_source in store
    for col in [args.status_col, "fail_reason", "label_source"]:
        vals = store.loc[upd_mask, args.id_col].map(res_lookup[col])
        if col == args.status_col:
            vals = _norm_status(vals)
        else:
            vals = vals.fillna("").astype(str)
        store.loc[upd_mask, col] = vals.values

    # Write store
    ordered = [c for c in base_cols if c in store.columns] + [c for c in store.columns if c not in base_cols]
    store = store[ordered].copy()
    store.to_csv(store_path, index=False)

    # ---- Update seed with trusted numeric statuses (and status field) ----
    seed[args.id_col] = seed[args.id_col].astype(str).str.strip()

    # Ensure seed has PBE column (again, in case seed came from an older schema)
    if args.pool_pbe_col not in seed.columns:
        seed[args.pool_pbe_col] = np.nan
    seed[args.pool_pbe_col] = pd.to_numeric(seed[args.pool_pbe_col], errors="coerce")

    trusted = res_norm[res_norm[args.status_col].map(is_trusted_numeric_status)].copy()
    trusted = trusted[[args.id_col, args.hse_col, args.pool_pbe_col, args.status_col]].copy()
    trusted[args.hse_col] = pd.to_numeric(trusted[args.hse_col], errors="coerce")
    trusted[args.pool_pbe_col] = pd.to_numeric(trusted[args.pool_pbe_col], errors="coerce")
    trusted[args.status_col] = _norm_status(trusted[args.status_col])

    # Fill HSE if missing (trusted only)
    trusted_map_hse = dict(zip(trusted[args.id_col].astype(str), trusted[args.hse_col].astype(float)))
    mask_in_seed = seed[args.id_col].isin(trusted[args.id_col].astype(str))
    mask_missing_hse = pd.to_numeric(seed[args.hse_col], errors="coerce").isna()
    mask_fill_hse = mask_in_seed & mask_missing_hse

    filled_hse = int(mask_fill_hse.sum())
    if filled_hse > 0:
        seed.loc[mask_fill_hse, args.hse_col] = seed.loc[mask_fill_hse, args.id_col].map(trusted_map_hse)

    # Fill PBE if missing (trusted only)
    trusted_map_pbe = dict(zip(trusted[args.id_col].astype(str), trusted[args.pool_pbe_col].astype(float)))
    mask_missing_pbe = pd.to_numeric(seed[args.pool_pbe_col], errors="coerce").isna()
    mask_fill_pbe = mask_in_seed & mask_missing_pbe

    filled_pbe = int(mask_fill_pbe.sum())
    if filled_pbe > 0:
        seed.loc[mask_fill_pbe, args.pool_pbe_col] = seed.loc[mask_fill_pbe, args.id_col].map(trusted_map_pbe)

    # Always update label_status for ids present in results (even if non-trusted)
    res_status_map = dict(zip(res_norm[args.id_col].astype(str), res_norm[args.status_col].astype(str)))
    mask_status = seed[args.id_col].isin(res_norm[args.id_col].astype(str))
    if int(mask_status.sum()) > 0:
        seed.loc[mask_status, args.status_col] = seed.loc[mask_status, args.id_col].map(res_status_map).fillna("")

    appended = 0
    if args.append_new:
        seed_ids_set = set(seed[args.id_col].astype(str).tolist())
        new_rows = trusted[~trusted[args.id_col].astype(str).isin(seed_ids_set)].copy()
        appended = int(len(new_rows))
        if appended > 0:
            # Ensure new rows have ALL seed columns (preserve schema)
            for c in seed.columns:
                if c not in new_rows.columns:
                    new_rows[c] = np.nan if c in {args.hse_col, args.pool_pbe_col} else ""
            new_rows = new_rows[seed.columns]
            seed = pd.concat([seed, new_rows], ignore_index=True)

    # FINALIZE: any finite HSE with blank status -> SUCCESS
    seed[args.hse_col] = pd.to_numeric(seed[args.hse_col], errors="coerce")
    seed[args.pool_pbe_col] = pd.to_numeric(seed[args.pool_pbe_col], errors="coerce")
    seed[args.status_col] = _norm_status(seed[args.status_col])

    blank = seed[args.status_col].eq("")
    finite = np.isfinite(seed[args.hse_col].to_numpy(dtype=float))
    seed.loc[blank & finite, args.status_col] = "SUCCESS"

    seed.to_csv(args.out_seed, index=False)

    log(f"[update] filled_hse={filled_hse} filled_pbe={filled_pbe} appended={appended} -> {args.out_seed}")
    log(f"[update] updated labels store -> {store_path}")
    log(f"[update] seed blank status (after): {int((seed[args.status_col].eq('')).sum())}")




# ----------------------------- CLI -----------------------------

def build_cli() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("select", help="AL v3 select: Gate(PBE≈0) classifier + GPR(Δ).")
    ps.add_argument("--cycle", type=int, required=True)
    ps.add_argument("--seed", required=True)
    ps.add_argument("--pool", required=True)
    ps.add_argument("--outdir", required=True)
    ps.add_argument("--id-col", default="structure_id")
    ps.add_argument("--hse-col", default="hse_gap")
    ps.add_argument("--status-col", default="label_status")
    ps.add_argument("--pool-pbe-col", default="pbe_gap")
    ps.add_argument("--split-col", default="split")

    # Phase-2 metal-like definition
    ps.add_argument("--metal-thr-ev", type=float, default=0.05, help="Truth threshold for metal-like label (HSE <= thr).")

    # Gate (PBE≈0) protocol
    ps.add_argument(
        "--pbe-zero-thr",
        type=float,
        default=0.05,
        help="Define PBE≈0 subset: PBE_eff <= thr.",
    )
    ps.add_argument(
        "--gate-decision-thr",
        type=float,
        default=0.5,
        help="Routing threshold on p_gate: below => metal (HSE_pred=0).",
    )
    ps.add_argument(
        "--k-gate",
        type=int,
        default=2,
        help="Quota: picks from PBE≈0 gate bucket (deterministic).",
    )
    ps.add_argument(
        "--gate-frac",
        type=float,
        default=None,
        help="Optional: quota fraction from gate bucket (overrides --k-gate).",
    )
    ps.add_argument(
        "--gate-entropy-quantile",
        type=float,
        default=0.9,
        help=(
            "Gate bucket boundary targeting: if 0<q<1, restrict gate candidates to those with "
            "entropy_gate >= quantile(q) before FPS diversification; if too few remain, backfill "
            "from the full gate bucket. Set q<=0 or q>=1 to disable."
        ),
    )


    # Gate training: use all labels but focus weighting near PBE≈0
    ps.add_argument("--gate-weight-lam", type=float, default=0.20,
                    help="Gate training focus weight for non-PBE0 points (0<lam<=1). 1.0 disables PBE focusing.")
    ps.add_argument("--gate-weight-scale", type=float, default=0.10,
                    help="Gate training focus decay scale in eV above --pbe-zero-thr (only used if >0).")
    # Optional DOS-derived PBE electronic features for the gate
    ps.add_argument("--dos-csv", default="dos_from_output.csv",
                    help="Optional CSV with DOS-derived PBE features (joined by structure_id). If missing, silently ignored.")
    # Gate acquisition: optional DOS ambiguity targeting (PBE0 only)
    ps.add_argument("--w-gate-dos", type=float, default=0.30,
                    help="Weight for DOS(Ef) ambiguity term in gate score (0 disables).")
    ps.add_argument("--w-gate-ratio", type=float, default=0.20,
                    help="Weight for dist_ratio ambiguity term in gate score (0 disables).")

    # Governance controls
    ps.add_argument("--feature-blacklist", default="", help="Comma-separated feature names to drop.")
    ps.add_argument("--feature-families-json", default="", help="Optional JSON defining feature families (F1,F1b,F2,F3,EXCLUDE).")
    ps.add_argument("--clf-feature-set", default="F2F3", choices=["F2F3", "F1", "F123", "AUTO"])
    ps.add_argument("--reg-feature-set", default="F123", choices=["F2F3", "F1", "F123", "AUTO"])
    ps.add_argument("--include-qc-flags", action="store_true", help="Opt-in QC flags as features (default off).")
    ps.add_argument("--include-missingness-flags", action="store_true", help="Opt-in missingness flags as features (default off).")
    ps.add_argument("--include-shortcut-flags", action="store_true", help="Opt-in shortcut flags (is_gap0/is_metal) (default off).")
    ps.add_argument("--log-feature-families", action="store_true", help="Dump inferred family assignment JSON per cycle.")
    ps.add_argument("--safe-mode-filter", action="store_true", help="Eligibility filter using qc_structure_loaded/qc_is_perovskite_like if present.")
    ps.set_defaults(safe_mode_filter=True)

    # Stage A model
    ps.add_argument("--clf-model", default="hgb", choices=["logreg", "hgb"])
    ps.add_argument("--clf-calibrate", action="store_true")
    ps.add_argument("--clf-calibration-method", default="sigmoid", choices=["sigmoid", "isotonic"])
    ps.add_argument("--clf-calibration-cv", type=int, default=3)
    ps.add_argument("--calib-bins", type=int, default=10)

    # HGB params
    ps.add_argument("--hgb-max-depth", type=int, default=6)
    ps.add_argument("--hgb-lr", type=float, default=0.08)
    ps.add_argument("--hgb-max-iter", type=int, default=300)

    # Stage B (GPR)
    ps.add_argument(
        "--target-mode",
        default="delta",
        choices=["delta", "direct_hse"],
        help="Stage-B regression target for Gap-3 ablation: delta learns HSE-PBE; direct_hse learns HSE directly.",
    )
    ps.add_argument("--restarts", type=int, default=2)
    ps.add_argument("--kernel", default="matern", choices=["matern", "rbf"])
    ps.add_argument("--alpha", type=float, default=1e-6)
    ps.add_argument("--clip-delta", action="store_true", help="Clip Stage-B Δ target to 1st--99th percentiles (delta mode only).")

    # Acquisition
    ps.add_argument("--acquisition", default="al", choices=["al", "random"])
    ps.add_argument("--random-seed", type=int, default=42)

    ps.add_argument("--committee-size", type=int, default=20)
    ps.add_argument("--qbc-random-state", type=int, default=42)
    ps.add_argument("--knn-k", type=int, default=20)
    ps.add_argument("--alpha-rho", type=float, default=1.0)
    ps.add_argument("--beta-iso", type=float, default=1.0)
    ps.add_argument("--w-cls", type=float, default=1.0, help="Exponent weight for gate entropy rank (PBE0 only).")
    ps.add_argument("--w-reg", type=float, default=1.0, help="Exponent weight for reg uncertainty rank.")
    ps.add_argument("--shortlist-mult", type=int, default=10)
    ps.add_argument("--dup-quantile", type=float, default=0.01)
    ps.add_argument("--tau-quantile", type=float, default=0.75)
    ps.add_argument("--iso-cutoff-quantile", type=float, default=0.95)
    ps.add_argument("--acq-salt", default="acq_v1")
    ps.add_argument("--topk", type=int, default=5)

    # Artifacts
    ps.add_argument("--dump-model", action="store_true")
    ps.add_argument("--dump-preproc", action="store_true")
    ps.add_argument("--dump-val-preds", action="store_true")
    ps.add_argument("--dump-diversity", action="store_true")
    ps.add_argument("--dump-committee-preds", action="store_true")
    ps.add_argument("--committee-preds-format", default="npy", choices=["npy", "csv"])
    ps.add_argument("--dump-ard", action="store_true", help="Dump ARD lengthscales/hit-upper per cycle.")

    pu = sub.add_parser("update", help="Update labels store and fill seed labels (Phase-2).")
    pu.add_argument("--cycle", type=int, required=True)
    pu.add_argument("--seed", required=True)
    pu.add_argument("--results-csv", required=True)
    pu.add_argument("--outdir", required=True)
    pu.add_argument("--out-seed", required=True)

    pu.add_argument("--id-col", default="structure_id")
    pu.add_argument("--hse-col", default="hse_gap")
    pu.add_argument("--status-col", default="label_status")
    pu.add_argument("--pool-pbe-col", default="pbe_gap")

    pu.add_argument("--append-new", action="store_true")
    pu.add_argument("--no-append-new", dest="append_new", action="store_false")
    pu.set_defaults(append_new=True)

    return ap


def main() -> None:
    ap = build_cli()
    args = ap.parse_args()

    if args.cmd == "select":
        cmd_select(args)
    elif args.cmd == "update":
        cmd_update(args)
    else:
        raise RuntimeError("Unknown command")


if __name__ == "__main__":
    main()

