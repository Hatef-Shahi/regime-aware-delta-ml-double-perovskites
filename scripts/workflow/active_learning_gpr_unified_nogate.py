#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: active_learning_gpr_unified_nogate.py

"""
Unified no-gate AL ablation for AL v3
=====================================

This script is intentionally CLI-compatible with active_learning_gpr.py so it can
be swapped into al_loop_v3.sh for a reviewer-facing end-to-end ablation.

Scientific purpose
------------------
Current method:
    gate-aware active learning with separate PBE≈0 gate bucket and non-ZBG
    Delta-regression bucket.

This ablation:
    removes the gate/routing idea during active learning and prediction.
    It trains one unified Delta-GPR on all trusted labelled structures with
    finite PBE and HSE truth, including PBE≈0 structures and HSE-metal-like
    structures, and allows all unseen candidates to compete in one acquisition
    pool.

Key invariants kept from active_learning_gpr.py
-----------------------------------------------
- Same subcommands: select/update.
- Same argument names, so al_loop_v3.sh can call this script with the same
  keywords.
- Same seed/pool/outdir/labels_store.csv convention.
- Same fixed validation-ID logic.
- Same feature governance keywords and default feature blacklists.
- Same GPR kernel helper, QBC-MAD helper, density, anti-isolation, FPS/diversity
  utilities, and metrics/output naming where possible.

Key intentional differences from active_learning_gpr.py
-------------------------------------------------------
- No gate classifier is trained.
- No PBE≈0/non-PBE0 acquisition buckets are used.
- --k-gate and --gate-frac are accepted for CLI compatibility but ignored for
  selection. Reported ablation k_gate is 0 and k_reg is topk.
- One unified Delta-GPR is trained on all trusted labels with finite PBE and
  usable HSE truth. METAL_ASSUMED rows with missing HSE are assigned HSE=0.0
  for the unified target.
- Acquisition ranks the full unseen candidate pool, including PBE≈0 candidates.
- Prediction is HSE_pred = max(0, PBE_eff + Delta_pred), with no hard gate route.
- Ablation-specific default: if --topk is not passed on the command line, this
  script selects topk=7 per cycle (see main()). This is a designated default
  for this benchmark only; active_learning_gpr.py's default is untouched.
  Passing --topk explicitly on the CLI always overrides this default.

Usage
-----
Use this file in place of active_learning_gpr.py inside al_loop_v3.sh for the
five no-gate benchmark replicates.

Python 3.8+ compatible.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

# Make sure the original active_learning_gpr.py next to this file can be imported.
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None

try:
    import active_learning_gpr as base
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "Could not import active_learning_gpr.py. Place "
        "active_learning_gpr_unified_nogate.py in the same directory as "
        "the original active_learning_gpr.py."
    ) from exc

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix


# -----------------------------------------------------------------------------
# Ablation-specific default topk
# -----------------------------------------------------------------------------

# This unified no-gate ablation is the designated topk=7 benchmark. It is
# applied only when the caller does not explicitly pass --topk on the CLI, so
# al_loop_v3.sh (or any other caller) can still override it per-run if needed.
NOGATE_DEFAULT_TOPK: int = 7


# -----------------------------------------------------------------------------
# Small local helpers
# -----------------------------------------------------------------------------

def _norm_id(x) -> str:
    return str(x).strip()


def _safe_numeric(x) -> pd.Series:
    return pd.to_numeric(x, errors="coerce")


def _parse_pbe_from_log(fp: Path) -> Tuple[float, str]:
    """
    Parse the DFT workflow's band_GGA/Band_gap.log.

    Patterns:
      - "This system is metallic." -> 0.0
      - "Band gap: X eV" -> X
    """
    try:
        txt = fp.read_text(errors="ignore")
    except Exception:
        return (float("nan"), "")

    low = txt.lower()
    if "metallic" in low:
        return (0.0, f"pbe:{fp.as_posix()}")

    m = re.search(r"Band gap:\s*([0-9]*\.?[0-9]+)\s*eV", txt)
    if m:
        return (float(m.group(1)), f"pbe:{fp.as_posix()}")

    return (float("nan"), "")


def _zscore(a: np.ndarray) -> np.ndarray:
    x = np.asarray(a, dtype=float)
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd < 1e-12:
        sd = 1.0
    z = (x - mu) / sd
    z[~np.isfinite(z)] = 0.0
    return np.clip(z, -3.0, 3.0)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if int(m.sum()) < 10:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


def _num_series_from_df(df: pd.DataFrame, col: str, default=np.nan) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series([default] * len(df), index=df.index, dtype=float)


def _truth_hse_for_unified(seed_df: pd.DataFrame, hse_col: str, status_col: str) -> pd.Series:
    """
    Return HSE truth for unified regression.

    SUCCESS rows use numeric HSE.
    METAL_ASSUMED rows with missing HSE are assigned 0.0 eV so that the unified
    regressor sees true metal-like examples instead of silently dropping them.
    """
    hse = pd.to_numeric(seed_df[hse_col], errors="coerce").astype(float)
    status = seed_df[status_col].fillna("").astype(str).str.strip().str.upper()
    hse = hse.mask(hse.isna() & status.eq("METAL_ASSUMED"), 0.0)
    return hse


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if len(y_true) == 0:
        return {
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "tp": 0,
            "n": 0,
        }
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
        "n": int(len(y_true)),
    }


def _write_metrics_csv(outdir: Path, cycle: int, row: Dict[str, object]) -> None:
    base.safe_mkdir(outdir / "metrics")
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


def _shortlist(df: pd.DataFrame, score_col: str, M: int) -> pd.DataFrame:
    if M <= 0 or len(df) == 0:
        return df.iloc[0:0].copy()
    if "_tie" in df.columns:
        return df.sort_values([score_col, "_tie"], ascending=[False, True]).head(M).copy()
    return df.sort_values(score_col, ascending=False).head(M).copy()


# -----------------------------------------------------------------------------
# Main no-gate select command
# -----------------------------------------------------------------------------

def cmd_select(args) -> None:
    outdir = Path(args.outdir)
    base.safe_mkdir(outdir)

    base.log("[nogate] mode = unified_delta_gpr_full_pool")
    base.log("[nogate] gate classifier disabled; Stage-C rescue regressor disabled; full-pool acquisition enabled")
    base.log("[nogate] --k-gate/--gate-frac are accepted for compatibility but ignored for selection")
    base.log(f"[nogate] effective topk for this cycle = {int(args.topk)}")

    # ------------------------------------------------------------------
    # 1) PBE overrides from labels_store/seed/Output, matching the v3 policy
    # ------------------------------------------------------------------
    pbe_override: Dict[str, float] = {}
    pbe_source: Dict[str, str] = {}

    store_path = outdir / "labels_store.csv"
    store_df = None

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
                base.log(f"[pbe] loaded DFT PBE overrides from labels_store.csv: {len(pbe_override)} ids")
        except Exception as e:
            base.log(f"[pbe] WARNING: failed to load pbe overrides from {store_path}: {e!r}")

    # ------------------------------------------------------------------
    # 2) Load pool and seed
    # ------------------------------------------------------------------
    pool = base.read_csv(Path(args.pool))
    if args.safe_mode_filter:
        pool = base.apply_safe_mode_filter(pool)

    seed_lab = base.read_csv(Path(args.seed))

    pool[args.id_col] = pool[args.id_col].astype(str).map(_norm_id)
    seed_lab[args.id_col] = seed_lab[args.id_col].astype(str).map(_norm_id)

    for req in [args.id_col, args.hse_col, args.status_col]:
        if req not in seed_lab.columns:
            if req == args.status_col and store_path.exists():
                try:
                    st = store_df if store_df is not None else pd.read_csv(store_path)
                    if args.id_col in st.columns and args.status_col in st.columns:
                        tmp = st[[args.id_col, args.status_col]].copy()
                        tmp[args.id_col] = tmp[args.id_col].astype(str).map(_norm_id)
                        seed_lab = seed_lab.merge(tmp, on=args.id_col, how="left", suffixes=("", "_store"))
                        if args.status_col not in seed_lab.columns and f"{args.status_col}_store" in seed_lab.columns:
                            seed_lab[args.status_col] = seed_lab[f"{args.status_col}_store"]
                        base.log("[seed] label_status sourced from labels_store.csv")
                        continue
                except Exception:
                    pass
            raise KeyError(f"Seed must contain '{args.id_col}', '{args.hse_col}', and '{args.status_col}'.")

    if args.pool_pbe_col in seed_lab.columns:
        try:
            tmp = seed_lab[[args.id_col, args.pool_pbe_col]].copy()
            tmp[args.id_col] = tmp[args.id_col].astype(str).map(_norm_id)
            tmp[args.pool_pbe_col] = pd.to_numeric(tmp[args.pool_pbe_col], errors="coerce")
            tmp = tmp[np.isfinite(tmp[args.pool_pbe_col])]
            for sid, val in zip(tmp[args.id_col].astype(str), tmp[args.pool_pbe_col].astype(float)):
                pbe_override[sid] = float(val)
                pbe_source.setdefault(sid, "seed:pbe_gap")
            base.log(f"[pbe] merged seed PBE into overrides (seed wins): now {len(pbe_override)} ids")
        except Exception as e:
            base.log(f"[pbe] WARNING: failed to merge seed PBE overrides: {e!r}")

    # Optional Output/ auto-parse, copied in behavior from v3.
    output_root = Path("Output")
    need_ids = set(pool[args.id_col].astype(str).tolist()) - set(pbe_override.keys())

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
                    args.hse_col: np.nan,
                    args.status_col: "",
                    "fail_reason": "",
                    "attempt_count": 0,
                    "label_source": "from_output_pbe",
                })
            else:
                missed += 1

        base.log(f"[pbe] auto-parsed Output/ PBE: added={parsed} missed={missed} overrides_total={len(pbe_override)}")

        if parsed > 0:
            base.safe_mkdir(outdir)
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
            base.log(f"[pbe] cached overrides -> {store_path} rows={len(df_all)}")

    base.log("[pbe] PBE policy: labelled truth uses DFT seed/labels_store/Output; candidates fall back to pool PBE if DFT missing.")

    def pbe_truth(df: pd.DataFrame) -> pd.Series:
        base_pbe = pd.to_numeric(df.get(args.pool_pbe_col, np.nan), errors="coerce")
        ov = df[args.id_col].astype(str).map(_norm_id).map(pbe_override)
        ov = pd.to_numeric(ov, errors="coerce")
        return base_pbe.where(np.isfinite(base_pbe), ov)

    def pbe_baseline(df: pd.DataFrame) -> pd.Series:
        base_pbe = pd.to_numeric(df.get(args.pool_pbe_col, np.nan), errors="coerce")
        ov = df[args.id_col].astype(str).map(_norm_id).map(pbe_override)
        ov = pd.to_numeric(ov, errors="coerce")
        return ov.where(np.isfinite(ov), base_pbe)

    # ------------------------------------------------------------------
    # 3) Dedup/split/join labels
    # ------------------------------------------------------------------
    split_col = args.split_col
    has_split = split_col in pool.columns

    if args.pool_pbe_col not in pool.columns:
        raise KeyError(f"Pool missing '{args.pool_pbe_col}'.")

    pool = base.dedup_pool(pool, args.id_col, args.pool_pbe_col)

    if has_split:
        pool_trainval = pool[pool[split_col].astype(str).isin(["train", "val"])].copy()
        pool_cand = pool[pool[split_col].astype(str).eq("train")].copy()
    else:
        pool_trainval = pool.copy()
        pool_cand = pool.copy()

    pool_trainval = pool_trainval.reset_index(drop=True)
    pool_cand = pool_cand.reset_index(drop=True)

    cand_dft = pd.to_numeric(pool_cand[args.id_col].astype(str).map(_norm_id).map(pbe_override), errors="coerce")
    cand_pool = pd.to_numeric(pool_cand.get(args.pool_pbe_col, np.nan), errors="coerce")
    base.log(f"[pbe] cand coverage: n={len(pool_cand)} dft_known={int(np.isfinite(cand_dft).sum())} pool_pbe_finite={int(np.isfinite(cand_pool).sum())}")

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

    seed = pd.merge(
        seed_left,
        pool_trainval,
        on=args.id_col,
        how="inner",
        validate="many_to_one",
        suffixes=("_seed", ""),
    )

    if len(seed) == 0:
        raise RuntimeError("After joining seed labels to pool by id, seed is empty.")

    seed_pbe_seed_col = f"{args.pool_pbe_col}_seed"
    seed_pool_pbe = pd.to_numeric(seed.get(args.pool_pbe_col, np.nan), errors="coerce")
    if seed_pbe_seed_col in seed.columns:
        seed_seed_pbe = pd.to_numeric(seed[seed_pbe_seed_col], errors="coerce")
        both = np.isfinite(seed_seed_pbe.values) & np.isfinite(seed_pool_pbe.values)
        if both.any():
            diff = np.abs(seed_seed_pbe.values - seed_pool_pbe.values)
            n_bad = int(np.sum(both & (diff > 1e-6)))
            if n_bad > 0:
                try:
                    ex = seed.loc[both & (diff > 1e-6), [args.id_col, seed_pbe_seed_col, args.pool_pbe_col]].head(5)
                    base.log(f"[seed] WARNING: {n_bad} rows have seed vs pool PBE mismatch; using seed/Output PBE. Examples:\n{ex.to_string(index=False)}")
                except Exception:
                    base.log(f"[seed] WARNING: {n_bad} rows have seed vs pool PBE mismatch; using seed/Output PBE.")
        seed[args.pool_pbe_col] = seed_seed_pbe.where(np.isfinite(seed_seed_pbe), np.nan)
        seed = seed.drop(columns=[seed_pbe_seed_col])
    else:
        seed[args.pool_pbe_col] = np.nan

    seed_pbe = pd.to_numeric(pbe_truth(seed), errors="coerce")
    seed_hse_for_truth = _truth_hse_for_unified(seed, args.hse_col, args.status_col)

    pbe_arr = seed_pbe.to_numpy(dtype=float)
    pbe0_arr = np.isfinite(pbe_arr) & (pbe_arr <= float(args.pbe_zero_thr))
    base.log(
        f"[seed] merged_seed n={len(seed)} finite_pbe={int(np.isfinite(pbe_arr).sum())} "
        f"pbe0={int(pbe0_arr.sum())} pbe_zero_thr={float(args.pbe_zero_thr):.3f}"
    )

    seed = seed.copy()
    seed["_pbe_eff"] = seed_pbe.values
    seed["_hse"] = seed_hse_for_truth.values
    seed["_delta_unified"] = (seed_hse_for_truth - seed_pbe).values

    metal_like_truth = base.compute_metal_like_hse(
        hse=seed_hse_for_truth.values,
        status=seed[args.status_col],
        thr_ev=float(args.metal_thr_ev),
    )
    seed["_metal_like_truth"] = metal_like_truth.astype(int)

    # ------------------------------------------------------------------
    # 4) Fixed validation split
    # ------------------------------------------------------------------
    val_ids = base.load_or_create_fixed_val_ids(
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
    val_ids_set = set(str(s).strip() for s in val_ids)

    seed_id_series = seed[args.id_col].astype(str).str.strip()
    va_mask = seed_id_series.isin(val_ids_set).values
    tr_mask = ~va_mask

    seed_train_all = seed.loc[tr_mask].copy()
    seed_val = seed.loc[va_mask].copy()

    if len(seed_val) < 3:
        base.log(f"[val] WARNING: only {len(seed_val)} VAL rows present after merge; metrics may be noisy.")
    if len(seed_train_all) < 3:
        base.log(f"[train] WARNING: only {len(seed_train_all)} TRAIN rows present after VAL split; training may fail.")

    # ------------------------------------------------------------------
    # 5) Feature governance: preserve v3 names and defaults
    # ------------------------------------------------------------------
    meta_cols = [args.id_col, "formula", "source", args.pool_pbe_col]
    if has_split:
        meta_cols.append(split_col)

    blacklist_cli = base.parse_blacklist(args.feature_blacklist or "")
    blacklist = set(base.DEFAULT_SELECT_FEATURE_BLACKLIST) | set(blacklist_cli)
    blacklist.add(str(args.pool_pbe_col).strip())
    blacklist.add(f"{str(args.pool_pbe_col).strip()}_seed")
    blacklist.add("_pbe_eff")
    blacklist.add("pbe_gap_effective")
    blacklist.add("_pbe_gap_pool_orig")
    blacklist.add("pbe_source")
    if split_col:
        blacklist.add(str(split_col).strip())
    blacklist.add("split")

    families_json = base.load_feature_families_json(args.feature_families_json)

    # Use the regression feature set for the unified regressor.
    num_cols_B, bool_cols_B, famB = base.select_feature_columns(
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

    def _drop_leaky(cols: Sequence[str]) -> List[str]:
        out = []
        sc = str(split_col).strip().lower() if split_col else ""
        for c in cols:
            if c is None:
                continue
            lc = str(c).strip().lower()
            if sc and lc == sc:
                continue
            if lc.startswith("split"):
                continue
            out.append(c)
        return out

    num_cols_B = _drop_leaky(num_cols_B)
    bool_cols_B = _drop_leaky(bool_cols_B)

    bad_pbe_features = [c for c in list(num_cols_B) + list(bool_cols_B) if str(c).strip() in {
        str(args.pool_pbe_col).strip(),
        f"{str(args.pool_pbe_col).strip()}_seed",
        "_pbe_eff",
        "pbe_gap_effective",
    }]
    if bad_pbe_features:
        raise RuntimeError(f"[features] BUG: PBE leaked into unified features: {bad_pbe_features}")

    base.log(f"[features] unified: num={len(num_cols_B)} bool={len(bool_cols_B)} feature_set={args.reg_feature_set}")
    if args.log_feature_families:
        base.dump_json(outdir / "metrics" / f"feature_families_cycle{int(args.cycle)}.json", {"B_unified": famB})

    # ------------------------------------------------------------------
    # 6) Unified Delta-GPR training set
    # ------------------------------------------------------------------
    status_tr = seed_train_all[args.status_col].fillna("").astype(str).str.strip().str.upper()
    trusted_tr = status_tr.isin(["SUCCESS", "METAL_ASSUMED"]).values
    pbe_tr = pd.to_numeric(seed_train_all["_pbe_eff"], errors="coerce").astype(float)
    hse_tr = pd.to_numeric(seed_train_all["_hse"], errors="coerce").astype(float)

    okU = trusted_tr & np.isfinite(pbe_tr.values) & np.isfinite(hse_tr.values)
    seed_train_U = seed_train_all.loc[okU].copy()
    yU_train = (pd.to_numeric(seed_train_U["_hse"], errors="coerce") - pd.to_numeric(seed_train_U["_pbe_eff"], errors="coerce")).values.astype(float)

    if len(seed_train_U) < 2:
        base.log("[nogate] Not enough unified labels to fit Delta regression. Exiting.")
        return

    try:
        q = np.nanpercentile(yU_train, [0, 5, 50, 95, 100])
        n_pbe0_train = int((pd.to_numeric(seed_train_U["_pbe_eff"], errors="coerce") <= float(args.pbe_zero_thr)).sum())
        n_metal_train = int(seed_train_U["_metal_like_truth"].astype(int).sum())
        base.log(
            f"[nogate] unified Δ train n={len(yU_train)} pbe0={n_pbe0_train} hse_metal_like={n_metal_train} "
            f"Δ pct=[{q[0]:.3f},{q[1]:.3f},{q[2]:.3f},{q[3]:.3f},{q[4]:.3f}]"
        )
    except Exception:
        pass

    Xb_tr, Xb_va, Xb_pool, Xb_tr_for_iso, artB, fillB = base.fit_transform_block(
        seed_train=seed_train_U,
        seed_val=seed_val,
        pool_trainval=pool_cand,
        pool_cand=pool_cand,
        num_cols=num_cols_B,
        bool_cols=bool_cols_B,
    )

    gpr = GaussianProcessRegressor(
        kernel=base.build_kernel(Xb_tr.shape[1], kind=args.kernel),
        alpha=float(args.alpha),
        n_restarts_optimizer=int(args.restarts),
        normalize_y=True,
        random_state=42,
    )
    gpr.fit(Xb_tr, yU_train)
    base.log(f"[nogate] fitted unified Delta-GPR: {getattr(gpr, 'kernel_', '')}")

    # ------------------------------------------------------------------
    # 7) Candidate predictions, uncertainty, acquisition score on FULL pool
    # ------------------------------------------------------------------
    X_cand = Xb_pool
    mu_delta_pool, sig_delta_pool = gpr.predict(X_cand, return_std=True)

    pbe_pool_series = pd.to_numeric(pbe_baseline(pool_cand), errors="coerce")
    pbe_pool_vals = pbe_pool_series.to_numpy(dtype=float)
    is_pbe0_pool = np.isfinite(pbe_pool_vals) & (pbe_pool_vals <= float(args.pbe_zero_thr))

    hse_pred_pool = pbe_pool_vals + np.asarray(mu_delta_pool, dtype=float)
    hse_pred_pool = np.clip(hse_pred_pool, 0.0, None)

    if args.acquisition == "random":
        rng = np.random.RandomState(int(args.random_seed))
        u_mad = rng.rand(X_cand.shape[0]).astype(float)
    else:
        if args.dump_committee_preds:
            u_mad, preds = base.qbc_mad_uncertainty(
                X_train=Xb_tr,
                y_train=yU_train,
                X_cand=X_cand,
                kernel_kind=args.kernel,
                alpha=float(args.alpha),
                restarts=int(args.restarts),
                committee_size=int(args.committee_size),
                random_state=int(args.qbc_random_state),
                return_preds=True,
            )
            base.safe_mkdir(outdir / "predictions")
            if args.committee_preds_format == "npy":
                np.save(outdir / "predictions" / f"committee_preds_cycle{int(args.cycle)}.npy", preds)
            else:
                pd.DataFrame(preds).to_csv(outdir / "predictions" / f"committee_preds_cycle{int(args.cycle)}.csv", index=False)
        else:
            u_mad = base.qbc_mad_uncertainty(
                X_train=Xb_tr,
                y_train=yU_train,
                X_cand=X_cand,
                kernel_kind=args.kernel,
                alpha=float(args.alpha),
                restarts=int(args.restarts),
                committee_size=int(args.committee_size),
                random_state=int(args.qbc_random_state),
                return_preds=False,
            )

    rho = base.knn_density(X_cand, k=int(args.knn_k), eps=1e-12)
    rho_iso, delta_iso, tau_iso, cutoff_iso = base.anti_isolation_rho(
        X_cand=X_cand,
        X_labeled_train=Xb_tr_for_iso,
        tau_quantile=float(args.tau_quantile),
        cutoff_quantile=float(args.iso_cutoff_quantile),
        eps=1e-12,
    )

    ids_all = pool_cand[args.id_col].astype(str).tolist()
    tie_all = np.array([(base.stable_hash_int(s, salt=args.acq_salt) % 1_000_003) for s in ids_all], dtype=float)
    micro = 1e-12 * tie_all

    # Unified no-gate uncertainty: no mixture with gate uncertainty.
    sigma_raw = np.asarray(sig_delta_pool, dtype=float)
    sigma_eff = sigma_raw.copy()
    u_reg_eff = np.asarray(u_mad, dtype=float)

    z_sig = _zscore(sigma_eff)
    z_mad = _zscore(u_mad)
    z_ureg = _zscore(u_reg_eff)
    z_rho = _zscore(rho)
    z_riso = _zscore(rho_iso)
    z_del = _zscore(delta_iso)

    w_sig = 2.00
    w_mad = 0.40
    w_ureg = 0.05
    w_del = 0.25
    w_rho = float(getattr(args, "alpha_rho", 0.0))
    w_riso = float(getattr(args, "beta_iso", 0.0))

    score_base = (
        w_sig * z_sig
        + w_mad * z_mad
        + w_ureg * z_ureg
        - w_del * z_del
        + w_rho * z_rho
        + w_riso * z_riso
        + micro
    ).astype(float)

    if args.acquisition == "random":
        rng = np.random.RandomState(int(args.random_seed) + 1000003 * int(args.cycle))
        score_base = (rng.rand(X_cand.shape[0]).astype(float) + micro).astype(float)

    base.log(
        "[acq-check][UnifiedNoGate] "
        f"corr(score_base,rho)={_corr(score_base, rho):+.3f} "
        f"corr(score_base,rho_iso)={_corr(score_base, rho_iso):+.3f} "
        f"corr(score_base,delta_iso)={_corr(score_base, delta_iso):+.3f} "
        f"alpha_rho={w_rho:.3f} beta_iso={w_riso:.3f}"
    )

    # ------------------------------------------------------------------
    # 8) Validation metrics and prediction dump
    # ------------------------------------------------------------------
    val_metrics_delta_all = {"MAE": float("nan"), "RMSE": float("nan"), "R2": float("nan"), "n": 0}
    val_metrics_delta_nonmetal_pbegt = {"MAE": float("nan"), "RMSE": float("nan"), "R2": float("nan"), "n": 0}
    end2end_metrics = {"MAE": float("nan"), "RMSE": float("nan"), "R2": float("nan"), "n": 0}
    zbg_decision = {"precision": float("nan"), "recall": float("nan"), "f1": float("nan"), "tn": 0, "fp": 0, "fn": 0, "tp": 0, "n": 0}

    if len(seed_val) > 0:
        pbe_val_series = pd.to_numeric(pbe_truth(seed_val), errors="coerce")
        pbe_val = pbe_val_series.to_numpy(dtype=float)
        hse_val_true = _truth_hse_for_unified(seed_val, args.hse_col, args.status_col).to_numpy(dtype=float)

        mu_delta_val, sig_delta_val = gpr.predict(Xb_va, return_std=True)
        hse_val_pred = pbe_val + np.asarray(mu_delta_val, dtype=float)
        hse_val_pred = np.clip(hse_val_pred, 0.0, None)

        delta_val_true = hse_val_true - pbe_val
        delta_val_pred = np.asarray(mu_delta_val, dtype=float)

        ok_delta_all = np.isfinite(delta_val_true) & np.isfinite(delta_val_pred)
        if int(ok_delta_all.sum()) >= 2:
            val_metrics_delta_all = base.regression_metrics(delta_val_true[ok_delta_all], delta_val_pred[ok_delta_all])

        metal_val_truth = base.compute_metal_like_hse(
            hse=hse_val_true,
            status=seed_val[args.status_col],
            thr_ev=float(args.metal_thr_ev),
        ).astype(int)
        is_pbe0_val = np.isfinite(pbe_val) & (pbe_val <= float(args.pbe_zero_thr))
        nonmetal_pbegt_mask = (
            (metal_val_truth == 0)
            & np.isfinite(hse_val_true)
            & np.isfinite(pbe_val)
            & (pbe_val > float(args.pbe_zero_thr))
        )
        if int(nonmetal_pbegt_mask.sum()) >= 2:
            val_metrics_delta_nonmetal_pbegt = base.regression_metrics(
                delta_val_true[nonmetal_pbegt_mask],
                delta_val_pred[nonmetal_pbegt_mask],
            )

        ok_end = np.isfinite(hse_val_true) & np.isfinite(hse_val_pred)
        if int(ok_end.sum()) >= 2:
            end2end_metrics = base.regression_metrics(hse_val_true[ok_end], hse_val_pred[ok_end])

        # For PBE0 validation only, the no-gate implicit rescue decision is HSE_pred > pred threshold.
        if int(is_pbe0_val.sum()) > 0:
            y_rescue_true = ((metal_val_truth == 0) & is_pbe0_val).astype(int)[is_pbe0_val]
            y_rescue_pred = (hse_val_pred[is_pbe0_val] > float(args.metal_thr_ev)).astype(int)
            zbg_decision = _classification_metrics(y_rescue_true, y_rescue_pred)

        if args.dump_val_preds:
            base.safe_mkdir(outdir / "metrics")
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
                "sigma_gpr": np.asarray(sig_delta_val, dtype=float),
                "sigma_mix": np.asarray(sig_delta_val, dtype=float),
                "p_gate": np.full(len(seed_val), np.nan),
                "entropy_gate": np.zeros(len(seed_val), dtype=float),
                "is_pbe0": is_pbe0_val.astype(int),
                "model_route": ["unified_delta_gpr"] * len(seed_val),
            })
            df_val["true_bin"] = base.gap_bins(df_val["hse_true"].values.astype(float))
            df_val["pred_bin"] = base.gap_bins(df_val["hse_pred"].values.astype(float))
            df_val.to_csv(outdir / "metrics" / f"val_pred_cycle{int(args.cycle)}.csv", index=False)

    # ------------------------------------------------------------------
    # 9) Model/preprocessor dumps
    # ------------------------------------------------------------------
    if args.dump_ard:
        ard = base.extract_ard_lengthscales(gpr, feature_names=list(artB["num_cols"]) + list(artB["bool_cols"]))
        ard["cycle"] = int(args.cycle)
        ard["reg_train_mode"] = "unified_delta_all_finite_pbe"
        base.dump_json(outdir / "metrics" / f"ard_cycle{int(args.cycle)}.json", ard)

    if args.dump_preproc:
        scB = artB["scaler"]
        pre = {
            "cycle": int(args.cycle),
            "id_col": str(args.id_col),
            "hse_col": str(args.hse_col),
            "status_col": str(args.status_col),
            "pool_pbe_col": str(args.pool_pbe_col),
            "split_col": str(split_col),
            "metal_thr_ev": float(args.metal_thr_ev),
            "pbe_zero_thr": float(args.pbe_zero_thr),
            "mode": "unified_nogate_delta_gpr",
            "gate_disabled": True,
            "clf_feature_set": str(args.clf_feature_set),
            "reg_feature_set": str(args.reg_feature_set),
            "include_qc_flags": bool(args.include_qc_flags),
            "include_missingness_flags": bool(args.include_missingness_flags),
            "include_shortcut_flags": bool(args.include_shortcut_flags),
            "B_num_cols": list(artB["num_cols"]),
            "B_bool_cols": list(artB["bool_cols"]),
            "B_fill_values": {k: float(v) for k, v in fillB.items()},
            "B_scaler_mean": scB.mean_.tolist() if hasattr(scB, "mean_") else [],
            "B_scaler_scale": scB.scale_.tolist() if hasattr(scB, "scale_") else [],
            "kernel_kind": str(args.kernel),
            "alpha": float(args.alpha),
            "restarts": int(args.restarts),
            "topk": int(args.topk),
            "k_gate": 0,
            "k_reg": int(args.topk),
            "ignored_input_k_gate": int(getattr(args, "k_gate", 0)) if getattr(args, "gate_frac", None) is None else int(round(float(args.gate_frac) * int(args.topk))),
            "reg_train_mode": "unified_delta_all_finite_pbe",
            "blacklist": sorted(list(blacklist)),
        }
        base.dump_json(outdir / "models" / f"preproc_cycle{int(args.cycle)}.json", pre)

    if args.dump_model:
        if joblib is None:
            base.log("[model] WARNING: joblib not available; skipping --dump-model.")
        else:
            base.safe_mkdir(outdir / "models")
            payload = {
                "cycle": int(args.cycle),
                "mode": "unified_nogate_delta_gpr",
                "gate_classifier": None,
                "classifier": None,
                "gpr": gpr,
                "gpr_pbe0": None,
                "reg0_mode": "disabled_unified_nogate",
                "kernel_": str(getattr(gpr, "kernel_", "")),
                "clf_feature_names": [],
                "reg_feature_names": list(artB["num_cols"]) + list(artB["bool_cols"]),
                "clf_scaler": None,
                "reg_scaler": artB["scaler"],
                "clf_fill_values": {},
                "reg_fill_values": {k: float(v) for k, v in fillB.items()},
                "unified_nogate": True,
            }
            joblib.dump(payload, outdir / "models" / f"phase2_cycle{int(args.cycle)}.joblib")

    # ------------------------------------------------------------------
    # 10) Acquisition dataframe over all unseen candidates
    # ------------------------------------------------------------------
    seen = base.infer_seen_ids(outdir, args.id_col, seed_lab[args.id_col])
    pool_ids = pool_cand[args.id_col].astype(str).reset_index(drop=True)
    mask_new = ~pool_ids.isin(seen)
    base.log(f"[seen] seed_ids={seed_lab[args.id_col].astype(str).nunique()} seen_total={len(seen)} pool_cand={len(pool_ids)} unseen={int(mask_new.sum())}")

    if int(mask_new.sum()) == 0:
        base.log("[select] No unseen items left after filtering. Exiting.")
        return

    unseen_idx = np.where(mask_new.values)[0]

    df_acq = pd.DataFrame({
        args.id_col: pool_ids.iloc[unseen_idx].values.astype(str),
        "formula": pool_cand["formula"].astype(str).iloc[unseen_idx].values if "formula" in pool_cand.columns else [""] * len(unseen_idx),
        "source": pool_cand["source"].astype(str).iloc[unseen_idx].values if "source" in pool_cand.columns else [""] * len(unseen_idx),
        "pbe_gap_effective": pbe_pool_series.iloc[unseen_idx].values,
        "is_pbe0": is_pbe0_pool[unseen_idx].astype(int),
        "p_gate": np.full(len(unseen_idx), np.nan),
        "entropy_gate": np.zeros(len(unseen_idx), dtype=float),
        "gate_margin": np.full(len(unseen_idx), np.nan),
        "u_mad": np.asarray(u_mad, dtype=float)[unseen_idx],
        "u_reg_eff": np.asarray(u_reg_eff, dtype=float)[unseen_idx],
        "sigma_gpr": np.asarray(sigma_raw, dtype=float)[unseen_idx],
        "sigma_eff": np.asarray(sigma_eff, dtype=float)[unseen_idx],
        "mu_delta_gpr": np.asarray(mu_delta_pool, dtype=float)[unseen_idx],
        "hse_pred": np.asarray(hse_pred_pool, dtype=float)[unseen_idx],
        "rho_knn": np.asarray(rho, dtype=float)[unseen_idx],
        "delta_iso": np.asarray(delta_iso, dtype=float)[unseen_idx],
        "rho_iso": np.asarray(rho_iso, dtype=float)[unseen_idx],
        "rrho": base.percentile_rank(rho + micro, higher_is_better=True)[unseen_idx],
        "riso": base.percentile_rank(rho_iso + micro, higher_is_better=True)[unseen_idx],
        "term_sig": np.asarray(w_sig * z_sig, dtype=float)[unseen_idx],
        "term_mad": np.asarray(w_mad * z_mad, dtype=float)[unseen_idx],
        "term_ureg": np.asarray(w_ureg * z_ureg, dtype=float)[unseen_idx],
        "term_rho": np.asarray(w_rho * z_rho, dtype=float)[unseen_idx],
        "term_riso": np.asarray(w_riso * z_riso, dtype=float)[unseen_idx],
        "term_del": np.asarray(-w_del * z_del, dtype=float)[unseen_idx],
        "score_base": np.asarray(score_base, dtype=float)[unseen_idx],
        "score_gate": np.full(len(unseen_idx), np.nan),
        "score_used": np.asarray(score_base, dtype=float)[unseen_idx],
        "pool_row": unseen_idx,
    })
    df_acq["_tie"] = df_acq[args.id_col].map(lambda s: base.stable_hash_int(s, salt=args.acq_salt))
    df_acq = df_acq.sort_values(["score_used", "_tie"], ascending=[False, True]).reset_index(drop=True)

    base.safe_mkdir(outdir / "predictions")
    pred_cols = [
        args.id_col, "formula", "source",
        "pbe_gap_effective", "is_pbe0",
        "p_gate", "entropy_gate", "gate_margin",
        "u_mad", "u_reg_eff",
        "sigma_gpr", "sigma_eff", "mu_delta_gpr",
        "rho_knn", "rho_iso", "delta_iso",
        "rrho", "riso",
        "term_sig", "term_mad", "term_ureg", "term_rho", "term_riso", "term_del",
        "score_base", "score_gate", "score_used",
        "hse_pred", "pool_row",
    ]
    pred_cols = [c for c in pred_cols if c in df_acq.columns]
    df_acq[pred_cols].to_csv(outdir / "predictions" / f"pred_cycle{int(args.cycle)}.csv", index=False)

    # ------------------------------------------------------------------
    # 11) Select topk from full pool, with small Stage-B-style diversification
    # ------------------------------------------------------------------
    topk = int(args.topk)
    if args.acquisition == "random":
        rng = np.random.RandomState(int(args.random_seed) + 1000003 * int(args.cycle))
        chosen_idx = rng.permutation(df_acq.index.to_numpy())[: min(topk, len(df_acq))].tolist()
        sel = df_acq.loc[chosen_idx].copy()
        base.log(f"[select][random][nogate] picked total={len(sel)}/{topk} from full unseen pool")
    else:
        shortlist_mult = int(args.shortlist_mult)
        M = min(len(df_acq), max(topk, shortlist_mult * max(1, topk)))
        short = _shortlist(df_acq, "score_base", M)

        # Optional score floor, kept from the Stage-B policy.
        q_floor = 0.35
        if len(short) > 0:
            sb = pd.to_numeric(short["score_base"], errors="coerce").to_numpy(dtype=float)
            sb_f = sb[np.isfinite(sb)]
            if sb_f.size >= max(10, topk):
                thr = float(np.quantile(sb_f, q_floor))
                short_floor = short[pd.to_numeric(short["score_base"], errors="coerce") >= thr].copy()
                if len(short_floor) >= topk:
                    base.log(f"[nogate] score floor applied: q_floor={q_floor:.2f} thr={thr:.6f} kept={len(short_floor)}/{len(short)}")
                    short = short_floor

        if "_tie" in short.columns:
            sorted_short = short.sort_values(["score_base", "_tie"], ascending=[False, True]).copy()
        else:
            sorted_short = short.sort_values("score_base", ascending=False).copy()

        if topk <= 2:
            k_div = 0
        else:
            k_div = 1
        k_score = max(0, topk - k_div)

        chosen_idx = []
        if k_score > 0:
            chosen_idx += sorted_short.head(min(k_score, len(sorted_short))).index.tolist()

        if k_div > 0:
            taken = set(chosen_idx)
            div_pool = sorted_short.loc[~sorted_short.index.isin(taken)].copy()
            if len(div_pool) > 0:
                M_div = int(min(len(div_pool), max(10, 6 * max(1, topk))))
                div_pool_top = div_pool.head(M_div).copy()
                skip_div_top = 1
                if len(div_pool_top) > skip_div_top:
                    div_pool_top2 = div_pool_top.iloc[skip_div_top:].copy()
                else:
                    div_pool_top2 = div_pool_top

                rows = div_pool_top2["pool_row"].values.astype(int)
                X_pool = X_cand[rows, :]

                if len(taken) > 0:
                    sel_rows = df_acq.loc[list(taken), "pool_row"].values.astype(int)
                    X_sel = X_cand[sel_rows, :]
                    min_dist = np.full(X_pool.shape[0], np.inf, dtype=float)
                    for j in range(X_sel.shape[0]):
                        dd = np.linalg.norm(X_pool - X_sel[j:j + 1], axis=1)
                        min_dist = np.minimum(min_dist, dd)
                else:
                    min_dist = np.zeros(X_pool.shape[0], dtype=float)

                dmin = float(np.min(min_dist)) if min_dist.size else 0.0
                dmax = float(np.max(min_dist)) if min_dist.size else 0.0
                base.log(f"[nogate-div] pool_top={len(div_pool_top2)} dist_spread(min,max)=({dmin:.6g},{dmax:.6g}) skip_div_top={skip_div_top} M_div={M_div}")

                if (dmax - dmin) <= 1e-12:
                    i0 = 0
                else:
                    i0 = int(np.argmax(min_dist))
                chosen_idx += [div_pool_top2.iloc[i0].name]

        # If still short, fill by score from full df_acq.
        if len(chosen_idx) < topk:
            need = topk - len(chosen_idx)
            fill = df_acq.loc[~df_acq.index.isin(set(chosen_idx))].copy()
            fill = _shortlist(fill, "score_base", min(len(fill), need))
            chosen_idx += fill.index.tolist()

        chosen_idx = list(dict.fromkeys(chosen_idx))[:topk]
        sel = df_acq.loc[chosen_idx].copy()
        base.log(f"[select][nogate] picked total={len(sel)}/{topk} from full unseen pool")

    n_sel_pbe0 = int(sel["is_pbe0"].astype(int).sum()) if len(sel) else 0
    base.log(f"[select][nogate] selected PBE0 candidates = {n_sel_pbe0}/{len(sel)}")

    keep_cols = [
        args.id_col, "formula", "source",
        "score_base", "score_gate", "score_used",
        "pbe_gap_effective", "is_pbe0",
        "p_gate", "entropy_gate",
        "u_mad", "u_reg_eff", "sigma_gpr", "sigma_eff",
        "rho_knn", "rho_iso", "delta_iso",
    ]
    keep_cols = [c for c in keep_cols if c in sel.columns]
    sel_out = sel[keep_cols].copy()
    base.write_selection(outdir, args.cycle, sel_out)

    if args.dump_diversity:
        try:
            X_sel = X_cand[sel["pool_row"].values.astype(int), :]
            summ = base.pairwise_distance_summary(X_sel)
        except Exception:
            summ = {"mean": float("nan"), "p10": float("nan"), "p50": float("nan"), "p90": float("nan")}
        summ.update({
            "cycle": int(args.cycle),
            "n_selected": int(len(sel_out)),
            "k_gate": 0,
            "k_reg": int(args.topk),
            "selected_pbe0": int(n_sel_pbe0),
            "mode": "unified_nogate_delta_gpr",
        })
        base.safe_mkdir(outdir / "metrics")
        base.dump_json(outdir / "metrics" / f"diversity_cycle{int(args.cycle)}.json", summ)

    # Update state.json
    st = base.load_state(outdir / "state.json")
    st.setdefault("selected", {})[str(args.cycle)] = sel_out[args.id_col].astype(str).tolist()
    st["seen_ids"] = list(sorted(set(st.get("seen_ids", [])) | set(sel_out[args.id_col].astype(str))))
    st["mode"] = "unified_nogate_delta_gpr"
    base.dump_state(outdir / "state.json", st)

    # ------------------------------------------------------------------
    # 12) Metrics output with original column names retained where possible
    # ------------------------------------------------------------------
    flat = {
        # Gate columns retained but intentionally NaN/0 for compatibility.
        "gate_thr": float("nan"),
        "gate_auc": float("nan"),
        "gate_ap": float("nan"),
        "gate_f1": float("nan"),
        "gate_prec": float("nan"),
        "gate_rec": float("nan"),
        "gate_val_n0": int(zbg_decision.get("n", 0)),
        "gate_pred_pos": int(zbg_decision.get("tp", 0) + zbg_decision.get("fp", 0)),
        "gate_pred_neg": int(zbg_decision.get("tn", 0) + zbg_decision.get("fn", 0)),
        "gate_tn": int(zbg_decision.get("tn", 0)),
        "gate_fp": int(zbg_decision.get("fp", 0)),
        "gate_fn": int(zbg_decision.get("fn", 0)),
        "gate_tp": int(zbg_decision.get("tp", 0)),
        "gate_best_f1": float("nan"),
        "gate_best_thr": float("nan"),
        "gate_best_prec": float("nan"),
        "gate_best_rec": float("nan"),
        "gate_best_pred_pos": 0,

        # No-gate implicit ZBG decision metrics.
        "nogate_zbg_prec": float(zbg_decision.get("precision", float("nan"))),
        "nogate_zbg_rec": float(zbg_decision.get("recall", float("nan"))),
        "nogate_zbg_f1": float(zbg_decision.get("f1", float("nan"))),
        "nogate_zbg_tn": int(zbg_decision.get("tn", 0)),
        "nogate_zbg_fp": int(zbg_decision.get("fp", 0)),
        "nogate_zbg_fn": int(zbg_decision.get("fn", 0)),
        "nogate_zbg_tp": int(zbg_decision.get("tp", 0)),
        "nogate_zbg_n": int(zbg_decision.get("n", 0)),

        # Preserve original delta column names for comparable plotting.
        "delta_val_MAE_nonmetal": float(val_metrics_delta_nonmetal_pbegt.get("MAE", float("nan"))),
        "delta_val_RMSE_nonmetal": float(val_metrics_delta_nonmetal_pbegt.get("RMSE", float("nan"))),
        "delta_val_R2_nonmetal": float(val_metrics_delta_nonmetal_pbegt.get("R2", float("nan"))),
        "delta_val_n_nonmetal": int(val_metrics_delta_nonmetal_pbegt.get("n", 0)),

        # Additional unified-delta metrics.
        "delta_val_MAE_all": float(val_metrics_delta_all.get("MAE", float("nan"))),
        "delta_val_RMSE_all": float(val_metrics_delta_all.get("RMSE", float("nan"))),
        "delta_val_R2_all": float(val_metrics_delta_all.get("R2", float("nan"))),
        "delta_val_n_all": int(val_metrics_delta_all.get("n", 0)),

        "end2end_hse_MAE": float(end2end_metrics.get("MAE", float("nan"))),
        "end2end_hse_RMSE": float(end2end_metrics.get("RMSE", float("nan"))),
        "end2end_hse_R2": float(end2end_metrics.get("R2", float("nan"))),
        "end2end_hse_n": int(end2end_metrics.get("n", 0)),

        "cycle": int(args.cycle),
        "topk": int(args.topk),
        "k_gate": 0,
        "k_reg": int(args.topk),
        "ignored_input_k_gate": int(getattr(args, "k_gate", 0)) if getattr(args, "gate_frac", None) is None else int(round(float(args.gate_frac) * int(args.topk))),
        "selected_pbe0": int(n_sel_pbe0),
        "pbe_zero_thr": float(args.pbe_zero_thr),
        "gate_decision_thr": float("nan"),
        "train_n_total": int(len(seed_train_all)),
        "train_n_nonmetal_used": int(len(seed_train_U)),
        "train_n_unified_used": int(len(seed_train_U)),
        "train_n_unified_pbe0": int((pd.to_numeric(seed_train_U["_pbe_eff"], errors="coerce") <= float(args.pbe_zero_thr)).sum()),
        "train_n_unified_hse_metal_like": int(seed_train_U["_metal_like_truth"].astype(int).sum()),
        "reg_train_mode_fallback": 0,
        "mode_unified_nogate": 1,
    }
    _write_metrics_csv(outdir, args.cycle, flat)

    strat = {
        "cycle": int(args.cycle),
        "mode": "unified_nogate_delta_gpr",
        "gate": {
            "disabled": True,
            "note": "No gate classifier was trained; no hard routing was applied.",
        },
        "stageB": {
            "val_nonmetal_delta": val_metrics_delta_nonmetal_pbegt,
            "val_all_delta": val_metrics_delta_all,
            "reg_train_mode": "unified_delta_all_finite_pbe",
        },
        "end2end": {"val_hse": end2end_metrics},
        "nogate_zbg_decision": zbg_decision,
        "acquisition": {
            "full_pool": True,
            "selected_pbe0": int(n_sel_pbe0),
            "w_sig": float(w_sig),
            "w_mad": float(w_mad),
            "w_ureg": float(w_ureg),
            "alpha_rho": float(args.alpha_rho),
            "beta_iso": float(args.beta_iso),
            "k_gate": 0,
            "k_reg": int(args.topk),
        },
    }
    base.dump_json(outdir / "metrics" / f"metrics_cycle{int(args.cycle)}.json", strat)

    base.log(f"[select] cycle {args.cycle}: selected={len(sel_out)} k_gate=0 k_reg={int(args.topk)} selected_pbe0={n_sel_pbe0}")
    base.log(f"[select] wrote selection -> {outdir / 'selections' / f'selected_cycle{int(args.cycle)}.csv'}")


# -----------------------------------------------------------------------------
# CLI: exact same keywords as active_learning_gpr.py
# -----------------------------------------------------------------------------

def build_cli() -> argparse.ArgumentParser:
    """
    Reuse the original parser so al_loop_v3.sh keyword arguments remain valid.
    """
    return base.build_cli()


def main() -> None:
    ap = build_cli()
    args = ap.parse_args()

    if args.cmd == "select":
        # Ablation-specific default: this unified no-gate script is the designated
        # topk=7 benchmark unless the caller explicitly overrides --topk on the
        # command line. active_learning_gpr.py's own --topk default (5) is
        # untouched, since this override only applies inside this script.
        if "--topk" not in sys.argv:
            args.topk = NOGATE_DEFAULT_TOPK
        cmd_select(args)
    elif args.cmd == "update":
        # Preserve the original update behavior exactly.
        base.cmd_update(args)
    else:  # pragma: no cover
        raise RuntimeError("Unknown command")


if __name__ == "__main__":
    main()
