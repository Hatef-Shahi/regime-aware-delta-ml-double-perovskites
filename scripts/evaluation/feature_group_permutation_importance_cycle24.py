#!/usr/bin/env python3
"""Feature-group permutation importance of the cycle-24 models (ESI Table S7, Fig. S10).

For each arm-01 replicate, the saved cycle-24 model is evaluated on the 30 held-out test
structures. Each feature group is permuted 50 times; the gate classifier is scored by the
decrease in ROC-AUC on the baseline-ZBG test subset and the Delta-GPR by the increase in
Delta-MAE on the non-ZBG test subset.

Usage (from the repository root):
    python scripts/evaluation/feature_group_permutation_importance_cycle24.py
Outputs: results/tableS7_figS10_permutation/feature_group_permutation_importance_{per_run,summary}.csv
plus simple per-branch bar charts (the published Fig. S10 is drawn by
scripts/figures/plot_feature_group_permutation_importance.py).
"""

import argparse
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"
import importlib.util
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, roc_auc_score


def load_eval_module(path):
    spec = importlib.util.spec_from_file_location("evalmod", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def group_feature(c: str) -> str:
    cl = str(c).lower()

    if cl.startswith("_dos_"):
        return "DOS near Fermi level"

    if c in ["efermi", "is_gap_direct"]:
        return "Baseline PBE electronic"

    if any(k in cl for k in ["spg_number", "volume", "density", "lattice"]):
        return "Global structure / symmetry"

    if any(k in cl for k in [
        "tolerance", "ra", "rb1", "rb2", "rbavg", "delta_rb",
        "radii", "radius", "mu", "oxidation", "ox_"
    ]):
        return "Oxidation / ionic-radius descriptors"

    if any(k in cl for k in [
        "atomic_mass", "en_pauling", "ea_", "ie1",
        "mulliken", "hardness", "dchi"
    ]):
        return "Elemental chemistry"

    if any(k in cl for k in [
        "bond", "bav", "bob", "tilt", "qe_", "cn_o", "coord"
    ]):
        return "Local coordination / distortion"

    return "Other retained descriptor"


def actual_feature_column(df, name):
    for col in (f"{name}_pool", name, f"{name}_test"):
        if col in df.columns:
            return col
    return None


def permute_feature_group(df, feature_names, rng):
    out = df.copy()
    for f in feature_names:
        col = actual_feature_column(out, f)
        if col is None:
            continue
        vals = out[col].to_numpy().copy()
        rng.shuffle(vals)
        out[col] = vals
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", default="OFFICIAL-RUNS/01_gate_aware_delta_ml_dos_on")
    ap.add_argument("--pool", default="data/trainpool_ml_augmented.csv")
    ap.add_argument("--test", default="data/test_labels_labeled.csv")
    ap.add_argument("--dos", default="data/dos_from_output.csv")
    ap.add_argument("--cycle", type=int, default=24)
    ap.add_argument("--evaluator", default="scripts/evaluation/evaluate_test_cycle.py")
    ap.add_argument("--runs", nargs="+", default=["al1", "al2", "al3", "al4", "al5"])
    ap.add_argument("--n-repeats", type=int, default=50)
    ap.add_argument("--out-dir", default="results/tableS7_figS10_permutation")
    args = ap.parse_args()

    evalmod = load_eval_module(Path(args.evaluator))

    run_root = Path(args.run_root)
    pool = pd.read_csv(args.pool)
    test = pd.read_csv(args.test)
    dos_path = Path(args.dos)

    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []

    for run in args.runs:
        run_dir = run_root / run
        model_path = run_dir / "models" / f"phase2_cycle{args.cycle}.joblib"
        preproc_path = run_dir / "models" / f"preproc_cycle{args.cycle}.json"

        if not model_path.exists():
            raise FileNotFoundError(model_path)
        if not preproc_path.exists():
            raise FileNotFoundError(preproc_path)

        payload = joblib.load(model_path)
        pre = json.loads(preproc_path.read_text())

        id_col = pre["id_col"]
        hse_col = pre["hse_col"]
        pbe_col = pre["pool_pbe_col"]
        pbe_zero_thr = float(pre.get("pbe_zero_thr", 0.05))
        metal_thr_ev = float(pre.get("metal_thr_ev", 0.05))
        target_mode = str(payload.get("target_mode", pre.get("target_mode", "delta")))

        p = pool.copy()
        t = test.copy()
        p[id_col] = evalmod.as_id(p[id_col])
        t[id_col] = evalmod.as_id(t[id_col])

        df = t.merge(p, on=id_col, how="inner", suffixes=("_test", "_pool"))
        if len(df) == 0:
            raise RuntimeError(f"{run}: test/pool merge produced 0 rows")

        fillA = payload.get("clf_fill_values", pre.get("A_fill_values", {}))
        fillB = payload.get("reg_fill_values", pre.get("B_fill_values", {}))

        A_num = list(fillA.keys())
        B_num = list(fillB.keys())
        A_bool = list(pre.get("A_bool_cols", []))
        B_bool = list(pre.get("B_bool_cols", []))

        need_dos = [c for c in (A_num + A_bool + B_num + B_bool) if str(c).startswith("_dos_")]
        df = evalmod.add_dos_features_if_needed(df, id_col, need_dos, dos_path)

        pbe_test = pd.to_numeric(evalmod.pick_test_label(df, pbe_col), errors="coerce").to_numpy(float)
        pbe_pool = pd.to_numeric(evalmod.pick_feature(df, pbe_col), errors="coerce").to_numpy(float)
        pbe = np.where(np.isfinite(pbe_test), pbe_test, pbe_pool)

        hse_true = pd.to_numeric(evalmod.pick_test_label(df, hse_col), errors="coerce").to_numpy(float)
        delta_true = hse_true - pbe

        is_pbe_zbg = np.isfinite(pbe) & (pbe <= pbe_zero_thr)
        is_non_zbg = np.isfinite(pbe) & (pbe > pbe_zero_thr)

        Xa = evalmod.build_X(df, A_num, A_bool, fillA, payload["clf_scaler"])
        Xb = evalmod.build_X(df, B_num, B_bool, fillB, payload["reg_scaler"])

        gate_model = payload.get("gate_classifier", payload.get("classifier", None))
        p_gate = evalmod.p_gate_from_model(gate_model, Xa)

        gpr = payload["gpr"]
        mu_stageB = np.asarray(gpr.predict(Xb), float)

        if target_mode == "delta":
            delta_pred_stageB = mu_stageB
        elif target_mode == "direct_hse":
            delta_pred_stageB = mu_stageB - pbe
        else:
            raise ValueError(f"Unknown target_mode: {target_mode}")

        # Gate classifier importance on PBE-ZBG test subset.
        if int(is_pbe_zbg.sum()) > 0:
            y_gate = (hse_true[is_pbe_zbg] > metal_thr_ev).astype(int)

            if len(np.unique(y_gate)) == 2:
                auc0 = roc_auc_score(y_gate, p_gate[is_pbe_zbg])

                clf_groups = {}
                for c in A_num + A_bool:
                    clf_groups.setdefault(group_feature(c), []).append(c)

                for group, cols in clf_groups.items():
                    vals = []
                    for rep in range(args.n_repeats):
                        rng = np.random.default_rng(10000 + rep)
                        dfp = permute_feature_group(df, cols, rng)
                        Xap = evalmod.build_X(dfp, A_num, A_bool, fillA, payload["clf_scaler"])
                        pp = evalmod.p_gate_from_model(gate_model, Xap)
                        aucp = roc_auc_score(y_gate, pp[is_pbe_zbg])
                        vals.append(auc0 - aucp)

                    rows.append({
                        "run": run,
                        "branch": "Gate classifier",
                        "metric": "AUC decrease",
                        "feature_group": group,
                        "baseline_metric": auc0,
                        "importance_mean": float(np.mean(vals)),
                        "importance_std_within_run": float(np.std(vals, ddof=1)),
                        "n_eval": int(is_pbe_zbg.sum()),
                        "n_features": int(len(cols)),
                    })
            else:
                print(f"[WARN] {run}: PBE-ZBG test subset has one class; skipping gate AUC.")

        # Δ-regression importance on non-ZBG test subset.
        if int(is_non_zbg.sum()) > 0:
            m = is_non_zbg & np.isfinite(delta_true) & np.isfinite(delta_pred_stageB)
            mae0 = mean_absolute_error(delta_true[m], delta_pred_stageB[m])

            reg_groups = {}
            for c in B_num + B_bool:
                reg_groups.setdefault(group_feature(c), []).append(c)

            for group, cols in reg_groups.items():
                vals = []
                for rep in range(args.n_repeats):
                    rng = np.random.default_rng(20000 + rep)
                    dfp = permute_feature_group(df, cols, rng)
                    Xbp = evalmod.build_X(dfp, B_num, B_bool, fillB, payload["reg_scaler"])
                    mup = np.asarray(gpr.predict(Xbp), float)

                    if target_mode == "delta":
                        deltap = mup
                    else:
                        deltap = mup - pbe

                    maep = mean_absolute_error(delta_true[m], deltap[m])
                    vals.append(maep - mae0)

                rows.append({
                    "run": run,
                    "branch": "Δ-regression",
                    "metric": "Δ-MAE increase / eV",
                    "feature_group": group,
                    "baseline_metric": mae0,
                    "importance_mean": float(np.mean(vals)),
                    "importance_std_within_run": float(np.std(vals, ddof=1)),
                    "n_eval": int(m.sum()),
                    "n_features": int(len(cols)),
                })

        print(f"[OK] {run}: completed feature-group permutation")

    imp = pd.DataFrame(rows)
    if imp.empty:
        raise RuntimeError("No importance rows generated.")

    imp.to_csv(outdir / "feature_group_permutation_importance_per_run.csv", index=False)

    summary = (
        imp.groupby(["branch", "metric", "feature_group"], as_index=False)
        .agg(
            importance_mean=("importance_mean", "mean"),
            importance_sd_across_runs=("importance_mean", "std"),
            n_features=("n_features", "mean"),
            n_eval=("n_eval", "mean"),
            baseline_metric=("baseline_metric", "mean"),
        )
    )

    summary["importance_sd_across_runs"] = summary["importance_sd_across_runs"].fillna(0.0)
    summary.to_csv(outdir / "feature_group_permutation_importance_summary.csv", index=False)

    for branch in ["Gate classifier", "Δ-regression"]:
        sub = summary[summary["branch"] == branch].copy()
        if sub.empty:
            continue

        sub = sub.sort_values("importance_mean", ascending=True)

        fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=300)
        ax.barh(
            sub["feature_group"],
            sub["importance_mean"],
            xerr=sub["importance_sd_across_runs"],
            capsize=3,
        )

        if branch == "Gate classifier":
            ax.set_xlabel("AUC decrease after feature-group permutation")
            ax.set_title("Gate classifier: feature-group importance")
        else:
            ax.set_xlabel("Increase in Δ-MAE after feature-group permutation / eV")
            ax.set_title("Δ-regression: feature-group importance")

        ax.axvline(0.0, linewidth=1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()

        safe = branch.lower().replace(" ", "_").replace("δ", "delta").replace("-", "_")
        for ext in ["png", "pdf", "svg"]:
            fig.savefig(outdir / f"fig_{safe}_feature_group_permutation_importance.{ext}",
                        dpi=600, bbox_inches="tight")

    print("[OK] wrote:", outdir)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
