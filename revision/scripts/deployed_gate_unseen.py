"""Deployed cycle-24 gates applied to labelled baseline-ZBG structures that each replicate
never labelled (R3.8, R3.9; Table S13).

For every replicate of arm 01 (DOS-on) and arm 05 (DOS-off), the saved cycle-24 gate scores
every labelled baseline-ZBG pool structure (PBE gap <= 0.05 eV) that is not in that
replicate's Seed_labels_cycle23.updated.csv, the 268-row ledger (validation rows included)
that trained the cycle-24 models. Labels are the union of the Seed_labels ledgers of all
official runs. A structure is "rescuable" when its HSE06 reference gap is > 0.05 eV.

A sensitivity line is also printed and written for a stricter rule: the structure appears in
none of the replicate's Seed_labels ledgers at any cycle (for runs that continued past
cycle 24, this also excludes structures labelled after cycle 23).

Usage (from the repository root):
    python revision/scripts/deployed_gate_unseen.py . revision/results
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"  # single-threaded BLAS so that repeated runs give identical output
import sys, glob, json
from pathlib import Path
import numpy as np, pandas as pd, joblib
from scipy.stats import beta
from sklearn.metrics import roc_auc_score

REPO, OUT = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(REPO / "scripts" / "evaluation"))
from evaluate_test_cycle import build_X, p_gate_from_model, add_dos_features_if_needed, as_id  # noqa: E402

THR, THETA = 0.05, 0.16
pool = pd.read_csv(REPO / "data" / "trainpool_ml_augmented.csv")
pool["structure_id"] = as_id(pool.structure_id)
pool = pool[pool.split_day0 != "test"]
L = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(REPO / "OFFICIAL-RUNS" / "*" / "*" / "Seed_labels_cycle*.updated.csv")))])
L = L[L.label_status.isin(["SUCCESS", "METAL_ASSUMED"])].drop_duplicates("structure_id")
L["hse"] = np.where(L.label_status == "METAL_ASSUMED", 0.0, L.hse_gap)
zbg = L[L.pbe_gap <= THR][["structure_id", "hse"]]
zbg["rescuable"] = (zbg.hse > THR).astype(int)

rows = []
for arm in ["01_gate_aware_delta_ml_dos_on", "05_dos_off_sensitivity"]:
    for run_dir in sorted((REPO / "OFFICIAL-RUNS" / arm).glob("al*")):
        led23 = set(as_id(pd.read_csv(run_dir / "Seed_labels_cycle23.updated.csv").structure_id))
        own_any = set()
        for f in run_dir.glob("Seed_labels_cycle*.updated.csv"):
            own_any |= set(as_id(pd.read_csv(f).structure_id))
        cand = zbg[~zbg.structure_id.isin(led23)].copy()
        cand["unseen_all_cycles"] = ~cand.structure_id.isin(own_any)
        df = cand.merge(pool, on="structure_id", how="inner")
        payload = joblib.load(run_dir / "models" / "phase2_cycle24.joblib")
        pre = json.loads((run_dir / "models" / "preproc_cycle24.json").read_text())
        fillA = payload.get("clf_fill_values", pre.get("A_fill_values", {}))
        A_num, A_bool = list(fillA.keys()), list(pre.get("A_bool_cols", []))
        need = [c for c in A_num + A_bool if str(c).startswith("_dos_")]
        df = add_dos_features_if_needed(df, "structure_id", need, REPO / "data" / "dos_from_output.csv")
        Xa = build_X(df, A_num, A_bool, fillA, payload["clf_scaler"])
        p = p_gate_from_model(payload.get("gate_classifier", payload.get("classifier")), Xa)
        for sid, r, strict, pg in zip(df.structure_id, df.rescuable, df.unseen_all_cycles, p):
            rows.append(dict(arm=arm[:2], run=run_dir.name, structure_id=sid, rescuable=int(r),
                             unseen_all_cycles=bool(strict), p_gate=float(pg)))

R = pd.DataFrame(rows)
R.to_csv(OUT / "deployed_gate_unseen.csv", index=False)


def cp(k, n):
    lo = beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
    return lo, hi


S = []
for (rule, arm), g in pd.concat([R.assign(rule="not_in_cycle23_ledger"),
                                 R[R.unseen_all_cycles].assign(rule="not_in_any_ledger")]).groupby(["rule", "arm"], sort=False):
    res, met = g[g.rescuable == 1], g[g.rescuable == 0]
    k1, n1 = int((res.p_gate >= THETA).sum()), len(res)
    k0, n0 = int((met.p_gate < THETA).sum()), len(met)
    S.append(dict(rule=rule, arm=arm, n_pairs=len(g), recall_k=k1, recall_n=n1, recall=k1 / n1, recall_lo=cp(k1, n1)[0], recall_hi=cp(k1, n1)[1],
                  blocked_k=k0, blocked_n=n0, blocked=k0 / n0, blocked_lo=cp(k0, n0)[0], blocked_hi=cp(k0, n0)[1],
                  auc_pooled=roc_auc_score(g.rescuable, g.p_gate),
                  auc_per_run_min=min(roc_auc_score(x.rescuable, x.p_gate) for _, x in g.groupby("run")),
                  auc_per_run_max=max(roc_auc_score(x.rescuable, x.p_gate) for _, x in g.groupby("run")),
                  unique_rescuable=res.structure_id.nunique(), unique_metal=met.structure_id.nunique()))
S = pd.DataFrame(S)
S.to_csv(OUT / "deployed_gate_unseen_summary.csv", index=False)
print(S.round(3).to_string(index=False))
