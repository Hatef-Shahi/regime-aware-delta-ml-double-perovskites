"""tau0 sensitivity (R3.7, Table S12).

End-to-end replica of the deployed three-stage predictor under composition-grouped CV
on every labelled pool structure (SUCCESS + METAL_ASSUMED, n=429):
  Stage A gate: calibrated (sigmoid, cv=3) LogisticRegression, balanced x PBE-focus sample
          weights (lam 0.20, scale 0.10 eV above tau0), desaturation eps 0.02, hard route
          p_gate < 0.16 -> HSE_pred = 0. Two gate feature sets are evaluated: arm 05 (DOS-off)
          and arm 01 (DOS-on, needs data/dos_from_output.csv).
  Stage C: GPR, build_kernel('matern') with optimizer=None, alpha 0.10, normalize_y, on rescued
          training rows (PBE <= tau0, HSE > 0.05).
  Stage B: Delta-GPR (paper settings) on SUCCESS rows with PBE > tau0.
Metal definition fixed at HSE <= 0.05 eV (metal_thr_ev); only the PBE routing threshold tau0 varies.

Usage (from the repository root):
    python revision/scripts/tau_sensitivity.py . revision/results 2 data/dos_from_output.csv
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"  # single-threaded BLAS so that repeated runs give identical output
import re, sys, json, glob, warnings
from collections import Counter
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern, WhiteKernel
from joblib import Parallel, delayed
warnings.filterwarnings("ignore")

REPO, OUT = sys.argv[1], sys.argv[2]
NREP = int(sys.argv[3]) if len(sys.argv) > 3 else 2
TAUS = [0.01, 0.05, 0.10, 0.20, 0.30]
METAL_THR, THETA = 0.05, 0.16
R1 = f"{REPO}/OFFICIAL-RUNS/01_gate_aware_delta_ml_dos_on/al1"
pB = json.load(open(f"{R1}/models/preproc_cycle24.json"))
pA = json.load(open(f"{REPO}/OFFICIAL-RUNS/05_dos_off_sensitivity/al1/models/preproc_cycle24.json"))
NUMB = [c for c in pB["B_num_cols"] if c in pB["B_fill_values"]]; BOOLB = pB["B_bool_cols"]
GATES = {"arm05_dos_off": ([c for c in pA["A_num_cols"] if c in pA["A_fill_values"]], pA["A_bool_cols"]),
         "arm01_dos_on": ([c for c in pB["A_num_cols"] if c in pB["A_fill_values"]], pB["A_bool_cols"])}
DOS_CSV = sys.argv[4] if len(sys.argv) > 4 else f"{REPO}/data/dos_from_output.csv"

pool = pd.read_csv(f"{R1}/pool_trainval.csv")
# DOS-derived gate features, exactly as active_learning_gpr.py _merge_dos + derivations
D = pd.read_csv(DOS_CSV)
pool = pool.merge(D[["structure_id", "dosEf", "maxDOS_between", "gap", "dist_ratio", "d_below", "d_above", "is_gap"]],
                  on="structure_id", how="left", validate="many_to_one", suffixes=("", "_dosraw"))
gcol = "gap_dosraw" if "gap_dosraw" in pool.columns else "gap"
pool["_dos_log_dosEf"] = np.log1p(np.clip(pd.to_numeric(pool.dosEf, errors="coerce"), 0, None))
pool["_dos_log_maxDOS_between"] = np.log1p(np.clip(pd.to_numeric(pool.maxDOS_between, errors="coerce"), 0, None))
pool["_dos_log_gap"] = np.log1p(np.clip(pd.to_numeric(pool[gcol], errors="coerce"), 0, None))
pool["_dos_abs_log_dist_ratio"] = np.abs(np.log(np.clip(pd.to_numeric(pool.dist_ratio, errors="coerce"), 1e-6, 1e6)))
pool["_dos_d_below"] = pd.to_numeric(pool.d_below, errors="coerce")
pool["_dos_d_above"] = pd.to_numeric(pool.d_above, errors="coerce")
pool["_dos_is_gap"] = pd.to_numeric(pool.is_gap, errors="coerce").fillna(0).astype(int)
L = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{REPO}/OFFICIAL-RUNS/*/al*/labels_store.csv"))])
L = L[L.label_status.isin(["SUCCESS", "METAL_ASSUMED"])].drop_duplicates("structure_id")
L = L[["structure_id", "hse_gap", "pbe_gap", "label_status"]].rename(columns={"pbe_gap": "pbe_own", "hse_gap": "hse"})
d = pool.drop(columns=[c for c in ["hse_gap"] if c in pool.columns]).merge(L, on="structure_id").reset_index(drop=True)
d.loc[d.label_status == "METAL_ASSUMED", "hse"] = 0.0
d["metal"] = ((d.hse <= METAL_THR) | (d.label_status == "METAL_ASSUMED")).astype(int)
succ = (d.label_status == "SUCCESS").values


def comp_key(f):
    c = Counter()
    for el, n in re.findall(r"([A-Z][a-z]?)(\d*\.?\d*)", str(f)):
        c[el] += float(n) if n else 1.0
    return "".join(f"{e}{c[e]:g}" for e in sorted(c))


d["comp"] = d.formula.map(comp_key)
GX = {}
for gname, (na, ba) in GATES.items():
    missing = [c for c in na + ba if c not in d.columns]
    assert not missing, (gname, missing)
    GX[gname] = (d[na].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan),
                 d[ba].apply(pd.to_numeric, errors="coerce").fillna(0).astype(float) if ba else pd.DataFrame(index=d.index))
XB = d[NUMB].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
BB = d[BOOLB].apply(pd.to_numeric, errors="coerce").fillna(0).astype(float)
pbe, hse = d.pbe_own.values, d.hse.values
print(f"n={len(d)} labelled ({succ.sum()} SUCCESS, {(~succ).sum()} METAL_ASSUMED), metals={d.metal.sum()}, "
      f"comps={d.comp.nunique()}; gates " + ", ".join(f"{g}: {len(a)}+{len(b)}" for g, (a, b) in GATES.items()) +
      f"; reg feats {len(NUMB)}+{len(BOOLB)}; labelled rows without DOS: {int(d._dos_log_dosEf.isna().sum())} "
      f"(PBE ZBG among them: {int((d._dos_log_dosEf.isna() & (d.pbe_own <= 0.05)).sum())})", flush=True)


def prep(X, Bo, tr, te):
    med = X.iloc[tr].median()
    a = X.iloc[tr].fillna(med); b = X.iloc[te].fillna(med)
    keep = a.columns[a.std() > 0]
    sc = StandardScaler().fit(a[keep])
    return (np.hstack([sc.transform(a[keep]), Bo.iloc[tr].values]), np.hstack([sc.transform(b[keep]), Bo.iloc[te].values]))


def kern(nf):
    return C(1.0, (1e-3, 1e3)) * Matern(length_scale=np.ones(nf), nu=2.5, length_scale_bounds=(1e-3, 1e3)) + WhiteKernel(1e-6)


def run_fold(rep, k, tr, te):
    rows = []
    gpr_cache = {}
    GA = {g: prep(XA_, BA_, tr, te) for g, (XA_, BA_) in GX.items()}
    for tau in TAUS:
        y = d.metal.values[tr]
        dab = np.maximum(pbe[tr] - tau, 0.0)
        sw = compute_sample_weight("balanced", y) * (0.20 + 0.80 * np.exp(-dab / 0.10))
        sw = sw / sw.mean()
        PG = {}
        for g, (Atr, Ate) in GA.items():
            cal = CalibratedClassifierCV(LogisticRegression(max_iter=2000), method="sigmoid", cv=3)
            cal.fit(Atr, y, sample_weight=sw)
            PG[g] = 0.02 + 0.96 * (1.0 - cal.predict_proba(Ate)[:, 1])
        cm = succ[tr] & (pbe[tr] <= tau) & (hse[tr] > METAL_THR)
        trC = tr[cm]
        if len(trC) >= 4:
            Ctr, Cte = prep(XB, BB, trC, te)
            g0 = GaussianProcessRegressor(kern(Ctr.shape[1]), alpha=0.10, optimizer=None, normalize_y=True, random_state=42).fit(Ctr, hse[trC])
            muC = g0.predict(Cte)
        else:
            muC = None
        bm = succ[tr] & (pbe[tr] > tau)
        trB = tr[bm]
        key = tuple(trB)
        if key not in gpr_cache:
            Btr, Bte = prep(XB, BB, trB, te)
            g = GaussianProcessRegressor(kern(Btr.shape[1]), alpha=1e-6, n_restarts_optimizer=5, normalize_y=True, random_state=0)
            g.fit(Btr, hse[trB] - pbe[trB])
            gpr_cache[key] = pbe[te] + g.predict(Bte)
        muB = gpr_cache[key]
        zbg = pbe[te] <= tau
        mu_nm = np.where(zbg, muC if muC is not None else muB, muB)
        for g, p_gate in PG.items():
            routed_metal = zbg & (p_gate < THETA)
            pred = np.where(routed_metal, 0.0, mu_nm)
            for i, j in enumerate(te):
                rows.append(dict(gate=g, rep=rep, fold=k, tau=tau, structure_id=d.structure_id[j], comp=d.comp[j], pbe=pbe[j], hse=hse[j],
                                 metal=d.metal[j], zbg=int(zbg[i]), p_gate=p_gate[i] if zbg[i] else np.nan,
                                 routed_metal=int(routed_metal[i]), pred=pred[i], n_stageC=len(trC), n_stageB=len(trB)))
    return rows


jobs = []
for rep in range(NREP):
    rng = np.random.RandomState(300 + rep)
    ug = d.comp.unique(); perm = dict(zip(ug, rng.permutation(len(ug))))
    for k, (tr, te) in enumerate(GroupKFold(5).split(d, groups=d.comp.map(perm).values)):
        assert not set(d.comp[tr]) & set(d.comp[te])
        jobs.append((rep, k, tr, te))
res = Parallel(n_jobs=2, verbose=5)(delayed(run_fold)(*j) for j in jobs)
r = pd.DataFrame([x for rows in res for x in rows])
r["abs_err"] = (r.pred - r.hse).abs()
r.to_csv(f"{OUT}/tau_predictions.csv", index=False)


def summ(x):
    z = x[x.zbg == 1]
    resc = z[z.metal == 0]; met = z[z.metal == 1]
    return pd.Series(dict(
        n_zbg=len(z) / NREP, n_rescuable=len(resc) / NREP, n_zbg_metal=len(met) / NREP,
        MAE_all=x.abs_err.mean(), MAE_nonzbg=x[x.zbg == 0].abs_err.mean(), MAE_zbg=z.abs_err.mean(),
        MAE_rescued_true=resc.abs_err.mean(),
        gate_recall=(resc.routed_metal == 0).mean(), gate_metal_TNR=(met.routed_metal == 1).mean(),
        recall_k=int((resc.routed_metal == 0).sum()), recall_n=len(resc),
        false_rescue_k=int((met.routed_metal == 0).sum()), false_rescue_n=len(met)))


S = r.groupby(["gate", "tau"]).apply(summ).reset_index()
S.to_csv(f"{OUT}/tau_summary.csv", index=False)
print(S.round(3).to_string(index=False))

pol = pool[["structure_id", "pbe_gap"]].merge(L[["structure_id", "pbe_own"]], on="structure_id", how="left")
pv = pol.pbe_own.fillna(pol.pbe_gap)
print("pool regime counts (n=%d):" % len(pol), {t: int((pv <= t).sum()) for t in TAUS})
