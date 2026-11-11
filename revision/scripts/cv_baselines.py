"""Grouped vs random CV harness for R1.3/R3.2 (composition-grouped validation),
R2.1/R3.11 (standard ML baselines) and R3.3 (space-group-resolved errors).

Data: every pool structure with a SUCCESS HSE06 label in any deposited arm
(labels_store.csv), non-ZBG regime (own PBE > 0.05 eV), which is exactly the
training population of the paper's Delta-GPR (reg_train_mode
nonmetal_pbe_gt_thr_delta). Features: the paper's regression feature set
(B_num_cols with numeric fill values + is_gap_direct), StandardScaler and median
imputation fitted per training fold. Delta-GPR replicates build_kernel('matern'),
alpha 1e-6, restarts 5, normalize_y True.

Usage (from the repository root):
    python revision/scripts/cv_baselines.py . revision/results [n_repeats=3]
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"  # single-threaded BLAS so that repeated runs give identical output
import re, sys, json, glob, time, warnings
from collections import Counter
import numpy as np, pandas as pd
from sklearn.model_selection import KFold, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern, WhiteKernel
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")
REPO = sys.argv[1]
OUT = sys.argv[2]
NREP = int(sys.argv[3]) if len(sys.argv) > 3 else 3
THR = 0.05

pre = json.load(open(f"{REPO}/OFFICIAL-RUNS/01_gate_aware_delta_ml_dos_on/al1/models/preproc_cycle16.json"))
NUM = [c for c in pre["B_num_cols"] if c in pre["B_fill_values"]]
BOOL = pre["B_bool_cols"]

pool = pd.read_csv(f"{REPO}/OFFICIAL-RUNS/01_gate_aware_delta_ml_dos_on/al1/pool_trainval.csv")
L = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{REPO}/OFFICIAL-RUNS/*/al*/labels_store.csv"))])
L = L[L.label_status == "SUCCESS"].drop_duplicates("structure_id")[["structure_id", "hse_gap", "pbe_gap"]]
L = L.rename(columns={"pbe_gap": "pbe_own"})
d = pool.drop(columns=[c for c in ["hse_gap"] if c in pool.columns]).merge(L, on="structure_id")
d = d[d.pbe_own > THR].reset_index(drop=True)
d["delta"] = d.hse_gap - d.pbe_own


def comp_key(f):
    toks = re.findall(r"([A-Z][a-z]?)(\d*\.?\d*)", str(f))
    c = Counter()
    for el, n in toks:
        c[el] += float(n) if n else 1.0
    return "".join(f"{e}{c[e]:g}" for e in sorted(c))


d["comp"] = d.formula.map(comp_key)
X_all = d[NUM].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
B_all = d[BOOL].apply(pd.to_numeric, errors="coerce").fillna(0).astype(float)
print(f"n={len(d)} structures, {d.comp.nunique()} compositions, "
      f"{(d.comp.value_counts() > 1).sum()} compositions with >1 polymorph; features {len(NUM)}+{len(BOOL)}", flush=True)


def prep(tr, te):
    med = X_all.iloc[tr].median()
    Xtr = X_all.iloc[tr].fillna(med); Xte = X_all.iloc[te].fillna(med)
    keep = Xtr.columns[Xtr.std() > 0]
    sc = StandardScaler().fit(Xtr[keep])
    return (np.hstack([sc.transform(Xtr[keep]), B_all.iloc[tr].values]),
            np.hstack([sc.transform(Xte[keep]), B_all.iloc[te].values]))


def gpr(nf):
    k = C(1.0, (1e-3, 1e3)) * Matern(length_scale=np.ones(nf), nu=2.5, length_scale_bounds=(1e-3, 1e3)) + WhiteKernel(1e-6)
    return GaussianProcessRegressor(kernel=k, alpha=1e-6, n_restarts_optimizer=5, normalize_y=True, random_state=0)


def models(nf):
    return {
        "GPR_ARD": lambda: gpr(nf),
        "RF": lambda: RandomForestRegressor(n_estimators=500, min_samples_leaf=2, max_features=0.33, random_state=0, n_jobs=1),
        "GBR": lambda: GradientBoostingRegressor(n_estimators=400, learning_rate=0.03, max_depth=3, subsample=0.8, random_state=0),
        "SVR": lambda: GridSearchCV(SVR(kernel="rbf"), {"C": [1, 3, 10, 30], "epsilon": [0.02, 0.05, 0.1], "gamma": ["scale", 0.003, 0.01]}, cv=3, n_jobs=1),
        "KRR": lambda: GridSearchCV(KernelRidge(kernel="rbf"), {"alpha": [1e-3, 1e-2, 1e-1, 1], "gamma": [0.001, 0.003, 0.01, 0.03]}, cv=3, n_jobs=1),
    }


def run_fold(split, rep, k, tr, te):
    Xtr, Xte = prep(tr, te)
    pbe_tr, pbe_te = d.pbe_own.values[tr], d.pbe_own.values[te]
    rows = []
    for mname, mk in models(Xtr.shape[1]).items():
        for target in ["delta", "direct", "direct+pbe"]:
            if mname == "GPR_ARD" and target == "direct+pbe":
                continue
            t0 = time.time()
            if target == "delta":
                ytr = d.delta.values[tr]; A, B = Xtr, Xte
            elif target == "direct":
                ytr = d.hse_gap.values[tr]; A, B = Xtr, Xte
            else:  # direct HSE with PBE gap supplied as an input feature
                ytr = d.hse_gap.values[tr]
                s = pbe_tr.std() or 1.0; m0 = pbe_tr.mean()
                A = np.hstack([Xtr, ((pbe_tr - m0) / s)[:, None]]); B = np.hstack([Xte, ((pbe_te - m0) / s)[:, None]])
            mdl = mk().fit(A, ytr)
            p = mdl.predict(B)
            hse_pred = p + pbe_te if target == "delta" else p
            for i, j in enumerate(te):
                rows.append(dict(split=split, rep=rep, fold=k, model=mname, target=target,
                                 structure_id=d.structure_id[j], comp=d.comp[j], spg=d.spg_symbol[j],
                                 hse_true=d.hse_gap[j], hse_pred=hse_pred[i], fit_s=time.time() - t0))
    return rows


if __name__ == "__main__":
    jobs = []
    for rep in range(NREP):
        kf = KFold(5, shuffle=True, random_state=100 + rep)
        for k, (tr, te) in enumerate(kf.split(d)):
            jobs.append(("random", rep, k, tr, te))
        # GroupKFold is deterministic; shuffle group labels per repeat
        rng = np.random.RandomState(200 + rep)
        ug = d.comp.unique(); perm = dict(zip(ug, rng.permutation(len(ug))))
        g = d.comp.map(perm).values
        for k, (tr, te) in enumerate(GroupKFold(5).split(d, groups=g)):
            assert not set(d.comp[tr]) & set(d.comp[te])
            jobs.append(("composition", rep, k, tr, te))

    res = Parallel(n_jobs=2, verbose=5)(delayed(run_fold)(*j) for j in jobs)
    r = pd.DataFrame([x for rows in res for x in rows])
    r["abs_err"] = (r.hse_pred - r.hse_true).abs()
    r.drop(columns=["fit_s"]).to_csv(f"{OUT}/cv_predictions.csv", index=False)

    per = r.groupby(["split", "model", "target", "rep"]).apply(
        lambda x: pd.Series(dict(MAE=x.abs_err.mean(), RMSE=np.sqrt(((x.hse_pred - x.hse_true) ** 2).mean()),
                                 R2=1 - ((x.hse_pred - x.hse_true) ** 2).sum() / ((x.hse_true - x.hse_true.mean()) ** 2).sum()))).reset_index()
    summ = per.groupby(["split", "model", "target"]).agg(MAE=("MAE", "mean"), MAE_sd=("MAE", "std"), RMSE=("RMSE", "mean"), R2=("R2", "mean")).reset_index()
    summ.to_csv(f"{OUT}/cv_summary.csv", index=False)
    print(summ.sort_values(["target", "split", "MAE"]).round(3).to_string(index=False))

    sg = r[(r.model == "GPR_ARD") & (r.target == "delta")].groupby(["split", "spg"]).agg(
        n=("abs_err", lambda s: len(s) // NREP), MAE=("abs_err", "mean"),
        RMSE=("abs_err", lambda s: float(np.sqrt((s ** 2).mean())))).reset_index()
    sg.to_csv(f"{OUT}/cv_spacegroup_gpr_delta.csv", index=False)
    print(sg.round(3).to_string(index=False))
