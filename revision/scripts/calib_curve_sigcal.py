"""Calibration curve with the noise-floor recalibration sigma_cal (Fig. S16 data).

Usage (from the repository root):
    python revision/scripts/calib_curve_sigcal.py . revision/results
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"  # single-threaded BLAS so that repeated runs give identical output
import sys, numpy as np, pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize
REPO, OUT = sys.argv[1], sys.argv[2]
B = f"{REPO}/OFFICIAL-RUNS/01_gate_aware_delta_ml_dos_on"
V, T = [], []
for r in ["al1", "al2", "al3", "al4", "al5"]:
    v = pd.read_csv(f"{B}/{r}/metrics/val_pred_cycle24.csv"); v = v[(v.is_pbe0 == 0) & v.delta_true.notna()]
    V.append(pd.DataFrame(dict(run=r, err=v.delta_pred_stageB - v.delta_true, sig=v.sigma_gpr)))
    t = pd.read_csv(f"{B}/{r}/test_eval/test_pred_cycle24.csv"); t = t[t.is_pbe0 == 0]
    T.append(pd.DataFrame(dict(run=r, sid=t.structure_id, err=t.delta_pred_stageB - t.delta_true, sig=t.sigma_stageB)))
V, T = pd.concat(V), pd.concat(T)
zv = V.err / V.sig; s_cov = float(np.quantile(np.abs(zv), 0.6827))
def nll(p, d):
    a, b = np.exp(p); s2 = (a*d.sig)**2 + b**2
    return float(np.mean(0.5*np.log(2*np.pi*s2) + 0.5*d.err**2/s2))
a, b = np.exp(minimize(nll, np.log([1.0, 0.05]), args=(V,), method="Nelder-Mead").x)
lev = np.linspace(0.05, 0.95, 19)
def curve(d, se): return [(np.abs(d.err/se) <= norm.ppf(0.5+p/2)).mean() for p in lev]
cc = pd.DataFrame(dict(expected=lev,
    val_raw=curve(V, V.sig), val_scaled=curve(V, s_cov*V.sig), val_sigcal=curve(V, np.sqrt((a*V.sig)**2+b**2)),
    test_raw=curve(T, T.sig), test_scaled=curve(T, s_cov*T.sig), test_sigcal=curve(T, np.sqrt((a*T.sig)**2+b**2))))
cc.to_csv(f"{OUT}/calibration_curve_with_sigcal.csv", index=False)
se_v = np.sqrt((a*V.sig)**2+b**2); se_t = np.sqrt((a*T.sig)**2+b**2)
print(f"n val {len(V)} test {len(T)}; s_cov {s_cov:.3f}; a {a:.3f} b {b:.3f}")
for nm,d,se in [("val raw",V,V.sig),("test raw",T,T.sig),("val cal",V,se_v),("test cal",T,se_t)]:
    print(nm, round((np.abs(d.err/se)<=1).mean(),3), round((np.abs(d.err/se)<=2).mean(),3))
