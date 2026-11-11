"""R3.14: calibration of the non-ZBG Delta-GPR posterior sigma at cycle 24 (arm 01, al1-al5).
Validation: metrics/val_pred_cycle24.csv (non-ZBG rows, sigma_gpr). Test: test_eval/test_pred_cycle24.csv
(non-ZBG rows, sigma_stageB). Reports coverage, z-score spread, a single variance-scaling factor fitted on
validation and applied unchanged to the test set, ranking quality (Spearman), and the noise-floor
recalibration sigma_cal = sqrt((a*sigma)^2 + b^2) fitted by Gaussian NLL on validation.

Usage (from the repository root):
    python revision/scripts/calibration.py . revision/results
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"  # single-threaded BLAS so that repeated runs give identical output
import sys, numpy as np, pandas as pd
from scipy.stats import norm, spearmanr
from scipy.optimize import minimize

REPO, OUT = sys.argv[1], sys.argv[2]
B = f"{REPO}/OFFICIAL-RUNS/01_gate_aware_delta_ml_dos_on"
V, T = [], []
for r in ["al1", "al2", "al3", "al4", "al5"]:
    v = pd.read_csv(f"{B}/{r}/metrics/val_pred_cycle24.csv")
    v = v[(v.is_pbe0 == 0) & v.delta_true.notna()]
    V.append(pd.DataFrame(dict(run=r, err=v.delta_pred_stageB - v.delta_true, sig=v.sigma_gpr)))
    t = pd.read_csv(f"{B}/{r}/test_eval/test_pred_cycle24.csv")
    t = t[t.is_pbe0 == 0]
    T.append(pd.DataFrame(dict(run=r, sid=t.structure_id, err=t.delta_pred_stageB - t.delta_true, sig=t.sigma_stageB)))
V, T = pd.concat(V), pd.concat(T)


def stats(d, s=1.0, tag=""):
    z = d.err / (s * d.sig)
    lev = np.linspace(0.05, 0.95, 19)
    obs = [(np.abs(z) <= norm.ppf(0.5 + p / 2)).mean() for p in lev]
    return dict(set=tag, scale=s, n=len(d), cov1=(np.abs(z) <= 1).mean(), cov2=(np.abs(z) <= 2).mean(),
                z_rms=np.sqrt((z ** 2).mean()), miscal_area=float(np.mean(np.abs(np.array(obs) - lev))),
                nll=float(np.mean(0.5 * np.log(2 * np.pi * (s * d.sig) ** 2) + 0.5 * z ** 2)),
                spearman=spearmanr(d.sig, d.err.abs()).correlation, spearman_p=spearmanr(d.sig, d.err.abs()).pvalue)


# scale fitted on validation: (a) RMS of z (variance scaling, the NLL-optimal factor); (b) factor giving 68.3 % coverage
zv = V.err / V.sig
s_rms = float(np.sqrt((zv ** 2).mean()))
s_cov = float(np.quantile(np.abs(zv), 0.6827))
rows = [stats(V, 1, "val"), stats(V, s_rms, "val"), stats(V, s_cov, "val"),
        stats(T, 1, "test"), stats(T, s_rms, "test"), stats(T, s_cov, "test")]
lo = []
for r in V.run.unique():
    s = float(np.quantile(np.abs(V[V.run != r].err / V[V.run != r].sig), 0.6827))
    d = V[V.run == r]
    lo.append(dict(run=r, scale=s, cov1_raw=(np.abs(d.err / d.sig) <= 1).mean(), cov1_scaled=(np.abs(d.err / (s * d.sig)) <= 1).mean(), n=len(d)))
R = pd.DataFrame(rows); R.to_csv(f"{OUT}/calibration_summary.csv", index=False)
print(R.round(3).to_string(index=False))
L = pd.DataFrame(lo); print("leave-one-run-out (validation):"); print(L.round(3).to_string(index=False))
print("pooled LORO coverage raw %.3f -> scaled %.3f" % ((L.cov1_raw * L.n).sum() / L.n.sum(), (L.cov1_scaled * L.n).sum() / L.n.sum()))
Tb = T[~T.sid.str.contains("Ba2BiSbO6")]
print("test excl. Ba2BiSbO6:", {k: round(v, 3) if isinstance(v, float) else v for k, v in stats(Tb, s_cov, "test-noBa").items()})
V["tert"] = pd.qcut(V.sig, 3, labels=["low", "mid", "high"])
print(V.groupby("tert").apply(lambda d: pd.Series(dict(n=len(d), mean_sig=d.sig.mean(), mae=d.err.abs().mean(),
      cov1_raw=(np.abs(d.err / d.sig) <= 1).mean(), cov1_scaled=(np.abs(d.err / (s_cov * d.sig)) <= 1).mean()))).round(3))
lev = np.linspace(0.05, 0.95, 19)
cc = pd.DataFrame(dict(expected=lev, **{f"{tag}_{nm}": [(np.abs(d.err / (s * d.sig)) <= norm.ppf(0.5 + p / 2)).mean() for p in lev]
                   for tag, d in [("val", V), ("test", T)] for nm, s in [("raw", 1.0), ("scaled", s_cov)]}))
cc.to_csv(f"{OUT}/calibration_curve.csv", index=False)
print("scale factors: rms %.3f, coverage-matched %.3f" % (s_rms, s_cov))


def nll_ab(p, d):
    a, b = np.exp(p); s2 = (a * d.sig) ** 2 + b ** 2
    return float(np.mean(0.5 * np.log(2 * np.pi * s2) + 0.5 * d.err ** 2 / s2))


res = minimize(nll_ab, x0=np.log([1.0, 0.05]), args=(V,), method="Nelder-Mead")
a, b = np.exp(res.x)


def st2(d, tag):
    se = np.sqrt((a * d.sig) ** 2 + b ** 2); z = d.err / se
    lev = np.linspace(0.05, 0.95, 19); obs = [(np.abs(z) <= norm.ppf(0.5 + p / 2)).mean() for p in lev]
    return dict(set=tag, a=a, b=b, n=len(d), cov1=(np.abs(z) <= 1).mean(), cov2=(np.abs(z) <= 2).mean(), z_rms=np.sqrt((z**2).mean()),
                miscal_area=float(np.mean(np.abs(np.array(obs) - lev))), nll=float(np.mean(0.5*np.log(2*np.pi*se**2)+0.5*z**2)),
                spearman=spearmanr(se, d.err.abs()).correlation)


F = pd.DataFrame([st2(V, "val"), st2(T, "test"), st2(T[~T.sid.str.contains("Ba2BiSbO6")], "test_noBa")])
print("noise-floor recalibration:"); print(F.round(3).to_string(index=False))
lo = []
for r in V.run.unique():
    rr = minimize(nll_ab, x0=np.log([1.0, 0.05]), args=(V[V.run != r],), method="Nelder-Mead"); aa, bb = np.exp(rr.x)
    d = V[V.run == r]; se = np.sqrt((aa*d.sig)**2 + bb**2); lo.append(dict(run=r, a=aa, b=bb, n=len(d), cov1=(np.abs(d.err/se) <= 1).mean(), cov2=(np.abs(d.err/se) <= 2).mean()))
L2 = pd.DataFrame(lo); print(L2.round(3).to_string(index=False))
print("pooled LORO cov1 %.3f cov2 %.3f" % ((L2.cov1*L2.n).sum()/L2.n.sum(), (L2.cov2*L2.n).sum()/L2.n.sum()))
V["se"] = np.sqrt((a*V.sig)**2 + b**2)
print(V.groupby("tert").apply(lambda d: pd.Series(dict(cov1_floor=(np.abs(d.err/d.se) <= 1).mean()))).round(3))
F.to_csv(f"{OUT}/calibration_noisefloor.csv", index=False)
