"""Comment 3.13 (main Section 2.3): Delta (HSE06 - PBE) regression accuracy of the cycle-24 Delta-GPR models of arm 01,
on a like-for-like pooled basis: each replicate's own cycle-24 model predicts its own
validation set and the non-ZBG held-out test structures, and the per-run predictions are
pooled over the five replicates before computing MAE, R2 and the variance of the true Delta.
Non-ZBG routing uses the true PBE gap (> 0.05 eV). Variances are sample variances (ddof 1).
Usage: python delta_r2.py <repo> [out.csv]"""
import sys, glob, pandas as pd
R = sys.argv[1]; OUT = sys.argv[2] if len(sys.argv) > 2 else "delta_r2.csv"
A = f"{R}/OFFICIAL-RUNS/01_gate_aware_delta_ml_dos_on"
r2 = lambda y, p: 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
V = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{A}/al*/metrics/val_pred_cycle24.csv"))])
V = V[(V.pbe_gap_effective > 0.05) & (V.label_status == "SUCCESS")]
T = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{A}/al*/test_eval/test_pred_cycle24.csv"))])
T = T[T.is_PBE_ZBG == 0]
sets = [("validation, pooled", V), ("test, pooled", T),
        ("test without Ba2BiSbO6, pooled", T[~T.structure_id.str.contains("Ba2BiSbO6")])]
rows = [dict(set=n, n=len(x), MAE=(x.delta_true - x.delta_pred_stageB).abs().mean(),
             R2=r2(x.delta_true, x.delta_pred_stageB), var_delta_true=x.delta_true.var(ddof=1))
        for n, x in sets]
out = pd.DataFrame(rows); out.to_csv(OUT, index=False); print(out.round(4).to_string(index=False))
