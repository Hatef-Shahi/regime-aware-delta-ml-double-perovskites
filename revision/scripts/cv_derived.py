"""Quantities derived from the cross-validation predictions of cv_baselines.py
(R1.3, R3.2, R3.3; ESI Table S9(a) and the paragraph on composition-grouped cross-validation).

Reads <OUT>/cv_predictions.csv and writes
  cv_family_gpr_delta.csv   Delta-GPR errors by prototype family (MP-derived as one family),
                            three repeats pooled (Table S9(a))
  cv_derived_summary.csv    MAE on compositions with more than one polymorph, median absolute
                            error of the Delta-GPR and of gradient boosting, and the number of
                            composition-grouped folds in which gradient boosting has the lower MAE

Family rule: GLAZER:<formula>_<prototype>.vasp belongs to its prototype family
(Fm-3m, I4/m, R-3, P2_1/c); MP entries and GLAZER entries started from a Materials Project
structure (GLAZER:<formula>_mp-<id>.vasp) are MP-derived.

Usage (from the repository root):
    python revision/scripts/cv_derived.py revision/results
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"  # single-threaded BLAS so that repeated runs give identical output
import re
import sys
import numpy as np
import pandas as pd

OUT = sys.argv[1]
r = pd.read_csv(f"{OUT}/cv_predictions.csv")
g = r[(r.model == "GPR_ARD") & (r.target == "delta")].copy()


def family(sid):
    if sid.startswith("MP:") or re.search(r"_mp-[a-z]+\.vasp$", sid):
        return "MP-derived"
    m = re.search(r"_(Fm-3m|I4-m|R-3|P2_1-c)\.vasp$", sid)
    return {"I4-m": "I4/m", "P2_1-c": "P2_1/c"}.get(m.group(1), m.group(1)) if m else "other"


g["family"] = g.structure_id.map(family)
nrep = g.rep.nunique()
fam = g.groupby(["split", "family"]).agg(n=("abs_err", lambda s: len(s) // nrep), MAE=("abs_err", "mean"),
                                         RMSE=("abs_err", lambda s: float(np.sqrt((s ** 2).mean())))).reset_index()
fam.to_csv(f"{OUT}/cv_family_gpr_delta.csv", index=False)
print(fam.round(3).to_string(index=False))

poly = g.groupby("comp").structure_id.nunique()
multi = set(poly[poly > 1].index)
rows = []
for split in ["random", "composition"]:
    s = g[g.split == split]
    rows.append(dict(quantity=f"GPR delta MAE, compositions with >1 polymorph, {split}",
                     value=s[s.comp.isin(multi)].abs_err.mean(), n=s[s.comp.isin(multi)].structure_id.nunique()))
c = r[(r.split == "composition") & (r.target == "delta")]
for m in ["GPR_ARD", "GBR"]:
    rows.append(dict(quantity=f"{m} delta median absolute error, composition", value=c[c.model == m].abs_err.median(), n=np.nan))
fm = c.groupby(["model", "rep", "fold"]).abs_err.mean().unstack("model")
rows.append(dict(quantity="composition folds with GBR MAE < GPR MAE", value=int((fm.GBR < fm.GPR_ARD).sum()), n=len(fm)))
S = pd.DataFrame(rows)
S.to_csv(f"{OUT}/cv_derived_summary.csv", index=False)
print(S.to_string(index=False))
