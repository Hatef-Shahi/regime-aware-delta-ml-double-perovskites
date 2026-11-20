"""Additional composition-grouped cross-validation (comments 3.11 and 3.13; ESI Table S8).
Same data, preprocessing, Delta-GPR and composition folds (3 repeats x GroupKFold(5),
group permutation seeds 200+rep) as revision/scripts/cv_baselines.py.
  full     : Delta-GPR, all 107 features (reference, should reproduce ~0.113 eV)
  top15/30 : Delta-GPR refitted on the 15/30 shortest ARD length scales of the
             full model fitted on the same training fold (selection inside the fold)
  direct+pbe : GPR on the HSE06 gap with the PBE gap as an extra standardised input
  delta+pbe  : Delta-GPR with the PBE gap as a 108th standardised input
Usage: python cv_extra.py <repo> <outdir>"""
import os
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"): os.environ[v]="1"
import sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
REPO, OUT = sys.argv[1], sys.argv[2]
sys.argv=["x",REPO,OUT,"3"]; sys.path.insert(0,f"{REPO}/revision/scripts")
os.chdir(REPO)
from cv_baselines import d, prep, gpr
from sklearn.model_selection import GroupKFold
from joblib import Parallel, delayed

def fold(rep,k,tr,te):
    Xtr,Xte=prep(tr,te); y=d.delta.values; h=d.hse_gap.values; pb=d.pbe_own.values
    m0,s=pb[tr].mean(),pb[tr].std() or 1.0
    ptr,pte=((pb[tr]-m0)/s)[:,None],((pb[te]-m0)/s)[:,None]
    out={}
    g=gpr(Xtr.shape[1]).fit(Xtr,y[tr]); out["full"]=g.predict(Xte)+pb[te]
    ls=g.kernel_.k1.k2.length_scale
    for n in (15,30):
        idx=np.argsort(ls)[:n]
        out[f"top{n}"]=gpr(n).fit(Xtr[:,idx],y[tr]).predict(Xte[:,idx])+pb[te]
    out["direct+pbe"]=gpr(Xtr.shape[1]+1).fit(np.hstack([Xtr,ptr]),h[tr]).predict(np.hstack([Xte,pte]))
    out["delta+pbe"]=gpr(Xtr.shape[1]+1).fit(np.hstack([Xtr,ptr]),y[tr]).predict(np.hstack([Xte,pte]))+pb[te]
    rows=[]
    for m,p in out.items():
        for i,j in enumerate(te): rows.append(dict(rep=rep,fold=k,model=m,structure_id=d.structure_id[j],hse_true=h[j],hse_pred=p[i]))
    print("done",rep,k,flush=True); return rows

jobs=[]
for rep in range(3):
    rng=np.random.RandomState(200+rep); ug=d.comp.unique(); perm=dict(zip(ug,rng.permutation(len(ug))))
    for k,(tr,te) in enumerate(GroupKFold(5).split(d,groups=d.comp.map(perm).values)): jobs.append((rep,k,tr,te))
res=Parallel(n_jobs=2)(delayed(fold)(*j) for j in jobs)
r=pd.DataFrame([x for rr in res for x in rr]); r["err"]=r.hse_pred-r.hse_true
r.to_csv(f"{OUT}/cv_extra_predictions.csv",index=False)
per=r.groupby(["model","rep"]).apply(lambda x: pd.Series(dict(MAE=x.err.abs().mean(),RMSE=np.sqrt((x.err**2).mean())))).reset_index()
summ=per.groupby("model").agg(MAE=("MAE","mean"),MAE_sd=("MAE","std"),RMSE=("RMSE","mean")).reset_index()
summ.to_csv(f"{OUT}/cv_extra_summary.csv",index=False); print(summ.round(4).to_string(index=False))
