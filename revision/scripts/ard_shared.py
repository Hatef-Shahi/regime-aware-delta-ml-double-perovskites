"""Comments 2.2 and 3.13 (ESI Section 3): ARD length scales of the five deployed cycle-24 Delta-GPRs (arm 01); count the
features with length scale < 10 in each model and how many are shared.
Usage: python ard_shared.py <repo> <outdir>"""
import os
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"): os.environ[v]="1"
import sys,json,joblib,numpy as np,pandas as pd,warnings
from pathlib import Path
from collections import Counter
warnings.filterwarnings("ignore")
REPO,OUT=Path(sys.argv[1]),Path(sys.argv[2]); rows=[]; sets={}
for r in range(1,6):
    rd=REPO/f"OFFICIAL-RUNS/01_gate_aware_delta_ml_dos_on/al{r}"
    P=joblib.load(rd/"models/phase2_cycle24.joblib"); pre=json.loads((rd/"models/preproc_cycle24.json").read_text())
    fillB=P.get("reg_fill_values",pre.get("B_fill_values",{})); names=list(fillB.keys())+list(pre.get("B_bool_cols",[]))
    k=P["gpr"].kernel_; ls=None
    for p,v in k.get_params().items():
        if p.endswith("length_scale") and np.ndim(v)==1 and len(v)>1: ls=np.asarray(v)
    assert ls is not None and len(ls)==len(names),(len(ls) if ls is not None else None,len(names))
    for n,l in zip(names,ls): rows.append(dict(run=f"al{r}",feature=n,length_scale=l))
    sets[r]=set(np.array(names)[ls<10]); print(f"al{r}: {len(names)} features, {int((ls<10).sum())} <10, {int((ls<100).sum())} <100, {int((ls>=900).sum())} >=900",flush=True)
pd.DataFrame(rows).to_csv(OUT/"ard_cycle24_models.csv",index=False)
cnt=Counter(f for s in sets.values() for f in s)
print("in all 5:",sum(c==5 for c in cnt.values()),"in >=4:",sum(c>=4 for c in cnt.values()),"in >=3:",sum(c>=3 for c in cnt.values()),"union:",len(cnt))
print("shared by all 5:",sorted(f for f,c in cnt.items() if c==5))
full=pd.read_csv(REPO/"revision/results/ard_length_scales_full.csv"); f10=set(full[full.ard_length_scale<10].feature)
print("of the 15 (<10) in the full 321-structure fit, present in all 5 / >=3 models:",sum(cnt.get(f,0)==5 for f in f10),sum(cnt.get(f,0)>=3 for f in f10))
