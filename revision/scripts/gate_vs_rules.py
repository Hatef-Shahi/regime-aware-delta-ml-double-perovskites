"""Comment 3.8 (ESI Table S13 note): end-to-end error on the Table S13 unseen baseline-ZBG structures for the deployed
gate, an always-rescue rule and a random gate passing at the deployed gate's pooled rate.
Pairs and gate scores are those of revision/results/deployed_gate_unseen.csv (arm 01,
rule not_in_cycle23_ledger); the rescued-branch (Stage C) prediction comes from the same
saved cycle-24 payload, as in scripts/evaluation/evaluate_test_cycle.py.
Usage: python gate_vs_rules.py <repo> <outdir>"""
import os
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"): os.environ[v]="1"
import sys, glob, json
from pathlib import Path
import numpy as np, pandas as pd, joblib
REPO,OUT=Path(sys.argv[1]),Path(sys.argv[2])
sys.path.insert(0,str(REPO/"scripts"/"evaluation"))
from evaluate_test_cycle import build_X, add_dos_features_if_needed, as_id
THETA=0.16
U=pd.read_csv(REPO/"revision/results/deployed_gate_unseen.csv"); U=U[U.arm.astype(str).str.zfill(2)=="01"]
pool=pd.read_csv(REPO/"data/trainpool_ml_augmented.csv"); pool["structure_id"]=as_id(pool.structure_id)
L=pd.concat([pd.read_csv(f) for f in glob.glob(str(REPO/"OFFICIAL-RUNS/*/*/Seed_labels_cycle*.updated.csv"))])
L=L[L.label_status.isin(["SUCCESS","METAL_ASSUMED"])].drop_duplicates("structure_id")
L["hse"]=np.where(L.label_status=="METAL_ASSUMED",0.0,L.hse_gap); hse=dict(zip(as_id(L.structure_id),L.hse))
rows=[]
for run,g in U.groupby("run"):
    rd=REPO/"OFFICIAL-RUNS/01_gate_aware_delta_ml_dos_on"/run
    P=joblib.load(rd/"models/phase2_cycle24.joblib"); pre=json.loads((rd/"models/preproc_cycle24.json").read_text())
    fillB=P.get("reg_fill_values",pre.get("B_fill_values",{})); Bn,Bb=list(fillB.keys()),list(pre.get("B_bool_cols",[]))
    df=g[["structure_id"]].merge(pool,on="structure_id",how="left")
    need=[c for c in Bn+Bb if str(c).startswith("_dos_")]
    df=add_dos_features_if_needed(df,"structure_id",need,REPO/"data/dos_from_output.csv")
    Xb=build_X(df,Bn,Bb,fillB,P["reg_scaler"])
    c=np.maximum(np.asarray(P["gpr_pbe0"].predict(Xb),float),0.0)
    for (_,r),cc in zip(g.iterrows(),c):
        rows.append(dict(run=run,structure_id=r.structure_id,rescuable=r.rescuable,p_gate=r.p_gate,hse_ref=hse[r.structure_id],stageC=cc))
D=pd.DataFrame(rows); D.to_csv(OUT/"gate_vs_rules_pairs.csv",index=False)
y=D.hse_ref.values; c=D.stageC.values; passg=(D.p_gate>=THETA).values; rate=passg.mean()
mae=lambda pred: np.abs(pred-y).mean()
res=dict(n_pairs=len(D),pass_rate=rate,gate=mae(np.where(passg,c,0)),always_rescue=mae(c),always_block=mae(np.zeros_like(y)))
rng=np.random.default_rng(0); draws=np.array([mae(np.where(rng.random(len(y))<rate,c,0)) for _ in range(10000)])
res.update(random_mean=draws.mean(),random_sd=draws.std(),random_p2_5=np.percentile(draws,2.5),random_p97_5=np.percentile(draws,97.5),
           frac_random_le_gate=(draws<=res["gate"]).mean())
# per-run
pr=[]
for run,g in D.groupby("run"):
    yy,cc,pp=g.hse_ref.values,g.stageC.values,(g.p_gate>=THETA).values
    pr.append(dict(run=run,n=len(g),gate=np.abs(np.where(pp,cc,0)-yy).mean(),always=np.abs(cc-yy).mean()))
pd.DataFrame([res]).to_csv(OUT/"gate_vs_rules_summary.csv",index=False); pd.DataFrame(pr).to_csv(OUT/"gate_vs_rules_per_run.csv",index=False)
print(pd.Series(res).round(4).to_string()); print(pd.DataFrame(pr).round(4).to_string(index=False))
print("mean stageC rescuable / metal:",D[D.rescuable==1].stageC.mean().round(3),D[D.rescuable==0].stageC.mean().round(3))
