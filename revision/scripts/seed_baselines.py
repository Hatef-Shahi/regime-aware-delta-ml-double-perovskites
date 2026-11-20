"""Seed-only baselines (comment 2.1; ESI Section 3): Delta-target GPR, RF, SVR trained on the 70 non-validation
seed labels of each arm-01 replicate (non-ZBG rows only), evaluated on that replicate's
non-ZBG validation structures and on the 23 non-ZBG held-out test structures.
Usage: python seed_baselines.py <repo> <out.csv>"""
import os, sys, json
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"): os.environ[v]="1"
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern, WhiteKernel
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
R=sys.argv[1]; THR=0.05
pre=json.load(open(f"{R}/OFFICIAL-RUNS/01_gate_aware_delta_ml_dos_on/al1/models/preproc_cycle16.json"))
NUM=[c for c in pre["B_num_cols"] if c in pre["B_fill_values"]]; BOOL=pre["B_bool_cols"]
F=pd.read_csv(f"{R}/data/trainpool_ml_augmented.csv").drop_duplicates("structure_id").set_index("structure_id")
test=pd.read_csv(f"{R}/data/test_labels_labeled.csv")
test=test[(test.label_status=="SUCCESS")&(test.pbe_gap>THR)]
def XY(ids_df):
    X=F.loc[ids_df.structure_id,NUM].apply(pd.to_numeric,errors="coerce").replace([np.inf,-np.inf],np.nan)
    B=F.loc[ids_df.structure_id,BOOL].apply(pd.to_numeric,errors="coerce").fillna(0).astype(float)
    return X,B,(ids_df.hse_gap-ids_df.pbe_gap).values
def gpr(nf):
    k=C(1.0,(1e-3,1e3))*Matern(length_scale=np.ones(nf),nu=2.5,length_scale_bounds=(1e-3,1e3))+WhiteKernel(1e-6)
    return GaussianProcessRegressor(kernel=k,alpha=1e-6,n_restarts_optimizer=5,normalize_y=True,random_state=0)
rows=[]
for r in range(1,6):
    seed=pd.read_csv(f"{R}/data/seed_0{r}.csv")
    val_ids=set(pd.read_csv(f"{R}/OFFICIAL-RUNS/01_gate_aware_delta_ml_dos_on/al{r}/phase2_val_ids.csv").iloc[:,0].astype(str))
    ok=(seed.label_status=="SUCCESS")&(seed.pbe_gap>THR)
    tr=seed[ok&~seed.structure_id.isin(val_ids)]; va=seed[ok&seed.structure_id.isin(val_ids)]
    Xtr,Btr,ytr=XY(tr); med=Xtr.median(); keep=Xtr.columns[Xtr.fillna(med).std()>0]
    sc=StandardScaler().fit(Xtr[keep].fillna(med))
    T=lambda X,B: np.hstack([sc.transform(X[keep].fillna(med)),B.values])
    A=T(Xtr,Btr); Xv,Bv,yv=XY(va); Xt,Bt,yt=XY(test)
    ms={"GPR":gpr(A.shape[1]),
        "RF":RandomForestRegressor(n_estimators=500,min_samples_leaf=2,max_features=0.33,random_state=0,n_jobs=1),
        "SVR":GridSearchCV(SVR(kernel="rbf"),{"C":[1,3,10,30],"epsilon":[0.02,0.05,0.1],"gamma":["scale",0.003,0.01]},cv=3)}
    for n,m in ms.items():
        m.fit(A,ytr)
        rows.append(dict(run=f"al{r}",model=n,n_train=len(tr),n_val=len(va),
            val_MAE=np.abs(m.predict(T(Xv,Bv))-yv).mean(),test_MAE=np.abs(m.predict(T(Xt,Bt))-yt).mean()))
    print(rows[-3:],flush=True)
df=pd.DataFrame(rows); df.to_csv(sys.argv[2],index=False)
print(df.groupby("model")[["n_train","val_MAE","test_MAE"]].agg(["mean","std"]).round(3))
