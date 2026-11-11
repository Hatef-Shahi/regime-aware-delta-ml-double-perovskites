"""SHAP for the non-ZBG Delta-GPR (R2.2, R3.12) on the same data and preprocessing
as cv_baselines.py. Model fitted on all labelled non-ZBG pool structures;
KernelExplainer with a k-means background; RF TreeExplainer as a cross-check.

Usage (from the repository root; needs shap 0.44.1):
    python revision/scripts/shap_gpr.py . revision/results [nsamples=600]
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"  # single-threaded BLAS so that repeated runs give identical output
import sys, numpy as np, pandas as pd, shap, warnings
warnings.filterwarnings("ignore")
REPO, OUT = sys.argv[1], sys.argv[2]
NS = int(sys.argv[3]) if len(sys.argv) > 3 else 600
sys.argv = ["x", REPO, OUT, "1"]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cv_baselines import d, X_all, BOOL, prep, gpr, RandomForestRegressor  # data and model definitions

idx = np.arange(len(d))
Xtr, _ = prep(idx, idx[:1])
med = X_all.median(); Xf = X_all.fillna(med); keep = list(Xf.columns[Xf.std() > 0])
names = keep + BOOL
y = d.delta.values

g = gpr(Xtr.shape[1]).fit(Xtr, y)
ls = g.kernel_.k1.k2.length_scale
pd.DataFrame(dict(feature=names, ard_length_scale=ls)).sort_values("ard_length_scale").to_csv(f"{OUT}/ard_length_scales_full.csv", index=False)

np.random.seed(0)  # KernelExplainer samples coalitions with numpy's global generator
bg = shap.kmeans(Xtr, 20)
ex = shap.KernelExplainer(g.predict, bg)
sv = ex.shap_values(Xtr, nsamples=NS, silent=True)
np.save(f"{OUT}/shap_gpr_values.npy", sv)

rf = RandomForestRegressor(n_estimators=500, min_samples_leaf=2, max_features=0.33, random_state=0).fit(Xtr, y)
sv_rf = shap.TreeExplainer(rf).shap_values(Xtr)

imp = pd.DataFrame(dict(feature=names, mean_abs_shap_gpr=np.abs(sv).mean(0), mean_abs_shap_rf=np.abs(sv_rf).mean(0),
                        ard_length_scale=ls))
imp["rank_gpr"] = imp.mean_abs_shap_gpr.rank(ascending=False)
imp["rank_rf"] = imp.mean_abs_shap_rf.rank(ascending=False)
imp.sort_values("rank_gpr").to_csv(f"{OUT}/shap_importance.csv", index=False)
print(imp.sort_values("rank_gpr").head(20).round(4).to_string(index=False))
from scipy.stats import spearmanr
print("Spearman GPR vs RF mean|SHAP|:", round(spearmanr(imp.mean_abs_shap_gpr, imp.mean_abs_shap_rf).correlation, 3))
print("top-10 overlap:", len(set(imp.nsmallest(10, "rank_gpr").feature) & set(imp.nsmallest(10, "rank_rf").feature)))
