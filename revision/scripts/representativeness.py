"""R3.1: quantitative representativeness of the 30-structure held-out test set
relative to the 472-entry screening pool (data/universe_master.csv).

Usage (from the repository root):
    python revision/scripts/representativeness.py . revision/results
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"  # single-threaded BLAS so that repeated runs give identical output
import sys, re, glob
import numpy as np, pandas as pd
from scipy.stats import ks_2samp, fisher_exact, chi2_contingency
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

REPO, OUT = sys.argv[1], sys.argv[2]
U = pd.read_csv(f"{REPO}/data/universe_master.csv")
T = pd.read_csv(f"{REPO}/data/test_labels_labeled.csv")
L = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{REPO}/OFFICIAL-RUNS/*/al*/labels_store.csv"))])
L = L[L.label_status.isin(["SUCCESS", "METAL_ASSUMED"])].drop_duplicates("structure_id")
U["set"] = np.where(U.split_day0 == "test", "test", "pool")
U["crystal_system"] = U.crystal_system.str.lower()   # MP labels were capitalised in v1.0.0


def fam(r):
    if r.source == "MP":
        return "MP"
    m = re.search(r"_(Fm-3m|I4-m|R-3|P2_1-c)\.vasp$", str(r.structure_id))
    return m.group(1) if m else "Glazer-other"


U["family"] = U.apply(fam, axis=1)
U["zbg"] = (U.pbe_gap <= 0.05).astype(int)
te, po = U[U.set == "test"], U[U.set == "pool"]
print(f"pool n={len(po)}, test n={len(te)}")

rows = []
for col in ["source", "family", "zbg", "crystal_system"]:
    ct = pd.crosstab(U[col], U.set)
    p = fisher_exact(ct.values)[1] if ct.shape == (2, 2) else chi2_contingency(ct.values)[1]
    for lev in ct.index:
        rows.append(dict(kind="categorical", variable=col, level=str(lev), pool=ct.loc[lev, "pool"] / len(po),
                         test=ct.loc[lev, "test"] / len(te), n_test=int(ct.loc[lev, "test"]), p_value=p))
CONT = ["pbe_gap", "tolerance_factor", "volume_per_atom", "density", "rA", "rB1", "rB2",
        "A_oxidation", "B1_oxidation", "B2_oxidation", "spg_number"]
for c in CONT:
    a, b = pd.to_numeric(po[c], errors="coerce").dropna(), pd.to_numeric(te[c], errors="coerce").dropna()
    rows.append(dict(kind="continuous", variable=c, level="median", pool=a.median(), test=b.median(),
                     n_test=len(b), p_value=ks_2samp(a, b).pvalue))
hp = po.merge(L[["structure_id", "hse_gap"]], on="structure_id").hse_gap.fillna(0)
ht = T.hse_gap.fillna(0)
rows.append(dict(kind="continuous", variable="hse_gap (labelled pool vs test)", level="median", pool=hp.median(),
                 test=ht.median(), n_test=len(ht), p_value=ks_2samp(hp, ht).pvalue))
R = pd.DataFrame(rows)
R.to_csv(f"{OUT}/representativeness_tests.csv", index=False)
print(R.round(3).to_string(index=False))

for site in ["A_element", "B1_element", "B2_element"]:
    pool_el = set(po[site]) | set(po["B1_element" if site == "B2_element" else ("B2_element" if site == "B1_element" else site)])
    miss = sorted(set(te[site]) - pool_el)
    print(f"{site}: test elements absent from pool at this site: {miss}")

X = U[CONT].apply(pd.to_numeric, errors="coerce")
X = X.fillna(X.median())
Z = StandardScaler().fit(X[U.set == "pool"]).transform(X)
zp, zt = Z[(U.set == "pool").values], Z[(U.set == "test").values]
nn = NearestNeighbors(n_neighbors=2).fit(zp)
d_pp = nn.kneighbors(zp)[0][:, 1]
d_tp = nn.kneighbors(zt, n_neighbors=1)[0][:, 0]
q95 = np.quantile(d_pp, 0.95)
print(f"NN distance to pool: pool LOO median {np.median(d_pp):.3f} (95th {q95:.3f}); test median {np.median(d_tp):.3f}; "
      f"test beyond pool 95th pct: {(d_tp > q95).sum()}/30; KS p={ks_2samp(d_pp, d_tp).pvalue:.3f}")
cov = pd.DataFrame(dict(structure_id=U.structure_id[U.set == "test"].values, nn_dist_to_pool=d_tp,
                        pool_percentile=[(d_pp < x).mean() * 100 for x in d_tp]))
cov = cov.merge(U[["structure_id", "formula", "spg_symbol", "family", "pbe_gap"]], on="structure_id").sort_values("nn_dist_to_pool", ascending=False)
cov.to_csv(f"{OUT}/test_coverage_nn.csv", index=False)
print(cov.head(6).round(3).to_string(index=False))

pca = PCA(2).fit(zp)
P = pca.transform(Z)
pd.DataFrame(dict(structure_id=U.structure_id, set=U.set, family=U.family, zbg=U.zbg, pc1=P[:, 0], pc2=P[:, 1])).to_csv(f"{OUT}/representativeness_pca.csv", index=False)
print("PCA explained variance:", pca.explained_variance_ratio_.round(3))
