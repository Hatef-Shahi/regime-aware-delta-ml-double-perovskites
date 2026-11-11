"""Universe count chain (Table S1, Fig. 2, Fig. S4): 502 entries, 11 duplicates, 491 unique
(148 MP-derived, 343 Glazer-generated) and post-relaxation symmetry retention.

Usage (from the repository root):
    python revision/scripts/universe_counts.py data/universe_master.csv
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"  # single-threaded BLAS so that repeated runs give identical output
import os, sys, pandas as pd
_path = os.environ.get("UNIVERSE_CSV") or (sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].endswith(".csv") else "data/universe_master.csv")
d = pd.read_csv(_path)
d["is_mpderived"] = (d.source == "MP") | d.structure_id.str.contains("_mp-")
mp_ids = set(d.loc[d.source == "MP", "mpid"])
d["gl_mpid"] = d.structure_id.str.extract(r"_(mp-[a-z]+)\.vasp")[0]
d["dup"] = (d.source == "GLAZER") & d.gl_mpid.isin(mp_ids)
u = d[~d.dup].copy()
u["src"] = u.is_mpderived.map({True: "MP-derived", False: "Glazer-generated"})
u["template"] = u.dir_name.str.extract(r"_(Fm-3m|I4-m|R-3|P2_1-c)\.vasp")[0].map({"Fm-3m": "Fm-3m", "I4-m": "I4/m", "R-3": "R-3", "P2_1-c": "P2_1/c"})
if __name__ == "__main__":
    print("entries", len(d), "dups", d.dup.sum(), "unique", len(u), u.src.value_counts().to_dict())
    if "twin_of" in d.columns:
        print("consistent with twin_of / counted_as_unique columns:",
              set(d.loc[d.dup, "structure_id"]) == set(d.loc[~d.counted_as_unique, "structure_id"]))
    print("pbe_gap==0 vs is_gap0:", ((u.pbe_gap == 0) == u.is_gap0).all(), " <=0.05:", (u.pbe_gap <= 0.05).sum(), " ==0:", (u.pbe_gap == 0).sum())
    print(pd.crosstab(u.src, u.pbe_gap == 0))
    g = u[u.src == "Glazer-generated"]
    print("Glazer template NaN:", g.template.isna().sum())
    print("retain:", (g.template == g.spg_symbol).sum(), "changed:", g[g.template != g.spg_symbol][["dir_name", "spg_symbol"]].values.tolist())
    print(g.spg_symbol.value_counts().to_dict())
    print(u.spg_symbol.value_counts().head(12).to_dict())
    pool = d[d.split_day0 != "test"]
    print("pool entries", len(pool), "pool unique", len(pool) - (pool.dup & pool.gl_mpid.map(dict(zip(d.mpid, d.split_day0))).ne("test")).sum(),
          "test", (d.split_day0 == "test").sum())
