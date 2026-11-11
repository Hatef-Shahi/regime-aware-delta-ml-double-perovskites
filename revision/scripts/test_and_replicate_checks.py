"""Held-out test and replicate checks from the deposited arm-01 artefacts.

Writes
  test_spacegroup_errors.csv   Table S9(b): end-to-end test error by post-relaxation space group
                               (5 runs pooled)
  test_metrics_twins.csv       run-wise test MAE/RMSE/R2 on all 30 structures and on the 28
                               without a duplicate in the screening pool (ESI Section 1)
  replicate_overlap.csv        pairwise overlap of the cycle-24 training sets (Section 4.8)
  seed_overlap.csv             seed labels of al2-al5 drawn from al1's labelled set, and the
                               number of HSE06 labels al2-al5 needed beyond al1 (Table 2 footnote)
  outlier_support.csv          Table S17: training support of the Ba2BiSbO6 P-1 test outlier

Usage (from the repository root):
    python revision/scripts/test_and_replicate_checks.py . revision/results
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"  # single-threaded BLAS so that repeated runs give identical output
import sys, glob, itertools
import numpy as np, pandas as pd

REPO, OUT = sys.argv[1], sys.argv[2]
B = f"{REPO}/OFFICIAL-RUNS/01_gate_aware_delta_ml_dos_on"
RUNS = ["al1", "al2", "al3", "al4", "al5"]
U = pd.read_csv(f"{REPO}/data/universe_master.csv")

# --- test errors -------------------------------------------------------------
T = pd.concat([pd.read_csv(f"{B}/{r}/test_eval/test_pred_cycle24.csv").assign(run=r) for r in RUNS])
T = T.merge(U[["structure_id", "spg_symbol", "twin_of"]], on="structure_id", how="left")
T["err"] = T.hse_pred - T.hse_true
sg = T.groupby("spg_symbol").apply(lambda x: pd.Series(dict(n_structures=x.structure_id.nunique(),
        MAE=x.err.abs().mean(), RMSE=float(np.sqrt((x.err ** 2).mean()))))).reset_index()
sg.to_csv(f"{OUT}/test_spacegroup_errors.csv", index=False)
print(sg.round(3).to_string(index=False))


def runwise(df):
    rows = []
    for r, x in df.groupby("run"):
        ss = ((x.hse_true - x.hse_true.mean()) ** 2).sum()
        rows.append(dict(run=r, n=len(x), MAE=x.err.abs().mean(), RMSE=float(np.sqrt((x.err ** 2).mean())),
                         R2=1 - (x.err ** 2).sum() / ss))
    return pd.DataFrame(rows)


pool_ids = set(U.loc[U.split_day0 != "test", "structure_id"])
twin_in_pool = T.twin_of.fillna("").isin(pool_ids)
rows = []
for name, df in [("all_30", T), ("without_pool_twins", T[~twin_in_pool])]:
    w = runwise(df)
    rows.append(dict(set=name, n_structures=df.structure_id.nunique(),
                     MAE_mean=w.MAE.mean(), MAE_sd_ddof1=w.MAE.std(ddof=1), MAE_sd_ddof0=w.MAE.std(ddof=0),
                     RMSE_mean=w.RMSE.mean(), RMSE_sd_ddof1=w.RMSE.std(ddof=1), RMSE_sd_ddof0=w.RMSE.std(ddof=0),
                     R2_mean=w.R2.mean(), R2_sd_ddof1=w.R2.std(ddof=1), R2_sd_ddof0=w.R2.std(ddof=0)))
tw = pd.DataFrame(rows)
tw.to_csv(f"{OUT}/test_metrics_twins.csv", index=False)
print(tw.round(4).to_string(index=False))
print("test structures with a pool twin:", sorted(T.loc[twin_in_pool, "structure_id"].unique()))

# --- replicate overlap ---------------------------------------------------------
train = {}
for r in RUNS:
    led = pd.read_csv(f"{B}/{r}/Seed_labels_cycle23.updated.csv")
    val = set(pd.read_csv(f"{B}/{r}/phase2_val_ids.csv").iloc[:, 0])
    train[r] = set(led.structure_id) - val
rows = [dict(a=a, b=b, shared=len(train[a] & train[b]), n_a=len(train[a]), n_b=len(train[b]),
             jaccard=len(train[a] & train[b]) / len(train[a] | train[b])) for a, b in itertools.combinations(RUNS, 2)]
ov = pd.DataFrame(rows)
ov.to_csv(f"{OUT}/replicate_overlap.csv", index=False)
print(ov.round(3).to_string(index=False))
print("common to all five:", len(set.intersection(*train.values())))

# --- seed provenance ------------------------------------------------------------
al1_all = set()
al1_23 = set()
for f in glob.glob(f"{B}/al1/Seed_labels_cycle*.updated.csv"):
    al1_all |= set(pd.read_csv(f).structure_id)
    if int(f.split("cycle")[-1].split(".")[0]) <= 23:
        al1_23 |= set(pd.read_csv(f).structure_id)
rows = []
beyond = set()
for r in RUNS[1:]:
    led0 = pd.read_csv(f"{B}/{r}/Seed_labels_cycle0.updated.csv")
    sel0 = set(pd.read_csv(f"{B}/{r}/selections/selected_cycle0.csv").structure_id)
    val = set(pd.read_csv(f"{B}/{r}/phase2_val_ids.csv").iloc[:, 0])
    seed = set(led0.structure_id) - sel0 - val
    for f in glob.glob(f"{B}/{r}/Seed_labels_cycle*.updated.csv"):
        if int(f.split("cycle")[-1].split(".")[0]) <= 23:
            beyond |= set(pd.read_csv(f).structure_id) - al1_23
    rows.append(dict(run=r, seed_non_val=len(seed), from_al1_labels=len(seed & al1_all), outside_al1=len(seed - al1_all)))
so = pd.DataFrame(rows)
so.to_csv(f"{OUT}/seed_overlap.csv", index=False)
print(so.to_string(index=False))
print("HSE06 labels needed by al2-al5 beyond al1 (ledgers up to cycle 23):", len(beyond))
pd.DataFrame([dict(run="al2-al5 labels beyond al1, ledgers up to cycle 23", seed_non_val=None, from_al1_labels=None, outside_al1=len(beyond))]).to_csv(f"{OUT}/seed_overlap.csv", mode="a", header=False, index=False)

# --- outlier support (Table S17) --------------------------------------------------
sid = "GLAZER:Ba2BiSbO6_mp-cefww.vasp"
o = U.set_index("structure_id").loc[sid]
rows = []
for r in RUNS:
    led = pd.read_csv(f"{B}/{r}/Seed_labels_cycle23.updated.csv")
    m = led.merge(U[["structure_id", "formula", "A_element", "B1_element", "B2_element", "spg_number", "crystal_system"]],
                  on="structure_id", how="left", suffixes=("_l", ""))
    pair = m.apply(lambda x: frozenset([x.B1_element, x.B2_element]) == frozenset([o.B1_element, o.B2_element]), axis=1)
    rows.append(dict(run=r, labels=len(led), same_formula=int((m.formula == o.formula).sum()), same_B_pair=int(pair.sum()),
                     same_A=int((m.A_element == o.A_element).sum()), same_spg=int((m.spg_number == o.spg_number).sum()),
                     same_crystal_system=int((m.crystal_system.str.lower() == str(o.crystal_system).lower()).sum())))
os_ = pd.DataFrame(rows)
os_.to_csv(f"{OUT}/outlier_support.csv", index=False)
print(os_.to_string(index=False))
