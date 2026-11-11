# Official manuscript results

Every comparison in the manuscript is made at cycle 24. For runs that continued past
cycle 24 (arm 01 al2, al3 and al5; arm 04 direct4), the reported metrics come from the
row with `cycle == 24` of `metrics/metrics.csv`, not from its last row, and the
reported test and remaining-pool results come from the saved cycle-24 model.

## Official run directories

| Folder | Replicates | Role in the paper |
|---|---|---|
| `OFFICIAL-RUNS/01_gate_aware_delta_ml_dos_on` | al1 to al5 | retained protocol; all headline results |
| `OFFICIAL-RUNS/02_nogate_unified_delta_fullpool` | al1 to al5 | controlled ablation of the gate (ESI Fig. S7) |
| `OFFICIAL-RUNS/03_random_quota_matched` | random1 to random5 | quota-matched random acquisition (ESI Fig. S8) |
| `OFFICIAL-RUNS/04_direct_hse_regression` | direct1 to direct5 | direct HSE06 regression instead of Δ (ESI Fig. S6) |
| `OFFICIAL-RUNS/05_dos_off_sensitivity` | al1 to al5 | DOS-off variant; trajectory-level comparison with arm 01 (ESI Fig. S9) |

All arms label seven structures per cycle and use the baseline-ZBG threshold
τ0 = 0.05 eV on the PBE gap. Arms 01, 03, 04 and 05 split each batch into two
baseline-ZBG and five non-ZBG labels and route with the gate at θ = 0.16; arm 02 has
no gate and no split. The retained non-ZBG selector settings are m_reg = 6 and
q_floor = 0.35. Arm 03 keeps the
two-regime quota but scores candidates at random within each regime. Its source folders
were stored under a DOS-off run root on the cluster because random acquisition does
not use the DOS term; this is a storage name only.

Arm 04 in this release (v1.1.0) contains the direct-HSE runs whose seeds and validation
sets match those of arm 01 and which give the values in the revised ESI (Table S2:
end-to-end validation MAE 0.492 ± 0.043 eV, non-ZBG MAE 0.519 ± 0.071 eV,
R² 0.809 ± 0.053, RMSE 0.681 ± 0.062 eV at cycle 24). It replaces the arm-04 folders of
v1.0.0.

## Validation sets

Validation IDs are matched by replicate index across all arms: al1 of arms 01, 02 and
05, random1 and direct1 share the same 30 validation structures, and likewise for
indices 2 to 5. The five indices have five different validation sets, each with 8
structures that are metallic at the HSE06 level and 22 that are not.
`audits/release_audits/official_validation_matching_all_arms.csv` and
`audits/validation_matching/` document the matching.

## Files in each official replicate

`phase2_val_ids.csv` (validation IDs), `Seed_labels_cycle<N>.updated.csv` (ledger at
the end of each cycle; the cycle-24 model is trained on the cycle-23 ledger, 268
rows), `labels_store.csv` (labels acquired by the run), `selections/`,
`metrics/metrics.csv` and `metrics/val_pred_cycle<N>.csv`, `models/phase2_cycle24.joblib`
and `models/preproc_cycle<N>.json`. Arms 01 and 05 also have `test_eval/` (cycle-24
test predictions and metrics), and arm 01 has `predictions/pred_cycle24.csv`, the cycle-24 predictions for
the structures each replicate left unlabelled. `README.md` describes the formats.

## Data files

`data/universe_master.csv` (502 entries, 491 unique structures),
`data/trainpool_ml_augmented.csv` (the same entries with all model descriptors and the
day-0 test split), `data/test_labels_labeled.csv` (30 test structures),
`data/dos_from_output.csv` (DOS quantities behind the gate descriptors),
`data/seed_0<k>.csv` (seed and validation cohort of replicate k),
`data/feature_schema_gate.csv` and `data/feature_schema_regression.csv` (saved feature
names), and `data/dropped39_classified.csv` (non-retained Glazer-bank candidates, ESI
Table S1).

## Audits

`audits/release_audits/` checks that every official replicate has its cycle-24 files
and matched validation IDs. `audits/feature_table_audit/` lists the feature names
stored in the saved models (ESI Tables S4 and S5). `audits/nogate_vs_gate_fullpool/`
holds the cycle-24 comparison of arms 01 and 02 and the Fig. S7 panels; the
`cycle24_*` files are the reported values, and the `final_cycle_*` files repeat the
comparison at the last cycle of each run. `audits/gate_threshold_false_rescues/` is
the post hoc threshold audit of ESI Section 4, and `audits/delta_vs_direct_variance/`
the replicate-variance comparison of arms 01 and 04.

## Environment

The saved models were written with scikit-learn 1.3.2 and numpy 1.24.4. Use
`environment.yml` or `requirements.txt` to load them. File checksums are in
`SHA256SUMS.txt`.
