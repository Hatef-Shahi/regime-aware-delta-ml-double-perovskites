# Official manuscript results

This repository contains the official run outputs, scripts, audits, and compact input data used for the manuscript.

The fixed-budget manuscript comparisons use cycle 24. For runs that continued beyond cycle 24, manuscript metrics were extracted from the row with `cycle == 24`, not from the final row of `metrics/metrics.csv`.

## Official run directories

### 01_gate_aware_delta_ml_dos_on

Directory: `OFFICIAL-RUNS/01_gate_aware_delta_ml_dos_on/al1--al5`

Role: retained gate-aware Δ-ML active-learning workflow.

Key settings:
- fixed-budget comparison cycle: 24
- batch size: 7
- gate quota: 2
- regression quota: 5
- gate threshold: 0.16
- `M_reg = 6`
- `q_floor = 0.35`
- DOS targeting: enabled
- baseline-ZBG threshold: `E_g^PBE <= 0.05 eV`

This is the retained production workflow used for the main gate-aware Δ-ML active-learning results.

### 02_nogate_unified_delta_fullpool

Directory: `OFFICIAL-RUNS/02_nogate_unified_delta_fullpool/al1--al5`

Role: true no-gate full-pool active-learning ablation.

This workflow uses one unified Δ-GPR, no gate classifier, no baseline-ZBG/non-ZBG acquisition split, and allows candidates with `E_g^PBE = 0` to compete in the same acquisition pool. This ablation tests whether explicit regime-aware routing improves active learning beyond simply allowing baseline-ZBG candidates to be selected.

The fixed-budget comparison against the gate-aware workflow uses cycle 24.

### 03_random_quota_matched

Directory: `OFFICIAL-RUNS/03_random_quota_matched/random1--random5`

Role: quota-matched random-learning baseline.

This baseline uses the same cycle-24 labelling budget as the retained gate-aware active-learning workflow. The source folders were originally stored under a DOS-off run root because random acquisition does not use the DOS-targeting acquisition score. This is a storage/name artifact, not a different evaluation target.

The fixed-budget comparison uses cycle 24.

### 04_direct_hse_regression

Directory: `OFFICIAL-RUNS/04_direct_hse_regression/direct1--direct5`

Role: direct-HSE regression baseline.

This baseline trains a direct model for the HSE06 band gap rather than using the Δ-ML target. The fixed-budget comparison uses cycle 24.

### 05_dos_off_sensitivity

Directory: `OFFICIAL-RUNS/05_dos_off_sensitivity/al1--al5`

Role: DOS-off sensitivity run.

This directory contains the corresponding gate-aware workflow with DOS targeting disabled. It is included for the DOS on/off sensitivity analysis.

The fixed-budget comparison uses cycle 24.

## Validation-set protocol

Validation IDs are replicate-index matched across all official arms: gate-aware, no-gate full-pool, quota-matched random, and direct-HSE.

For example:
- `01_gate_aware_delta_ml_dos_on/al1`
- `02_nogate_unified_delta_fullpool/al1`
- `03_random_quota_matched/random1`
- `04_direct_hse_regression/direct1`

use the same 30 validation IDs.

The same replicate-index matching holds for replicates 2 through 5. The validation split varies across replicate indices, giving five unique validation sets total. Therefore, differences between methods are not caused by different validation splits within a replicate-index comparison.

## Required files in each official run

Each official replicate contains the files required to reproduce or audit the cycle-24 manuscript metrics:

- `phase2_val_ids.csv`: frozen validation IDs for that replicate
- `metrics/metrics.csv`: cycle-wise validation metrics
- `selections/selected_cycle*.csv`: active-learning selections by cycle
- `Seed_labels_cycle*.updated.csv`: cycle-wise training ledgers
- `models/preproc_cycle24.json`: preprocessing metadata for the cycle-24 model
- `models/phase2_cycle24.joblib`: saved cycle-24 model checkpoint
- `test_eval/`: final test-set predictions and metrics where applicable

## Data files

Compact input and schema files are stored in `data/`.

Expected files include:
- `data/trainpool_ml_augmented.csv`
- `data/test_labels_labeled.csv`
- `data/feature_schema_gate.csv`
- `data/feature_schema_regression.csv`

## Scripts

Scripts are organized as follows:

- `scripts/workflow/`: active-learning workflow scripts
- `scripts/evaluation/`: final validation/test evaluation scripts
- `scripts/figures/`: manuscript and ESI figure-generation scripts
- `scripts/audits/`: reproducibility and consistency audit scripts

Exploratory, obsolete, and replacement scripts are intentionally excluded from the reviewer-facing release.

## Audits

Release and diagnostic audit outputs are stored in `audits/`.

Important audit folders include:
- `audits/release_audits/`
- `audits/validation_matching/`
- `audits/feature_table_audit/`
- `audits/gate_threshold_false_rescues/`
- `audits/delta_vs_direct_variance/`
- `audits/nogate_vs_gate_fullpool/`

These files document the validation-set matching, cycle-24 availability, feature-schema checks, no-gate ablation, gate-threshold sensitivity, and Δ-ML versus direct-HSE variance diagnostics.

## Environment

Saved joblib models were generated with scikit-learn 1.3.2. Loading them with newer scikit-learn versions may raise `InconsistentVersionWarning`.

Use `environment.yml` for exact reproduction of the saved model checkpoints.

## Checksums

File checksums are provided in `SHA256SUMS.txt`.
