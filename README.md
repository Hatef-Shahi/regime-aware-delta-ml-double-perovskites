# Regime-aware Δ-ML active learning for HSE06 band-gap prediction

This repository contains the official run outputs, scripts, audits, and data files needed to reproduce the manuscript results.

## Main folders

- `OFFICIAL-RUNS/`: official manuscript runs only.
- `scripts/`: workflow, evaluation, figure, and audit scripts.
- `audits/`: release checks and diagnostic CSVs used to support manuscript statements.
- `data/`: compact input tables and feature-schema files.
- `OFFICIAL_RESULTS.md`: mapping between manuscript claims and official run folders.
- `environment.yml`: software environment used for the saved models.

## Reproducibility notes

The official fixed-budget comparison uses cycle 24. Saved joblib models were generated with scikit-learn 1.3.2. Use the supplied environment for exact loading of model checkpoints.
