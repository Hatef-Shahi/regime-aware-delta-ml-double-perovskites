# Regime-aware Δ-ML active learning for HSE06 band-gap prediction

Code, data, and analysis scripts accompanying the manuscript submitted to
RSC Digital Discovery:

> **Regime-aware active learning for hybrid-functional band gap
> prediction across the unfiltered A2BB'O6 double-perovskite space**
> Hatef Shahmohamadi, Jiří Hostaš, Daya Gaur, Maicon Pierre Lourenço,
> Dennis R. Salahub
> (DOI to be added upon publication)

This repository implements a regime-aware Δ-machine-learning active-learning
protocol that predicts HSE06-level band gaps for A2BB'O6 double perovskites
from inexpensive PBE calculations, using a dedicated gate classifier to
handle near-zero-PBE-gap candidates rather than excluding them. It
contains the official run outputs, scripts, and audits needed to reproduce
every reported result across the five experimental configurations compared
in the paper (see `OFFICIAL_RESULTS.md` for the full mapping between
manuscript claims and run folders).

## Quick start

See `OFFICIAL_RESULTS.md` for which run folder corresponds to which
manuscript result, and `scripts/evaluation/` for the scripts used to
generate the reported cycle-24 metrics from the saved model checkpoints.

## Main folders

- `OFFICIAL-RUNS/`: official manuscript runs only.
- `scripts/`: workflow, evaluation, figure, and audit scripts.
- `audits/`: release checks and diagnostic CSVs used to support manuscript statements.
- `data/`: compact input tables and feature-schema files.
- `OFFICIAL_RESULTS.md`: mapping between manuscript claims and official run folders.
- `environment.yml`: software environment used for the saved models.

## Reproducibility notes

The official fixed-budget comparison uses cycle 24. Saved joblib models
were generated with scikit-learn 1.3.2. Use the supplied environment for
exact loading of model checkpoints.

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## Citation

If you use this code or data, please cite the paper above (citation
details to be finalized once the DOI is assigned) and/or this repository
directly via its Zenodo DOI.
