# Analyses added in the major revision

This folder holds the scripts and result tables produced while answering the referee
reports on manuscript DD-ART-08-2026-000610. Every script runs from the repository root
on files that are already in the repository, needs no new DFT calculation, and writes
its output to `revision/results/`. Table and figure numbers refer to the revised ESI.
Comment numbers refer to the response letter (Referee 1 = 1.x, Referee 2 = 2.x,
Referee 3 = 3.x).

The scripts were run with Python 3.11 and the versions in `revision/requirements.txt`
(numpy 1.26.4, pandas 1.5.3, scikit-learn 1.3.2, scipy 1.11.4, joblib 1.4.2,
matplotlib 3.7.5, shap 0.44.1):

```bash
python3.11 -m venv .venv-revision && source .venv-revision/bin/activate
pip install -r revision/requirements.txt
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
```

The scripts must run single-threaded (`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
`OPENBLAS_NUM_THREADS=1`) to reproduce the deposited SHAP, ARD and Δ-GPR values, which
were produced in that setting.

Every script also sets these three variables to 1 itself before importing numpy, and `shap_gpr.py` seeds numpy's generator before the SHAP
sampling. With the pinned environment, two consecutive runs of the full set of scripts
give byte-identical output files, and the files in `revision/results/` are the record
from which every number in the revised paper and response letter is taken.

The package versions matter. scikit-learn 1.3.2 assigns compositions to the grouped
cross-validation folds by sorting group sizes with numpy, and the order of the many
equal-sized groups changes between numpy releases, so another numpy version gives
different grouped folds. The fitted Δ-GPR hyperparameters, and with them the ARD length
scales and SHAP values, also depend on the scipy optimiser version.

## Scripts, outputs and where they are used

| Script | Output in `revision/results/` | Revised ESI / main text | Comments |
|---|---|---|---|
| `universe_counts.py` | printed count chain | Table S1, ESI Section 1, Table 1 | 1.5 and the general audit note |
| `cv_baselines.py` | `cv_summary.csv`, `cv_spacegroup_gpr_delta.csv`, `cv_predictions.csv` | Table S8 | 1.3, 2.1, 3.2, 3.11, 3.13 |
| `cv_derived.py` | `cv_family_gpr_delta.csv`, `cv_derived_summary.csv` | Table S9(a); polymorph, median-error and fold-count figures in ESI Section 3 | 1.3, 3.2, 3.3 |
| `shap_gpr.py` | `shap_importance.csv`, `ard_length_scales_full.csv`, `shap_gpr_values.npy` | Table S10 | 2.2, 3.12, 3.13 |
| `shap_derived.py` | `shap_summary.csv` | attribution shares, GPR and random-forest agreement and ARD counts in Section 4.4.2 and ESI Section 3 | 2.2, 3.12, 3.13 |
| `tau_sensitivity.py` | `tau_summary.csv`, `tau_predictions.csv` | Table S12 | 3.7 |
| `deployed_gate_unseen.py` | `deployed_gate_unseen.csv`, `deployed_gate_unseen_summary.csv` | Table S13 (see the note below) | 3.8, 3.9 |
| `calibration.py` | `calibration_summary.csv`, `calibration_noisefloor.csv`, `calibration_curve.csv` | Table S15, sigma_cal column of Table S23 | 3.14 |
| `calib_curve_sigcal.py` | `calibration_curve_with_sigcal.csv` | Fig. S16 | 3.14 |
| `representativeness.py` | `representativeness_tests.csv`, `representativeness_pca.csv`, `test_coverage_nn.csv` | Table S16 | 3.1, 3.20 |
| `test_and_replicate_checks.py` | `test_spacegroup_errors.csv`, `test_metrics_twins.csv`, `replicate_overlap.csv`, `seed_overlap.csv`, `outlier_support.csv` | Table S9(b), ESI Section 1 (duplicate-excluded test metrics), Table 2 footnote, Table S17 | 1.4, 3.3, 3.13, 3.18 |
| `make_figs.py` | `figures/fig2ab_composition.png`, `figures/fig2c_spacegroups.png`, `figures/SI_glazer_prototype_usage.png`, `figures/SI_calibration_curve.pdf` | Fig. 2, Fig. S4, Fig. S16 | 1.5, 3.14 |

Commands, in the order that respects their dependencies:

```bash
python revision/scripts/universe_counts.py data/universe_master.csv
python revision/scripts/cv_baselines.py . revision/results 3
python revision/scripts/cv_derived.py revision/results
python revision/scripts/shap_gpr.py . revision/results 600
python revision/scripts/shap_derived.py revision/results
python revision/scripts/tau_sensitivity.py . revision/results 2 data/dos_from_output.csv
python revision/scripts/deployed_gate_unseen.py . revision/results
python revision/scripts/calibration.py . revision/results
python revision/scripts/calib_curve_sigcal.py . revision/results
python revision/scripts/representativeness.py . revision/results
python revision/scripts/test_and_replicate_checks.py . revision/results
python revision/scripts/make_figs.py revision/results/figures revision/results/calibration_curve_with_sigcal.csv
```

`shap_gpr.py` imports the data loader of `cv_baselines.py`, `cv_derived.py` and
`shap_derived.py` read the outputs of `cv_baselines.py` and `shap_gpr.py`, and
`make_figs.py` reads the output of `calib_curve_sigcal.py`. The cross-validation, the routing-threshold sweep and
the SHAP run each take tens of minutes on two cores; the rest finish within a minute.

## Tables built from DFT checks

Four result tables come from additional PBE-level calculations made for the revision
rather than from a script in this folder. They are deposited as tables.

| File | Revised ESI | Comments | Content |
|---|---|---|---|
| `kpoint_convergence_picks.csv` | Table S20 (family representatives) | 1.2, 3.4 | the four representative structures chosen for the test, with their production PBE bandgap |
| `nearzbg_kpoint_table.csv` | Table S20 (structures with 0 < PBE gap <= 0.10 eV) | 1.2, 3.4 | production, control and denser-mesh PBE bandgaps and the routing at each mesh, with `labelled_in_arm01` giving the arm-01 replicates that labelled the structure |
| `soc_benchmark.csv` | Table S21 | 1.1, 3.6 | PBE and PBE+SOC bandgaps; HSE06 reference only where it is an AL ledger or test label (`source` column); HSE06+SOC only for Tl2GeMoO6 |
| `plusU_vs_plain_pbe.csv` | Table S22 | general audit note, Table S19 footnote | PBE+U baseline against plain PBE from its own relaxation for 51 structures, with cell volumes and regime at tau0 = 0.05 eV |

`candidates84_novelty.csv` (Table S23, comment 2.3) lists the 84 visible-range
remaining-pool candidates with their Materials Project status from a composition query.
The candidate list itself is reproduced by
`scripts/figures/make_remaining_pool_candidate_table.py` (see the top-level README).

The 39 non-retained Glazer-bank candidates of Table S1 are listed in
`data/dropped39_classified.csv`.

## Reproduction status

`calibration.py`, `calib_curve_sigcal.py`, `representativeness.py`,
`test_and_replicate_checks.py` and the non-GPR models of `cv_baselines.py` give the same
values as the analyses first run for the revision. The Δ-GPR results (the GPR rows of
Table S8, Table S9(a), the GPR columns of Table S12 and the SHAP and ARD analysis of
Table S10) were recomputed in the pinned single-threaded environment, and the revised
paper reports these deposited values.

`deployed_gate_unseen.py` treats a structure as unseen by a replicate when it is not in
that replicate's `Seed_labels_cycle23.updated.csv`, the 268-row ledger (validation rows
included) that trained its cycle-24 models. It also reports, as a sensitivity check, the
stricter rule that the structure appears in none of the replicate's ledgers at any cycle.

## Not included

The plotting scripts for ESI Figs. S1 to S3 (feasibility map and per-site element
frequencies) are not in this release. The feature-group permutation importance of
Table S7 and Fig. S10 is computed by
`scripts/evaluation/feature_group_permutation_importance_cycle24.py` (see the top-level
README).
