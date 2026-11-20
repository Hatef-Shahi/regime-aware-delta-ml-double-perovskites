# Analyses added in the major revision

This folder holds the scripts and result tables produced while answering the referee
reports on manuscript DD-ART-08-2026-000610. Except for `eig_gap_band_index.py`, which
reads a VASP EIGENVAL file, every script runs from the repository root on files that are
already in the repository, needs no new DFT calculation, and writes its output to
`revision/results/`. Table and figure numbers refer to the revised ESI.
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
| `calibration.py` | `calibration_summary.csv`, `calibration_noisefloor.csv`, `calibration_curve.csv` | Table S15, sigma_cal column of Table S24 | 3.14 |
| `calib_curve_sigcal.py` | `calibration_curve_with_sigcal.csv` | Fig. S16 | 3.14 |
| `representativeness.py` | `representativeness_tests.csv`, `representativeness_pca.csv`, `test_coverage_nn.csv` | Table S16 | 3.1, 3.20 |
| `test_and_replicate_checks.py` | `test_spacegroup_errors.csv`, `test_metrics_twins.csv`, `replicate_overlap.csv`, `seed_overlap.csv`, `outlier_support.csv` | Table S9(b), ESI Section 1 (duplicate-excluded test metrics), Table 2 footnote, Table S17 | 1.4, 3.3, 3.13, 3.18 |
| `cv_extra.py` | `cv_extra_summary.csv`, `cv_extra_predictions.csv` | Table S8 (direct + PBE cell of the Δ-GPR row; Δ-GPR with the PBE gap as a feature; top-30 and top-15 ARD refits), Sections 2.2 and 4.4.2, ESI Section 3 | 3.11, 3.13 |
| `seed_baselines.py` | `seed_baselines.csv` | ESI Section 3 (seed-only baselines) | 2.1 |
| `delta_r2.py` | `delta_r2.csv` | Section 2.3 (validation and test Δ-MAE, Δ R² and Δ variances, pooled over replicates) | 3.13 |
| `gate_vs_rules.py` | `gate_vs_rules_summary.csv`, `gate_vs_rules_per_run.csv`, `gate_vs_rules_pairs.csv` | Section 2.3, Table S13 note | 3.8 |
| `ard_shared.py` | `ard_cycle24_models.csv` | ESI Section 3 (short ARD length scales of the five deployed models) | 2.2, 3.13 |
| `eig_gap_band_index.py` | printed gap | Table S22 (HSE06+SOC gaps by band index) | 1.1, 3.6 |
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
python revision/scripts/cv_extra.py . revision/results
python revision/scripts/seed_baselines.py . revision/results/seed_baselines.csv
python revision/scripts/delta_r2.py . revision/results/delta_r2.csv
python revision/scripts/gate_vs_rules.py . revision/results
python revision/scripts/ard_shared.py . revision/results
```

`shap_gpr.py` imports the data loader of `cv_baselines.py`, `cv_derived.py` and
`shap_derived.py` read the outputs of `cv_baselines.py` and `shap_gpr.py`, and
`make_figs.py` reads the output of `calib_curve_sigcal.py`. The cross-validation, the routing-threshold sweep and
the SHAP run each take tens of minutes on two cores, as does `cv_extra.py`; the rest
finish within a minute.

`cv_extra.py` uses the data loader, preprocessing, Δ-GPR and composition folds of
`cv_baselines.py`; its full-feature reference reproduces the Δ-GPR value of Table S8
(0.113 eV). `gate_vs_rules.py` uses the pairs of `deployed_gate_unseen.csv`.
`eig_gap_band_index.py EIGENVAL [ncl]` takes the gap by band index (band N_occ and
N_occ + 1, N_occ = NELECT/2, or NELECT for a non-collinear run) over all k-points,
including the zero-weight band-edge points that the HSE06 runs add to the mesh.

## Tables built from DFT checks

Six result tables come from additional DFT calculations made for the revision (PBE, and
HSE06 for the SOC check) rather than from a script in this folder. They are deposited
as tables.

| File | Revised ESI | Comments | Content |
|---|---|---|---|
| `kpoint_convergence_picks.csv` | Table S20 (family representatives) | 1.2, 3.4 | the four representative structures chosen for the test, with their production PBE bandgap |
| `nearzbg_kpoint_table.csv` | Table S20 (structures with 0 < PBE gap <= 0.10 eV) | 1.2, 3.4 | production, control and denser-mesh PBE bandgaps and the routing at each mesh, with `labelled_in_arm01` giving the arm-01 replicates that labelled the structure |
| `encut_convergence.csv` | Table S21 | 3.4 | PBE bandgap of the four family representatives at the production cutoff and at +100 and +200 eV, with geometry and band-path k-points fixed; the production-cutoff runs reproduce the production gaps to 0.0001 eV |
| `soc_benchmark.csv` | Table S22 | 1.1, 3.6 | PBE and PBE+SOC bandgaps of eight structures; HSE06 reference only where it is an AL ledger or test label (`source` column); HSE06+SOC for Tl2GeMoO6, Ca2PtSiO6 and Ba2PbWO6 |
| `soc_hse06.csv` | Table S22 | 1.1, 3.6 | HSE06 and HSE06+SOC gaps with the PBE and PBE+SOC gaps and shifts for the three structures; HSE06+SOC are single points at the k-points of the HSE06 reference band edges, with the gap taken by band index over all k-points, including the zero-weight ones (`eig_gap_band_index.py`) |
| `plusU_vs_plain_pbe.csv` | Table S23 | general audit note, Table S19 footnote | PBE+U baseline against plain PBE from its own relaxation for 51 structures, with cell volumes and regime at tau0 = 0.05 eV |

`candidates84_novelty.csv` (Table S24, comment 2.3) lists the 84 visible-range
remaining-pool candidates with their Materials Project status from a composition query.
The candidate list itself is reproduced by
`scripts/figures/make_remaining_pool_candidate_table.py` (see the top-level README).

`shortlist_hse06_check.csv` (Table 3, Section 2.5, comment 3.15) compares the
predictions for the 20 shortlisted candidates with their HSE06 references. The references
come from the label ledgers of the other runs of this study (arms 01 to 05), computed
with the same workflow and settings; they are not new calculations. `labelled_in` lists
the runs that labelled each structure, `n_replicates` the arm-01 replicates whose
remaining pool contained it (the predictions are pooled over these only), and
`hse_pred_table` and `hse_pred_ownPBE` the predictions with the Materials Project and
our own PBE baseline.

The 39 non-retained Glazer-bank candidates of Table S1 are listed in
`data/dropped39_classified.csv`.

## Reproduction status

The scripts added in v1.1.1 (`cv_extra.py`, `seed_baselines.py`, `delta_r2.py`,
`gate_vs_rules.py`, `ard_shared.py`) were run single-threaded with Python 3.11.18,
numpy 1.26.4, scikit-learn 1.3.2 and scipy 1.11.4. `seed_baselines.py`, `delta_r2.py`,
`gate_vs_rules.py` and `ard_shared.py` give byte-identical outputs with pandas 1.5.3 and
joblib 1.4.2 (the pinned versions) and with pandas 2.1.4 and joblib 1.6.0.
`cv_extra.py` was run with pandas 2.1.4 and joblib 1.6.0; its composition folds depend on
the numpy version, so other numpy versions give slightly different values.

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
