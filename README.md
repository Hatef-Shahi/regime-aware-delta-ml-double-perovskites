# Regime-aware Δ-ML active learning for HSE06 bandgap prediction

Code, data and run outputs for the manuscript

> **Regime-aware active learning for hybrid-functional bandgap prediction
> across a gap-unfiltered A2BB'O6 double-perovskite space**
> Hatef Shahmohamadi, Jiří Hostaš, Daya Gaur, Maicon Pierre Lourenço,
> Dennis R. Salahub. Digital Discovery (under revision; DOI to be added on publication).

The protocol predicts HSE06-level bandgaps of A2BB'O6 double perovskites from a PBE
baseline. A Gaussian-process regressor learns the correction
Δ = E_g(HSE06) − E_g(PBE) for structures with a finite PBE gap, and a gate classifier
scores the structures whose PBE gap is at or below τ0 = 0.05 eV, so that those likely
to be semiconductors at the HSE06 level are passed to a separate rescued branch
instead of being discarded. HSE06 labels are acquired over 24 active-learning cycles
of seven structures each.

This is release v1.1.0, prepared for the revised manuscript. The changes relative to
v1.0.0 are listed in the release notes.

## Repository layout

`OFFICIAL-RUNS/` holds the five comparison arms of the paper, five replicates each.
`SENSITIVITY-RUNS/` holds the validation metrics of the single-configuration
hyperparameter sweeps (Figs. S11 to S13). `data/` holds the candidate universe, the
screening pool with all descriptors, the held-out test labels, the DOS descriptors and
the feature schemas. `scripts/` contains the active-learning driver (`workflow/`), the
test-set evaluation and the permutation-importance computation (`evaluation/`), the
figure and table scripts (`figures/`), consistency audits (`audits/`) and the parser that
built the DOS table (`data/`). `results/` holds the outputs of the figure and table
scripts as regenerated from the files in this repository, and `audits/` the outputs
of the audit scripts. `revision/` contains the analyses added during the revision,
with their own README and pinned requirements; those scripts must be run with
`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1` to reproduce the
deposited SHAP, ARD and Δ-GPR values.

## Installation

The official runs were produced with Python 3.8.10 and the package versions pinned in
`requirements.txt` (numpy 1.24.4, pandas 1.5.3, scikit-learn 1.3.2, scipy 1.10.1,
joblib 1.4.2, matplotlib 3.3.2). On the cluster these were Compute Canada wheels
(`+computecanada` builds) of the same version numbers; the public wheels installed
from `requirements.txt` load the saved models and reproduce the test predictions to
within 4e-14 eV.

```bash
conda env create -f environment.yml
conda activate delta-ml-al
```

or, with any Python 3.8 interpreter,

```bash
python -m venv .venv && source .venv/bin/activate
pip install --only-binary :all: -r requirements.txt
```

Every saved model records scikit-learn 1.3.2, and the pickles require numpy below 2.0.
The Python version (3.8.10) and the scipy, joblib and matplotlib versions are taken
from the environment of the production runs as recorded by the authors; they could not
be read back from the saved artefacts themselves. Loading the models with a newer
scikit-learn raises `InconsistentVersionWarning` and is not supported.

## Quick start

Re-evaluate the saved cycle-24 model of one replicate on the 30 held-out test
structures (about two seconds):

```bash
python scripts/evaluation/evaluate_test_cycle.py \
    --run-dir OFFICIAL-RUNS/01_gate_aware_delta_ml_dos_on/al1 \
    --pool data/trainpool_ml_augmented.csv \
    --test data/test_labels_labeled.csv \
    --dos data/dos_from_output.csv \
    --cycle 24 --out-dir /tmp/quickstart
```

The predictions written to `/tmp/quickstart` match
`OFFICIAL-RUNS/01_gate_aware_delta_ml_dos_on/al1/test_eval/test_pred_cycle24.csv`.
Without `--out-dir` the script writes into the run's own `test_eval/` folder and
overwrites the deposited file. Running it for al1 to al5 and averaging the run-wise
metrics gives the headline test result, MAE 0.196 ± 0.014 eV, RMSE 0.290 eV and
R² 0.958 ± 0.004 (standard deviation over the five replicates, population form).

## The comparison arms

All arms use the same screening pool, seed cohorts, validation sets and budget of
seven labels per cycle, and every number in the paper is taken at cycle 24, when each
replicate's ledger holds 268 rows (70 seed, 168 acquired, 30 validation).

`01_gate_aware_delta_ml_dos_on` is the retained protocol and gives every headline
result. `02_nogate_unified_delta_fullpool` is the controlled ablation of the gate: one
Δ-GPR is trained on all labels and one acquisition score ranks the whole pool.
`03_random_quota_matched` keeps the two-regime quota of two baseline-ZBG and five
non-ZBG labels per cycle but picks within each regime at random. `04_direct_hse_regression`
is identical to arm 01 except that the regressor is trained on the HSE06 gap instead
of on Δ. `05_dos_off_sensitivity` removes the DOS-derived descriptors from the gate and
the DOS-extreme pick from the baseline-ZBG acquisition. Because this changes which
structures are labelled from the first cycle on, arm 01 against arm 05 is a
trajectory-level comparison, not an ablation.

Validation sets are matched by replicate index across arms (al1 of each arm uses the
same 30 validation structures, and so on), and each validation set holds 8 structures
that are metallic at the HSE06 level and 22 that are not. The five replicates of an arm
are not independent. Replicate al1 was seeded from the screening pool; the 70
non-validation seed labels of al2 to al5 were drawn partly from the structures al1 had
already labelled (48 to 50 of the 70), after which each replicate ran 24 cycles of its
own. At cycle 24 any two replicates of arm 01 share 157 to 172 of their 238 training
structures (`revision/results/replicate_overlap.csv`).

Some replicates were continued past cycle 24 (arm 01 al2 to cycle 26, al3 and al5 to
cycle 25; arm 04 direct4 has validation metrics to cycle 25). Their `metrics.csv` files
contain the extra rows, but every reported value, figure and table uses the cycle-24
row or the cycle-24 model. The model trained at cycle 24 is fitted on the ledger
written at the end of cycle 23, `Seed_labels_cycle23.updated.csv`.

## Where each figure and table comes from

Commands are run from the repository root; see each script's `--help` for options.
Output folders under `results/` hold the regenerated figure together with the CSV
values behind it.

| Paper item | Script | Output |
|---|---|---|
| Fig. 1 (workflow schematic) | hand-made, no script | |
| Fig. 2 (universe composition) | `revision/scripts/make_figs.py` | `revision/results/figures/` |
| Fig. 3 (test parity) | `scripts/figures/fig_parity_mean_sd_final.py --mode test` | `results/fig3_test_parity/` |
| Fig. 4, Table S14 (uncertainty diagnostics) | `scripts/figures/fig_nonzbg_uncertainty_main.py` | `results/fig4_tableS14_uncertainty/` |
| Fig. 5 (remaining pool) | `scripts/figures/fig_remaining_pool_hse_vs_uncertainty.py` | `results/fig5_remaining_pool/` |
| Table 2 (cost) | `revision/scripts/universe_counts.py` and the cycle-23 ledgers | printed |
| Table 3, Table S23 (candidates) | `scripts/figures/make_remaining_pool_candidate_table.py` | `results/table3_tableS23_remaining_pool/` |
| Figs. S1 to S3 (feasibility map, element frequencies) | script not included in this release (the candidate-generation code computes t and μ but does not draw these figures) | |
| Fig. S4 (Glazer space groups) | `revision/scripts/make_figs.py` | `revision/results/figures/` |
| Fig. S5 (crystal-system examples) | hand-made, no script | |
| Fig. S6, Table S2 direct-HSE row | `scripts/figures/fig4_gap3_delta_vs_direct_hse_delta_panel.py`, `scripts/audits/variance_delta_vs_direct_trajectory.py` | `results/figS6_delta_vs_direct/`, `audits/delta_vs_direct_variance/` |
| Fig. S7 (no-gate ablation) | `scripts/audits/audit_nogate_vs_gate_fullpool.py`, `scripts/figures/plot_nogate_vs_gate_fullpool_ablation.py` | `audits/nogate_vs_gate_fullpool/` |
| Fig. S8 (AL against random) | `scripts/figures/fig4_1_al_vs_random_overlay.py` | `results/figS8_al_vs_random/` |
| Fig. S9 (DOS-on against DOS-off) | `scripts/figures/fig2_1_dos_ablation_learning_curves.py` | `results/figS9_dos_on_off/` |
| Table S2 (configurations at cycle 24) | `metrics/metrics.csv` of each arm, cycle-24 row | |
| Tables S3 to S6 (feature schemas, leakage control) | `scripts/audits/audit_saved_feature_names_vs_tables.py` | `audits/feature_table_audit/`, `data/feature_schema_*.csv` |
| Table S7, Fig. S10 (permutation importance) | `scripts/evaluation/feature_group_permutation_importance_cycle24.py` (or `_noplot.py`) on the deposited arm-01 cycle-24 models, then `scripts/figures/plot_feature_group_permutation_importance.py`; `scripts/evaluation/dump_cycle24_models_for_feature_importance.sh` optionally refits those models (gate scores within 1e-9, Δ-GPR within 1e-4 eV) | `results/tableS7_figS10_permutation/` |
| Tables S8 to S10, S12, S13, S15 to S17 | `revision/scripts/` (see `revision/README.md`) | `revision/results/` |
| Fig. S11 (gate threshold θ) | `scripts/figures/theta_sensitivity_plot_npj.py` on `SENSITIVITY-RUNS/sweep-thr/` | `results/figS11_theta/` |
| Fig. S12 (m_reg, q_floor) | `scripts/figures/sweep_grid_qfloor_mult_plot_npj.py` on `SENSITIVITY-RUNS/sweep/` | `results/figS12_mreg_qfloor/` |
| Fig. S13 (ZBG quota K_g) | `scripts/figures/k_gate_sensitivity_plot_npj.py` on `SENSITIVITY-RUNS/k-gate/` | `results/figS13_kgate/` |
| Fig. S14 (gate rescue) | `scripts/figures/fig_gate_rescue_evidence.py` | `results/figS14_gate_rescue/` |
| Fig. S15 (validation parity) | `scripts/figures/fig_parity_mean_sd_final.py --mode val` | `results/figS15_validation_parity/` |
| Fig. S16 (calibration curves) | `revision/scripts/calib_curve_sigcal.py`, `revision/scripts/make_figs.py` | `revision/results/` |
| Post hoc threshold audit (ESI Section 4) | `scripts/audits/audit_gate_threshold_false_rescues.py` | `audits/gate_threshold_false_rescues/` |
| Table S11 (gate-training composition) | cycle-23 ledgers minus `phase2_val_ids.csv` of each arm-01 replicate | |
| Tables S18, S19 (VASP settings) | archived VASP inputs; not a data product | |
| Tables S20 to S22 (k-points, SOC, Hubbard U) | additional PBE calculations, deposited as tables | `revision/results/` |

The sensitivity sweeps of Figs. S11 to S13 were run with a single representative
configuration each; `SENSITIVITY-RUNS/` holds only their `metrics/metrics.csv`.

## File formats

**Universe and pool.** `data/universe_master.csv` has one row per universe entry (502
rows, 491 unique structures). `structure_id` is `MP:<mp-id>` for structures from the
Materials Project API subset and `GLAZER:<file name>` for entries of the Glazer bank.
The column `source` records where an entry was taken from (MP API or Glazer bank). The
column `source_paper` gives the classification used in the paper: the 14 Glazer-bank
entries that were started from a Materials Project structure are counted as
MP-derived, which gives 148 MP-derived and 343 Glazer-generated unique structures.
`twin_of` names, for the 11 such entries that are identical after relaxation to an MP
entry, the matching MP entry (and the reverse), and `counted_as_unique` is false for
the 11 Glazer-bank copies. `crystal_system` is lower case for all entries.
`data/trainpool_ml_augmented.csv` contains the same 502 entries with every descriptor
used by the models and the day-0 split (`split_day0` is `test` for the 30 held-out
structures); `data/test_labels_labeled.csv` holds the PBE and HSE06 reference gaps of
the test set. `data/dos_from_output.csv` holds the parsed near-Fermi-level DOS
quantities from which the seven gate descriptors are computed. It was written by
`scripts/data/build_dos_from_output.py` from the PBE band-structure output of the DFT
workflow (`calc_dir` and `doscar_path` give the folder used for each structure). The
script documents how the file was made; the DFT output folders are not part of this
repository, so it cannot be rerun from the repository alone. For the MP API entries,
`pbe_gap` in the universe and pool files is the value reported by the Materials
Project. Once a structure is labelled, the ledgers carry the gap from our own PBE
workflow, and that value is used for training and reconstruction.

**Label ledgers.** `Seed_labels_cycle<N>.updated.csv` in each run folder is the
cumulative ledger at the end of cycle N (columns `structure_id`, `pbe_gap`, `hse_gap`,
`label_status`). `label_status` is `SUCCESS` for an explicit HSE06 calculation and
`METAL_ASSUMED` for a structure treated as metallic, whose `hse_gap` is read as 0.
`labels_store.csv` lists the labels acquired by the run itself, with provenance
columns, and `phase2_val_ids.csv` the 30 validation IDs. `data/seed_0<k>.csv` are the
seed and validation cohorts of replicate k.

**Per-cycle outputs.** `selections/selected_cycle<N>.csv` lists the seven structures
picked in cycle N with every acquisition signal and score. `metrics/metrics.csv` has
one row per cycle with validation metrics (`end2end_hse_MAE`,
`delta_val_MAE_nonmetal`, gate statistics). `metrics/val_pred_cycle<N>.csv` holds the
validation predictions, `test_eval/test_pred_cycle24.csv` and
`test_eval/test_metrics_cycle24.json` the test predictions and metrics (arms 01 and
05), and
`predictions/pred_cycle24.csv` (arm 01) the cycle-24 predictions for the unlabelled
remaining pool. `models/phase2_cycle24.joblib` is the saved cycle-24 model (gate,
Stage B and Stage C regressors, scalers and fill values), and
`models/preproc_cycle<N>.json` the preprocessing metadata of each cycle.

## Consistency with the revised paper

The DFT workflow behind the data is described in Methods and ESI Section 6 of the
revised paper. In short, each structure was relaxed spin-polarised with PBE, a
self-consistent PBE single point wrote the charge density, and a non-self-consistent
band-structure run on 132 high-symmetry k-points gave the PBE bandgap and the DOS
descriptors. No separate tetrahedron-method DOS calculation was made; the DOS
descriptors come from this band-structure run. For 55 structures the workflow applied
its default Hubbard U in the PBE steps, while all HSE06 reference calculations were
run without U, so for these structures the learned target is HSE06 minus PBE+U (ESI
Table S22). The Stage-B regressor is a GPR with an ARD Matérn kernel (ν = 5/2) plus a
white-noise term, with hyperparameters optimised at every cycle.

## Licence

MIT; see [LICENSE](LICENSE).

## Citation

Please cite the paper above and this repository through its Zenodo concept DOI,
10.5281/zenodo.21787222, which always resolves to the latest version.
