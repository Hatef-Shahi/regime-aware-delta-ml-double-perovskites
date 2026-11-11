#!/usr/bin/env bash
# Refit the cycle-24 models of the five arm-01 replicates from their cycle-23 ledgers,
# with the settings of the official runs, and dump them for the feature-group
# permutation importance (ESI Table S7, Fig. S10).
#
# This step is optional: the deposited OFFICIAL-RUNS/01_gate_aware_delta_ml_dos_on/al*/models/
# phase2_cycle24.joblib files are the cycle-24 models, and
# scripts/evaluation/feature_group_permutation_importance_cycle24.py reproduces Table S7
# from them directly. A refit reproduces the deposited gate scores to 1e-9 and the Delta-GPR
# predictions to within 1e-4 eV (optimiser restarts).
#
# Usage (from the repository root):
#   bash scripts/evaluation/dump_cycle24_models_for_feature_importance.sh [OUT_ROOT]
# Set PYTHON to choose the interpreter (default: python).
set -euo pipefail

PY="${PYTHON:-python}"
AL_PY="scripts/workflow/active_learning_gpr.py"

ROOT="OFFICIAL-RUNS/01_gate_aware_delta_ml_dos_on"
OUT_ROOT="${1:-results/tableS7_figS10_permutation/model_dumps}"
CYCLE=24

TOPK=7
K_GATE=2
PBE_ZERO_THR=0.05
METAL_THR_EV=0.05
GATE_DECISION_THR=0.16
GATE_ENTROPY_Q=0.75
RESTARTS=5

GATE_WEIGHT_LAM=0.2
GATE_WEIGHT_SCALE=0.10
W_GATE_DOS=0.3
W_GATE_RATIO=0.2

TARGET_MODE="delta"
ACQ_MODE="al"

for r in al1 al2 al3 al4 al5; do
  RUN="$ROOT/$r"
  OUT="$OUT_ROOT/$r"

  if [[ -s "$RUN/pool_trainval.csv" ]]; then
    POOL="$RUN/pool_trainval.csv"
  elif [[ -s "$RUN/run_v3/pool_trainval.csv" ]]; then
    POOL="$RUN/run_v3/pool_trainval.csv"
  else
    echo "[ERROR] cannot find pool_trainval.csv for $r"
    exit 1
  fi

  if [[ -s "$RUN/Seed_labels_cycle23.updated.csv" ]]; then
    SEED="$RUN/Seed_labels_cycle23.updated.csv"
  elif [[ -s "$RUN/run_v3/Seed_labels_cycle23.updated.csv" ]]; then
    SEED="$RUN/run_v3/Seed_labels_cycle23.updated.csv"
  else
    echo "[ERROR] cannot find Seed_labels_cycle23.updated.csv for $r"
    echo "Available seed files:"
    find "$RUN" -maxdepth 3 -type f \( -iname "*Seed*cycle*.csv" -o -iname "*seed*cycle*.csv" \) | sort
    exit 1
  fi

  mkdir -p "$OUT"
  # Reuse the replicate's frozen validation set; the driver stops if none is present
  # after cycle 0 rather than drawing a new one.
  cp "$RUN/phase2_val_ids.csv" "$OUT/phase2_val_ids.csv"

  echo
  echo "============================================================"
  echo "[dump] $r cycle $CYCLE"
  echo "[dump] seed = $SEED"
  echo "[dump] pool = $POOL"
  echo "[dump] out  = $OUT"
  echo "============================================================"

  "$PY" "$AL_PY" select \
    --cycle "$CYCLE" \
    --seed "$SEED" \
    --pool "$POOL" \
    --dos-csv data/dos_from_output.csv \
    --target-mode "$TARGET_MODE" \
    --outdir "$OUT" \
    --id-col structure_id \
    --hse-col hse_gap \
    --status-col label_status \
    --pool-pbe-col pbe_gap \
    --split-col split \
    --restarts "$RESTARTS" \
    --topk "$TOPK" \
    --acquisition "$ACQ_MODE" \
    --random-seed 42 \
    --metal-thr-ev "$METAL_THR_EV" \
    --pbe-zero-thr "$PBE_ZERO_THR" \
    --gate-decision-thr "$GATE_DECISION_THR" \
    --k-gate "$K_GATE" \
    --gate-entropy-quantile "$GATE_ENTROPY_Q" \
    --gate-weight-lam "$GATE_WEIGHT_LAM" \
    --gate-weight-scale "$GATE_WEIGHT_SCALE" \
    --w-gate-dos "$W_GATE_DOS" \
    --w-gate-ratio "$W_GATE_RATIO" \
    --dump-model \
    --dump-ard \
    --dump-preproc \
    --dump-val-preds \
    --dump-diversity \
    --clf-model logreg \
    --clf-calibrate \
    --clf-calibration-method sigmoid \
    --clf-calibration-cv 3
done

echo
echo "[OK] model/ARD dump complete."
