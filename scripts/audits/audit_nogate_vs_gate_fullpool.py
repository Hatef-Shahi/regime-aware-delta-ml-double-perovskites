#!/usr/bin/env python3
"""Audit of the gate-aware (arm 01) and no-gate full-pool (arm 02) runs.

Writes per-cycle selection counts, the composition of the labelled ledger
used to train the cycle-N models (default N = 24), and the validation metrics
at cycle N and at the last cycle each run reached, for every replicate.

Run from the repository root:

    python scripts/audits/audit_nogate_vs_gate_fullpool.py \
        --outdir audits/nogate_vs_gate_fullpool

Missing validation-ID files, empty validation-ID files, missing ledgers,
missing selections and missing metrics files raise an error instead of
being skipped. Cycle-numbered files are ordered by their parsed cycle
number, never by file name.
"""

import argparse
from pathlib import Path
import re
import warnings
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]

_ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
_ap.add_argument("--runs-root", default=str(REPO / "OFFICIAL-RUNS"))
_ap.add_argument("--gate-arm", default="01_gate_aware_delta_ml_dos_on")
_ap.add_argument("--nogate-arm", default="02_nogate_unified_delta_fullpool")
_ap.add_argument("--outdir", default=str(REPO / "audits" / "nogate_vs_gate_fullpool"))
_ap.add_argument("--cycle", type=int, default=24,
                 help="budget-matched comparison cycle (default 24); the cycle-N model "
                      "is trained on the Seed_labels_cycle<N-1> ledger")
ARGS = _ap.parse_args()

ROOT = Path(ARGS.runs_root).resolve()
CYCLE = int(ARGS.cycle)

METHOD_DIRS = {
    "gate_AL": ROOT / ARGS.gate_arm,
    "nogate_fullpool_AL": ROOT / ARGS.nogate_arm,
}

OUTDIR = Path(ARGS.outdir).resolve()
OUTDIR.mkdir(parents=True, exist_ok=True)

PBE_ZERO_THR = 0.05
HSE_ZERO_THR = 0.05

LEDGER_CYCLE_RE = re.compile(r"^Seed_labels_cycle(\d+)\.")


def cycle_from_name(path: Path):
    m = re.search(r"cycle(\d+)", path.name)
    return int(m.group(1)) if m else None


def ledger_cycle(path: Path):
    """Cycle number of a Seed_labels_cycle<N>.*.csv ledger, or None."""
    m = LEDGER_CYCLE_RE.match(path.name)
    return int(m.group(1)) if m else None


def latest_by_cycle(paths, parse, what):
    """Return the path with the highest parsed cycle number.

    Files whose cycle cannot be parsed are skipped with a warning, so they
    can never be chosen. Two files with the same highest cycle number raise
    an error, because the choice between them would be arbitrary.
    """
    parsed = []
    for p in paths:
        c = parse(p)
        if c is None:
            warnings.warn(f"{what}: cycle number not parsed, file skipped: {p}")
            continue
        parsed.append((c, p))
    if not parsed:
        return None
    top = max(c for c, _ in parsed)
    best = [p for c, p in parsed if c == top]
    if len(best) > 1:
        raise RuntimeError(f"{what}: several files for cycle {top}: {best}")
    return best[0]


def display_path(path: Path):
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def read_csv(path: Path):
    return pd.read_csv(path, low_memory=False)


def find_col(df, candidates, required=False):
    norm = {re.sub(r"[^a-z0-9]+", "", c.lower()): c for c in df.columns}
    for cand in candidates:
        key = re.sub(r"[^a-z0-9]+", "", cand.lower())
        if key in norm:
            return norm[key]
    if required:
        raise KeyError(f"Could not find one of {candidates}. Available: {list(df.columns)}")
    return None


VAL_ID_FILES = ["phase2_val_ids.csv", "val_ids.csv"]


def read_val_ids(run_dir: Path):
    """Return the set of validation structure IDs of one run.

    Raises FileNotFoundError if no validation-ID file exists and ValueError
    if the file exists but holds no IDs. An empty set would silently let the
    validation structures into the training counts.
    """
    for name in VAL_ID_FILES:
        path = run_dir / name
        if path.exists():
            df = read_csv(path)
            col = find_col(df, ["structure_id", "id", "bank_file"], required=False)
            if col is None:
                col = df.columns[0]
            ids = df[col].dropna().astype(str).str.strip()
            ids = set(ids[ids != ""])
            if not ids:
                raise ValueError(f"Validation-ID file holds no IDs: {path}")
            return ids
    raise FileNotFoundError(
        f"No validation-ID file in {run_dir} (tried: {', '.join(VAL_ID_FILES)})"
    )


def find_final_ledger(run_dir: Path):
    # The cycle-N model is trained on the ledger written at the end of cycle N-1.
    preferred = [
        run_dir / f"Seed_labels_cycle{CYCLE - 1}.updated.csv",
        run_dir / "run_v3" / f"Seed_labels_cycle{CYCLE - 1}.updated.csv",
        run_dir / f"Seed_labels_cycle{CYCLE}.updated.csv",
        run_dir / "run_v3" / f"Seed_labels_cycle{CYCLE}.updated.csv",
    ]
    for p in preferred:
        if p.exists():
            return p

    # Fallbacks: highest numerical cycle, not the last file name in
    # lexicographic order (which would put cycle 9 after cycle 24).
    for pattern in ["Seed_labels_cycle*.updated.csv", "Seed_labels_cycle*.csv"]:
        best = latest_by_cycle(run_dir.rglob(pattern), ledger_cycle, f"{run_dir} ledger")
        if best is not None:
            return best

    raise FileNotFoundError(f"No Seed_labels_cycle*.csv ledger found under {run_dir}")


selection_rows = []
final_label_rows = []
final_metric_rows = []
cycle_metric_rows = []

METRIC_COLS = [
    "end2end_hse_MAE", "end2end_hse_RMSE", "end2end_hse_R2",
    "delta_val_MAE_nonmetal", "delta_val_RMSE_nonmetal", "delta_val_R2_nonmetal",
    "gate_auc", "gate_prec", "gate_rec", "gate_f1",
    "train_n_total", "train_n_nonmetal_used",
    "selected_pbe0", "mode_unified_nogate",
]


# The cycle-N table keeps the column set of the committed cycle24_* files,
# which do not include gate_auc; the last-cycle table keeps it.
CYCLE_METRIC_COLS = [c for c in METRIC_COLS if c != "gate_auc"]


def metric_row(method, run, cycle_key, cycle_value, r):
    cols = CYCLE_METRIC_COLS if cycle_key == "cycle" else METRIC_COLS
    row = {"method": method, "run": run, cycle_key: cycle_value}
    row.update({c: r.get(c, np.nan) for c in cols})
    return row


def write_metrics(rows, by_run_path, summary_path, cycle_key):
    df = pd.DataFrame(rows)
    df.to_csv(by_run_path, index=False)
    num = [c for c in df.select_dtypes(include=[np.number]).columns if c != cycle_key]
    df.groupby("method")[num].agg(["mean", "std", "min", "max"]).to_csv(summary_path)
    return df

for method, root in METHOD_DIRS.items():
    print("\n" + "=" * 90)
    print(method, root)
    print("=" * 90)

    for run_dir in sorted(root.glob("al*")):
        if not run_dir.is_dir():
            continue

        run = run_dir.name

        # ------------------------------------------------------------
        # Selection audit
        # ------------------------------------------------------------
        sel_files = []
        for p in (run_dir / "selections").glob("selected_cycle*.csv"):
            if cycle_from_name(p) is None:
                warnings.warn(f"{method}/{run}: cycle number not parsed, file skipped: {p}")
                continue
            sel_files.append(p)
        sel_files.sort(key=cycle_from_name)

        if not sel_files:
            raise FileNotFoundError(f"{method}/{run}: no selected_cycle*.csv files in {run_dir / 'selections'}")

        cum_pbe0 = 0
        cum_total = 0

        for sf in sel_files:
            cycle = cycle_from_name(sf)
            df = read_csv(sf)

            pbe0_col = find_col(df, ["is_pbe0"], required=False)
            pbe_col = find_col(df, ["pbe_gap_effective", "pbe_gap", "pbe"], required=False)

            if pbe0_col is not None:
                is_pbe0 = pd.to_numeric(df[pbe0_col], errors="coerce").fillna(0).astype(int).values == 1
            elif pbe_col is not None:
                pbe = pd.to_numeric(df[pbe_col], errors="coerce")
                is_pbe0 = pbe.le(PBE_ZERO_THR).fillna(False).values
            else:
                is_pbe0 = np.zeros(len(df), dtype=bool)

            n_total = len(df)
            n_pbe0 = int(np.sum(is_pbe0))

            cum_total += n_total
            cum_pbe0 += n_pbe0

            selection_rows.append({
                "method": method,
                "run": run,
                "cycle": cycle,
                "n_selected": n_total,
                "n_selected_pbe0": n_pbe0,
                "frac_selected_pbe0": n_pbe0 / n_total if n_total else np.nan,
                "cum_selected": cum_total,
                "cum_selected_pbe0": cum_pbe0,
            })

        print(f"[selection] {run}: cycles={len(sel_files)} cumulative_PBE0={cum_pbe0}/{cum_total}")

        # ------------------------------------------------------------
        # Final labelled-ledger composition
        # ------------------------------------------------------------
        ledger_path = find_final_ledger(run_dir)
        val_ids = read_val_ids(run_dir)

        led = read_csv(ledger_path)

        id_col = find_col(led, ["structure_id", "id", "bank_file"], required=True)
        pbe_col = find_col(led, ["pbe_gap", "pbe_gap_effective", "pbe_gap_ev"], required=False)
        hse_col = find_col(led, ["hse_gap", "hse_true", "hse06_gap", "hse_gap_ev"], required=True)
        status_col = find_col(led, ["label_status", "status", "hse_status"], required=False)

        led[id_col] = led[id_col].astype(str).str.strip()
        led[hse_col] = pd.to_numeric(led[hse_col], errors="coerce")

        if pbe_col is not None:
            led[pbe_col] = pd.to_numeric(led[pbe_col], errors="coerce")
        else:
            led[pbe_col] = np.nan
            pbe_col = pbe_col or "__missing_pbe__"

        if status_col is not None:
            status = led[status_col].fillna("").astype(str).str.upper()
        else:
            status = pd.Series([""] * len(led), index=led.index)

        trusted = led[hse_col].notna() | status.eq("METAL_ASSUMED")
        not_val = ~led[id_col].isin(val_ids)

        train = led.loc[trusted & not_val].copy()
        train_status = status.loc[train.index]

        is_zbg = train[pbe_col].le(PBE_ZERO_THR)
        is_metal = train[hse_col].le(HSE_ZERO_THR) | train_status.eq("METAL_ASSUMED")
        is_rescued = is_zbg & (~is_metal) & train[hse_col].gt(HSE_ZERO_THR)

        row = {
            "method": method,
            "run": run,
            "ledger_file": display_path(ledger_path),
            "n_train_nonval": int(len(train)),
            "n_train_baseline_zbg": int(is_zbg.sum()),
            "n_train_zbg_true_metal": int((is_zbg & is_metal).sum()),
            "n_train_zbg_rescued_nonmetal": int(is_rescued.sum()),
            "n_train_non_zbg": int((~is_zbg).sum()),
            "n_val_ids": int(len(val_ids)),
        }

        final_label_rows.append(row)

        print(
            f"[ledger] {run}: train={row['n_train_nonval']} "
            f"ZBG={row['n_train_baseline_zbg']} "
            f"ZBG_metal={row['n_train_zbg_true_metal']} "
            f"ZBG_rescued={row['n_train_zbg_rescued_nonmetal']}"
        )

        # ------------------------------------------------------------
        # Final validation metrics from metrics.csv
        # ------------------------------------------------------------
        metrics_path = run_dir / "metrics" / "metrics.csv"
        if not metrics_path.exists():
            raise FileNotFoundError(f"{method}/{run}: missing {metrics_path}")

        met = read_csv(metrics_path)
        if "cycle" not in met.columns or not len(met):
            raise ValueError(f"{method}/{run}: no cycle column or no rows in {metrics_path}")
        met_cycle = pd.to_numeric(met["cycle"], errors="coerce")

        # Budget-matched cycle (default 24): this is the comparison used in the paper.
        at_cycle = met.loc[met_cycle.eq(CYCLE)].tail(1)
        if not len(at_cycle):
            raise ValueError(f"{method}/{run}: cycle {CYCLE} missing from {metrics_path}")
        cycle_metric_rows.append(metric_row(method, run, "cycle", CYCLE, at_cycle.iloc[0].to_dict()))

        # Last cycle reached by the run (some runs continued past cycle 24).
        final_cycle = int(met_cycle.max())
        last = met.loc[met_cycle.eq(final_cycle)].tail(1)
        final_metric_rows.append(metric_row(method, run, "final_cycle", final_cycle, last.iloc[0].to_dict()))


selection_df = pd.DataFrame(selection_rows)
label_df = pd.DataFrame(final_label_rows)

selection_path = OUTDIR / "selection_counts_by_cycle.csv"
label_path = OUTDIR / "final_label_composition_by_run.csv"
cycle_metric_path = OUTDIR / f"cycle{CYCLE}_validation_metrics_by_run.csv"
cycle_metric_summary_path = OUTDIR / f"cycle{CYCLE}_validation_metrics_summary.csv"
metric_path = OUTDIR / "final_cycle_validation_metrics_by_run.csv"
metric_summary_path = OUTDIR / "final_cycle_validation_metrics_summary.csv"
label_summary_path = OUTDIR / "final_label_composition_summary.csv"

selection_df.to_csv(selection_path, index=False)
label_df.to_csv(label_path, index=False)
cycle_metric_df = write_metrics(cycle_metric_rows, cycle_metric_path, cycle_metric_summary_path, "cycle")
metric_df = write_metrics(final_metric_rows, metric_path, metric_summary_path, "final_cycle")

if not label_df.empty:
    label_numeric_cols = label_df.select_dtypes(include=[np.number]).columns.tolist()
    label_numeric_cols = [c for c in label_numeric_cols if c not in {"n_val_ids"}]

    label_summary = (
        label_df
        .groupby("method")[label_numeric_cols]
        .agg(["mean", "std", "min", "max"])
    )
    label_summary.to_csv(label_summary_path)

print("\n" + "=" * 90)
print("WROTE")
print("=" * 90)
print(selection_path)
print(label_path)
print(cycle_metric_path)
print(cycle_metric_summary_path)
print(metric_path)
print(metric_summary_path)
print(label_summary_path)

print("\nFINAL LABEL COMPOSITION")
print(label_df.to_string(index=False))

print(f"\nVALIDATION METRICS AT CYCLE {CYCLE}")
print(cycle_metric_df.to_string(index=False))

print("\nVALIDATION METRICS AT THE LAST CYCLE OF EACH RUN")
cols = [
    "method", "run", "final_cycle",
    "end2end_hse_MAE",
    "delta_val_MAE_nonmetal",
    "gate_prec", "gate_rec", "gate_f1",
    "train_n_total", "train_n_nonmetal_used",
    "selected_pbe0", "mode_unified_nogate",
]
cols = [c for c in cols if c in metric_df.columns]
print(metric_df[cols].to_string(index=False))

if not selection_df.empty:
    print("\nCUMULATIVE PBE0 SELECTION AT LAST CYCLE")
    last_sel = (
        selection_df
        .sort_values(["method", "run", "cycle"])
        .groupby(["method", "run"])
        .tail(1)
    )
    print(last_sel[["method", "run", "cycle", "cum_selected", "cum_selected_pbe0"]].to_string(index=False))
