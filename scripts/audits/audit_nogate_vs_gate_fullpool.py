#!/usr/bin/env python3

from pathlib import Path
import re
import numpy as np
import pandas as pd


ROOT = Path("/home/hatef/Desktop/Final-thesis-plots/experiments")

METHOD_DIRS = {
    "gate_AL": ROOT / "paper-runs" / "delta_kg2_m6_q035_theta016_dosON",
    "nogate_fullpool_AL": ROOT / "paper-runs" / "nogate_unified_delta_fullpool",
}

OUTDIR = ROOT / "artifacts" / "ablations" / "nogate_vs_gate_fullpool"
OUTDIR.mkdir(parents=True, exist_ok=True)

PBE_ZERO_THR = 0.05
HSE_ZERO_THR = 0.05


def cycle_from_name(path: Path):
    m = re.search(r"cycle(\d+)", path.name)
    return int(m.group(1)) if m else None


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


def read_val_ids(run_dir: Path):
    for name in ["phase2_val_ids.csv", "val_ids.csv"]:
        path = run_dir / name
        if path.exists():
            df = read_csv(path)
            col = find_col(df, ["structure_id", "id", "bank_file"], required=False)
            if col is None:
                col = df.columns[0]
            return set(df[col].astype(str).str.strip())
    return set()


def find_final_ledger(run_dir: Path):
    preferred = [
        run_dir / "Seed_labels_cycle23.updated.csv",
        run_dir / "run_v3" / "Seed_labels_cycle23.updated.csv",
        run_dir / "Seed_labels_cycle24.updated.csv",
        run_dir / "run_v3" / "Seed_labels_cycle24.updated.csv",
    ]
    for p in preferred:
        if p.exists():
            return p

    hits = sorted(run_dir.rglob("Seed_labels_cycle*.updated.csv"))
    if hits:
        return hits[-1]

    hits = sorted(run_dir.rglob("Seed_labels_cycle*.csv"))
    if hits:
        return hits[-1]

    return None


selection_rows = []
final_label_rows = []
final_metric_rows = []

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
        sel_files = sorted(
            (run_dir / "selections").glob("selected_cycle*.csv"),
            key=lambda p: cycle_from_name(p) if cycle_from_name(p) is not None else 10**9,
        )

        if not sel_files:
            print(f"[WARN] {method}/{run}: no selected_cycle*.csv files found")

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

        if ledger_path is None:
            print(f"[WARN] {method}/{run}: no final ledger found")
        else:
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
                "ledger_file": str(ledger_path),
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

        if metrics_path.exists():
            met = read_csv(metrics_path)
            if "cycle" in met.columns and len(met):
                final_cycle = int(pd.to_numeric(met["cycle"], errors="coerce").max())
                last = met.loc[pd.to_numeric(met["cycle"], errors="coerce").eq(final_cycle)].tail(1).copy()

                if len(last):
                    r = last.iloc[0].to_dict()
                    final_metric_rows.append({
                        "method": method,
                        "run": run,
                        "final_cycle": final_cycle,
                        "end2end_hse_MAE": r.get("end2end_hse_MAE", np.nan),
                        "end2end_hse_RMSE": r.get("end2end_hse_RMSE", np.nan),
                        "end2end_hse_R2": r.get("end2end_hse_R2", np.nan),
                        "delta_val_MAE_nonmetal": r.get("delta_val_MAE_nonmetal", np.nan),
                        "delta_val_RMSE_nonmetal": r.get("delta_val_RMSE_nonmetal", np.nan),
                        "delta_val_R2_nonmetal": r.get("delta_val_R2_nonmetal", np.nan),
                        "gate_auc": r.get("gate_auc", np.nan),
                        "gate_prec": r.get("gate_prec", np.nan),
                        "gate_rec": r.get("gate_rec", np.nan),
                        "gate_f1": r.get("gate_f1", np.nan),
                        "train_n_total": r.get("train_n_total", np.nan),
                        "train_n_nonmetal_used": r.get("train_n_nonmetal_used", np.nan),
                        "selected_pbe0": r.get("selected_pbe0", np.nan),
                        "mode_unified_nogate": r.get("mode_unified_nogate", np.nan),
                    })


selection_df = pd.DataFrame(selection_rows)
label_df = pd.DataFrame(final_label_rows)
metric_df = pd.DataFrame(final_metric_rows)

selection_path = OUTDIR / "selection_counts_by_cycle.csv"
label_path = OUTDIR / "final_label_composition_by_run.csv"
metric_path = OUTDIR / "final_validation_metrics_by_run.csv"
metric_summary_path = OUTDIR / "final_validation_metrics_summary.csv"
label_summary_path = OUTDIR / "final_label_composition_summary.csv"

selection_df.to_csv(selection_path, index=False)
label_df.to_csv(label_path, index=False)
metric_df.to_csv(metric_path, index=False)

if not metric_df.empty:
    metric_numeric_cols = metric_df.select_dtypes(include=[np.number]).columns.tolist()
    metric_numeric_cols = [c for c in metric_numeric_cols if c not in {"final_cycle"}]

    metric_summary = (
        metric_df
        .groupby("method")[metric_numeric_cols]
        .agg(["mean", "std", "min", "max"])
    )
    metric_summary.to_csv(metric_summary_path)

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
print(metric_path)
print(metric_summary_path)
print(label_summary_path)

print("\nFINAL LABEL COMPOSITION")
print(label_df.to_string(index=False))

print("\nFINAL VALIDATION METRICS")
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
