#!/usr/bin/env python3

from pathlib import Path
import pandas as pd

ROOT = Path.cwd()

GATE_ROOT = ROOT / "paper-runs" / "delta_kg2_m6_q035_theta016_dosON"
NOGATE_ROOT = ROOT / "paper-runs" / "nogate_unified_delta_fullpool"

def find_val_file(run_dir):
    candidates = [
        run_dir / "phase2_val_ids.csv",
        run_dir / "val_ids.csv",
        run_dir / "run_v3" / "phase2_val_ids.csv",
        run_dir / "run_v3" / "val_ids.csv",
    ]
    for p in candidates:
        if p.exists():
            return p

    hits = sorted(run_dir.rglob("*val*ids*.csv"))
    return hits[0] if hits else None

def read_ids(path):
    df = pd.read_csv(path)
    if "structure_id" in df.columns:
        col = "structure_id"
    else:
        col = df.columns[0]
    return set(df[col].astype(str).str.strip())

def main():
    rows = []
    all_sets = {}

    for i in range(1, 6):
        run = f"al{i}"
        gate_dir = GATE_ROOT / run
        nogate_dir = NOGATE_ROOT / run

        gate_file = find_val_file(gate_dir)
        nogate_file = find_val_file(nogate_dir)

        if gate_file is None or nogate_file is None:
            rows.append({
                "replicate": run,
                "gate_file": str(gate_file) if gate_file else "MISSING",
                "nogate_file": str(nogate_file) if nogate_file else "MISSING",
                "n_gate": None,
                "n_nogate": None,
                "matched_gate_vs_nogate": False,
                "only_in_gate": None,
                "only_in_nogate": None,
            })
            continue

        gate_ids = read_ids(gate_file)
        nogate_ids = read_ids(nogate_file)

        all_sets[f"gate_{run}"] = gate_ids
        all_sets[f"nogate_{run}"] = nogate_ids

        rows.append({
            "replicate": run,
            "gate_file": str(gate_file),
            "nogate_file": str(nogate_file),
            "n_gate": len(gate_ids),
            "n_nogate": len(nogate_ids),
            "matched_gate_vs_nogate": gate_ids == nogate_ids,
            "only_in_gate": len(gate_ids - nogate_ids),
            "only_in_nogate": len(nogate_ids - gate_ids),
        })

    out = pd.DataFrame(rows)

    print("\nValidation-set matching by replicate")
    print(out.to_string(index=False))

    # Check whether all replicate validation sets are identical or replicate-specific.
    unique_sets = {}
    for name, ids in all_sets.items():
        key = tuple(sorted(ids))
        unique_sets.setdefault(key, []).append(name)

    print("\nUnique validation ID sets across all gate/no-gate runs:")
    print("n_unique_sets =", len(unique_sets))
    for j, (ids, names) in enumerate(unique_sets.items(), start=1):
        print(f"set {j}: n_ids={len(ids)} used_by={', '.join(names)}")

    outdir = ROOT / "artifacts" / "diagnostics" / "validation_matching"
    outdir.mkdir(parents=True, exist_ok=True)
    out.to_csv(outdir / "validation_matching_by_replicate.csv", index=False)

    with open(outdir / "validation_set_groups.txt", "w") as f:
        f.write(f"n_unique_sets = {len(unique_sets)}\n")
        for j, (ids, names) in enumerate(unique_sets.items(), start=1):
            f.write(f"set {j}: n_ids={len(ids)} used_by={', '.join(names)}\n")

    print("\nWrote:")
    print(outdir / "validation_matching_by_replicate.csv")
    print(outdir / "validation_set_groups.txt")

if __name__ == "__main__":
    main()
