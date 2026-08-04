#!/usr/bin/env python3

from pathlib import Path
import re
import json
import joblib
import pandas as pd

ROOT = Path.cwd()
RUN_ROOT = ROOT / "paper-runs" / "delta_kg2_m6_q035_theta016_dosON"
TABLE_TXT = Path("/mnt/data/Pasted text(5).txt")  # only works inside ChatGPT sandbox; ignore locally if unavailable

# Local fallback: paste your table file path here if you saved it locally.
LOCAL_TABLE_TXT = ROOT / "Pasted text(5).txt"

def norm_name(s):
    return (
        str(s)
        .replace("\\_", "_")
        .replace("\\", "")
        .strip()
    )

def load_saved_names(run="al1", cycle=24):
    model_path = RUN_ROOT / run / "models" / f"phase2_cycle{cycle}.joblib"
    preproc_path = RUN_ROOT / run / "models" / f"preproc_cycle{cycle}.json"

    payload = joblib.load(model_path)
    pre = json.loads(preproc_path.read_text())

    clf = payload.get("clf_feature_names")
    reg = payload.get("reg_feature_names")

    if clf is None:
        clf = list(payload.get("clf_fill_values", pre.get("A_fill_values", {})).keys()) + list(pre.get("A_bool_cols", []))
    if reg is None:
        reg = list(payload.get("reg_fill_values", pre.get("B_fill_values", {})).keys()) + list(pre.get("B_bool_cols", []))

    return [norm_name(x) for x in clf], [norm_name(x) for x in reg]

def parse_tables(path):
    txt = path.read_text()

    names = [norm_name(x) for x in re.findall(r"\\texttt\{\\seqsplit\{([^}]*)\}\}", txt)]

    # Your pasted file contains gate table first, regression table second.
    gate_table = names[:119]
    reg_table = names[119:119+114]

    return gate_table, reg_table

def main():
    clf_saved, reg_saved = load_saved_names(run="al1", cycle=24)

    print("\nSaved model feature counts")
    print("gate saved:", len(clf_saved))
    print("reg saved :", len(reg_saved))

    outdir = ROOT / "artifacts" / "diagnostics" / "feature_table_audit"
    outdir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({"gate_feature_saved": clf_saved}).to_csv(outdir / "gate_saved_feature_names.csv", index=False)
    pd.DataFrame({"reg_feature_saved": reg_saved}).to_csv(outdir / "reg_saved_feature_names.csv", index=False)

    table_path = LOCAL_TABLE_TXT if LOCAL_TABLE_TXT.exists() else TABLE_TXT
    if not table_path.exists():
        print("\nNo local pasted-table text file found. Saved feature-name CSVs were written.")
        print(outdir / "gate_saved_feature_names.csv")
        print(outdir / "reg_saved_feature_names.csv")
        return

    gate_table, reg_table = parse_tables(table_path)

    print("\nTable feature counts")
    print("gate table:", len(gate_table))
    print("reg table :", len(reg_table))

    gate_extra = [x for x in gate_table if x not in clf_saved]
    gate_missing = [x for x in clf_saved if x not in gate_table]

    reg_extra = [x for x in reg_table if x not in reg_saved]
    reg_missing = [x for x in reg_saved if x not in reg_table]

    print("\nGate descriptors in table but not saved fitted model:")
    for x in gate_extra:
        print("  REMOVE:", x)

    print("\nGate descriptors in saved fitted model but missing from table:")
    for x in gate_missing:
        print("  ADD:", x)

    print("\nRegression descriptors in table but not saved fitted model:")
    for x in reg_extra:
        print("  REMOVE:", x)

    print("\nRegression descriptors in saved fitted model but missing from table:")
    for x in reg_missing:
        print("  ADD:", x)

    pd.DataFrame({"remove_from_gate_table": gate_extra}).to_csv(outdir / "remove_from_gate_table.csv", index=False)
    pd.DataFrame({"remove_from_reg_table": reg_extra}).to_csv(outdir / "remove_from_reg_table.csv", index=False)
    pd.DataFrame({"missing_from_gate_table": gate_missing}).to_csv(outdir / "missing_from_gate_table.csv", index=False)
    pd.DataFrame({"missing_from_reg_table": reg_missing}).to_csv(outdir / "missing_from_reg_table.csv", index=False)

    print("\nWrote audit files to:")
    print(outdir)

if __name__ == "__main__":
    main()
