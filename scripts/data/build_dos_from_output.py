#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Parser that wrote data/dos_from_output.csv.

For every pool structure it finds the PBE band-structure calculation of the DFT workflow
(<output-root>/<folder>/band_GGA/DOSCAR, where <folder> is the calculation folder of the
structure) and extracts the near-Fermi-level quantities from which the seven gate DOS
descriptors are derived (DOS at E_F, distances to the bracketing DOS points, gap check).

This script documents how data/dos_from_output.csv was made. It needs the DFT output
folders, which are not part of this repository, so it cannot be rerun from the repository
alone. The columns calc_dir and doscar_path of the deposited file record the folder used
for each structure (doscar_path relative to the output root).

Usage (on a machine that holds the DFT output folders):
    python scripts/data/build_dos_from_output.py --pool data/trainpool_ml_augmented.csv \
        --output-root <path to the DFT output folders> --out dos_from_output.csv
"""

import argparse
import re
from pathlib import Path
from typing import Optional, List
import numpy as np
import pandas as pd

def _sid_to_outdir(sid: str) -> str:
    sid = str(sid).strip()
    if sid.startswith("GLAZER:"):
        sid = sid[len("GLAZER:"):]
    elif sid.startswith("MP:"):
        sid = sid[len("MP:"):]
    return sid

def norm_id(x) -> str:
    return str(x).strip()


def parse_mpid(structure_id: str) -> str:
    # "MP:mp-bifn" -> "mp-bifn"
    s = norm_id(structure_id)
    if s.upper().startswith("MP:"):
        return s.split(":", 1)[1].strip()
    return ""


def parse_doscar(doscar: Path, thr_edge: float = 1e-3, thr_gap: float = 1e-3, min_pts: int = 3, eps: float = 1e-12):
    """
    Implements defensible EF-bracketing features:
      - dosEf: DOS at Ef (closest grid point)
      - d_below: Ef - Eb (Eb is grid point just below Ef)
      - d_above: Ea - Ef (Ea is grid point just above Ef)
      - dist_ratio: (d_below+eps)/(d_above+eps)

    Also computes optional "gap-check" diagnostics:
      - vbm/cbm via thr_edge (fallback to bracket points)
      - is_gap via DOS between vbm and cbm: need >=min_pts and maxDOS_between<=thr_gap
      - metal_like = 1 if not is_gap else 0
    """
    try:
        lines = doscar.read_text(errors="ignore").splitlines()
    except Exception:
        return None
    if len(lines) < 7:
        return None

    # DOSCAR header line 6 (1-indexed) -> lines[5]
    toks = lines[5].split()
    if len(toks) < 4:
        return None

    try:
        n = int(float(toks[2]))
        Ef = float(toks[3])
    except Exception:
        return None

    start = 6
    end = min(start + n, len(lines))

    E = []
    D = []
    for i in range(start, end):
        row = lines[i].split()
        if len(row) < 2:
            continue
        try:
            e = float(row[0])
        except Exception:
            continue
        try:
            dos = float(row[1]) + float(row[2]) if len(row) >= 5 else float(row[1])
        except Exception:
            continue
        E.append(e)
        D.append(dos)

    if len(E) < 2:
        return None

    E = np.asarray(E, dtype=float)
    D = np.asarray(D, dtype=float)

    # --- bracket Ef on DOS grid ---
    # ib = last index with E<=Ef, ia = first index with E>=Ef
    ib = np.where(E <= Ef)[0]
    ia = np.where(E >= Ef)[0]
    if ib.size == 0 or ia.size == 0:
        # Ef outside DOS window
        return None

    ib = int(ib[-1])
    ia = int(ia[0])

    Eb = float(E[ib])
    Ea = float(E[ia])
    DOSb = float(D[ib])
    DOSa = float(D[ia])

    d_below = float(Ef - Eb)
    d_above = float(Ea - Ef)
    dist_ratio = float((d_below + eps) / (d_above + eps))

    # DOS at Ef: closest grid point
    dosEf = float(DOSb if (Ef - Eb) <= (Ea - Ef) else DOSa)
    dosEf_minbracket = float(min(DOSb, DOSa))

    # --- threshold "edges" (only meaningful if actually gapped) ---
    # vbi = last j with E<Ef and DOS>thr_edge; cbi = first j with E>Ef and DOS>thr_edge
    vbi = np.where((E < Ef) & (D > thr_edge))[0]
    cbi = np.where((E > Ef) & (D > thr_edge))[0]

    vbi = int(vbi[-1]) if vbi.size else ib
    cbi = int(cbi[0]) if cbi.size else ia

    vbm = float(E[vbi])
    cbm = float(E[cbi])
    gap = float(cbm - vbm)

    # --- gap validation by DOS between vbm and cbm ---
    n_gap_pts = int(cbi - vbi - 1)
    if n_gap_pts > 0:
        between = D[(vbi + 1):cbi]
        maxDOS_between = float(np.max(between)) if between.size else float("inf")
    else:
        maxDOS_between = float("inf")  # same behavior as awk (forces is_gap=0)

    is_gap = int((n_gap_pts >= int(min_pts)) and (maxDOS_between <= float(thr_gap)))
    metal_like = int(0 if is_gap else 1)

    out = {
        # identity
        "Ef": float(Ef),

        # defensible EF-bracket features
        "dosEf": dosEf,
        "d_below": d_below,
        "d_above": d_above,
        "dist_ratio": dist_ratio,

        # bracket metadata (auditing)
        "Eb": Eb,
        "Ea": Ea,
        "DOSb": DOSb,
        "DOSa": DOSa,
        "dosEf_minbracket": dosEf_minbracket,

        # optional diagnostics (auditing/plots)
        "vbm": vbm,
        "cbm": cbm,
        "gap": gap,
        "n_gap_pts": n_gap_pts,
        "maxDOS_between": maxDOS_between,
        "thr_edge": float(thr_edge),
        "thr_gap": float(thr_gap),
        "min_pts": int(min_pts),
        "is_gap": is_gap,
        "metal_like": metal_like,
    }
    return out



def build_output_dir_index(output_root: Path):
    # index all first-level subdirectories once (fast)
    dirs = []
    try:
        for p in output_root.iterdir():
            if p.is_dir():
                dirs.append(p.name)
    except Exception:
        pass
    return dirs

def resolve_calc_dir(row, output_root: Path, out_dirs: List[Path],
                     dir_col: Optional[str], bank_col: Optional[str], id_col: str):
    """
    Try (in order): dir_name -> bank_file -> structure_id
    Normalizes:
      - strips leading "Output/" and any full path -> basename
      - strips "GLAZER:" / "MP:" prefixes from IDs
    If not found, fallback search by mpid substring in output subdir names.
    """
    sid = norm_id(row[id_col])

    def _strip_prefixes(s: str) -> str:
        s = str(s).strip()
        if s.startswith("GLAZER:"):
            s = s[len("GLAZER:"):]
        elif s.startswith("MP:"):
            s = s[len("MP:"):]
        return s

    def _clean_candidate(c: str) -> str:
        c = str(c).strip().rstrip("/")

        # remove leading Output/ if present
        c = re.sub(r"^\.?/?Output/", "", c)

        # if a path was passed, keep only the folder/file name
        c = Path(c).name

        # strip dataset prefixes if they accidentally appear in candidate
        c = _strip_prefixes(c)

        return c

    candidates = []
    if dir_col and dir_col in row and str(row[dir_col]).strip() and str(row[dir_col]).lower() != "nan":
        candidates.append(str(row[dir_col]).strip())
    if bank_col and bank_col in row and str(row[bank_col]).strip() and str(row[bank_col]).lower() != "nan":
        candidates.append(str(row[bank_col]).strip())

    # Try both raw SID and prefix-stripped SID
    candidates.append(sid)
    candidates.append(_strip_prefixes(sid))

    cand_norm = [_clean_candidate(c) for c in candidates if str(c).strip()]

    # direct checks
    for dname in cand_norm:
        fp = output_root / dname / "band_GGA" / "DOSCAR"
        if fp.exists():
            return dname, fp

    # fallback: MP substring search (match mp-xxxx in Output subdir name)
    mpid = parse_mpid(sid)
    if mpid:
        hits = []
        for d in out_dirs:
            # out_dirs may be Path or str; normalize to a name string
            dn = d.name if isinstance(d, Path) else Path(str(d)).name
            if mpid in dn:
                hits.append(dn)

        for dn in hits[:25]:
            fp = output_root / dn / "band_GGA" / "DOSCAR"
            if fp.exists():
                return dn, fp

    return "", None



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="trainpool_ml_augmented.csv")
    ap.add_argument("--output-root", default="Output")
    ap.add_argument("--thr-edge", type=float, default=1e-3)
    ap.add_argument("--thr-gap", type=float, default=1e-3)
    ap.add_argument("--min-pts", type=int, default=3)
    ap.add_argument("--eps", type=float, default=1e-12)
    ap.add_argument("--out", default="dos_from_output.csv")
    ap.add_argument("--id-col", default="structure_id")
    ap.add_argument("--dir-col", default="dir_name")
    ap.add_argument("--bank-col", default="bank_file")
    ap.add_argument("--thr", type=float, default=1e-3)
    args = ap.parse_args()

    pool_path = Path(args.pool).resolve()
    output_root = Path(args.output_root).resolve()
    out_path = Path(args.out).resolve()

    df = pd.read_csv(pool_path)
    if args.id_col not in df.columns:
        raise SystemExit(f"Missing id col {args.id_col} in {pool_path}")

    dir_col = args.dir_col if args.dir_col in df.columns else None
    bank_col = args.bank_col if args.bank_col in df.columns else None

    df[args.id_col] = df[args.id_col].astype(str).map(norm_id)

    out_dirs = build_output_dir_index(output_root)

    rows = []
    n = len(df)
    found = 0
    parsed = 0
    missing = 0
    failed = 0

    for i, row in df.iterrows():
        sid = norm_id(row[args.id_col])
        dname, dos_fp = resolve_calc_dir(row, output_root, out_dirs, dir_col, bank_col, args.id_col)

        rec = {
            args.id_col: sid,
            "calc_dir": dname,
            "doscar_path": str(dos_fp.relative_to(output_root)) if dos_fp else "",
            "has_doscar": int(dos_fp is not None),
            "parsed_ok": 0,
        }

        if dos_fp is None:
            missing += 1
            rows.append(rec)
            continue

        found += 1
        met = parse_doscar(
            dos_fp,
            thr_edge=args.thr_edge,
            thr_gap=args.thr_gap,
            min_pts=args.min_pts,
            eps=args.eps,
        )


        rec.update(met)
        rec["parsed_ok"] = 1
        parsed += 1
        rows.append(rec)

        if (i + 1) % 200 == 0:
            print(f"[progress] {i+1}/{n}  parsed_ok={parsed}  missing={missing}  failed={failed}")

    out_df = pd.DataFrame(rows)

    # stable column order
    cols_first = [args.id_col, "calc_dir", "doscar_path", "has_doscar", "parsed_ok",
                 "Ef", "vbm", "cbm", "dv", "dc", "dist_ratio", "energy_ratio", "inv_energy_ratio", "dos_Ef"]
    cols = [c for c in cols_first if c in out_df.columns] + [c for c in out_df.columns if c not in cols_first]
    out_df = out_df[cols]

    out_df.to_csv(out_path, index=False)

    print("\n=== DOSCAR coverage summary ===")
    print("pool rows:", n)
    print("has DOSCAR:", found)
    print("parsed OK:", parsed)
    print("missing DOSCAR:", missing)
    print("parse failed:", failed)
    print("wrote:", str(out_path))

    # show a few missing ids
    miss_ids = out_df.loc[out_df["has_doscar"] == 0, args.id_col].head(20).tolist()
    if miss_ids:
        print("\nfirst missing IDs:", miss_ids)


if __name__ == "__main__":
    main()

