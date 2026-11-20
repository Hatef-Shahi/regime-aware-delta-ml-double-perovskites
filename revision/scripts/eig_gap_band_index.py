#!/usr/bin/env python3
"""Bandgap from a VASP EIGENVAL file by band index: VBM = highest energy of band N_occ,
CBM = lowest energy of band N_occ+1, over all k-points including zero-weight ones.
N_occ = NELECT/2 (ISPIN=1) or NELECT (non-collinear, pass 'ncl').
Usage: eig_gap_band_index.py EIGENVAL [ncl]"""
import sys
f = sys.argv[1]; ncl = len(sys.argv) > 2
L = open(f).read().split("\n"); ns = int(L[0].split()[3])
ne, nk, nb = (int(float(x)) for x in L[5].split()[:3])
nocc = ne if ncl else ne // 2
if ns == 2: sys.exit("ISPIN=2: not handled")
i = 7; v = (-1e9, None); c = (1e9, None)
for k in range(nk):
    while not L[i].strip(): i += 1
    kp = " ".join(L[i].split()[:3]); i += 1
    e = [float(L[i + j].split()[1]) for j in range(nb)]; i += nb
    if e[nocc - 1] > v[0]: v = (e[nocc - 1], kp)
    if e[nocc] < c[0]: c = (e[nocc], kp)
print(f"{f}: N_occ {nocc}  VBM {v[0]:.4f} at ({v[1]})  CBM {c[0]:.4f} at ({c[1]})  gap {c[0]-v[0]:.4f} eV")
