"""Fig. 2a-c (universe composition), Fig. S4 (Glazer post-relaxation space groups) and
Fig. S16 (calibration curves).

Usage (from the repository root):
    python revision/scripts/make_figs.py revision/results/figures revision/results/calibration_curve_with_sigcal.csv
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"  # single-threaded BLAS so that repeated runs give identical output
import os, sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT = sys.argv[1]; CAL = sys.argv[2]
sys.argv = sys.argv[:1]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from universe_counts import u
C = {"MP-derived": "#2a78d6", "Glazer-generated": "#eb6834"}
INK, MUTED, GRID = "#222222", "#666666", "#e3e3e0"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})
def tex_sg(s):
    m = {"Fm-3m": r"$Fm\bar{3}m$", "I4/m": r"$I4/m$", "P2_1/c": r"$P2_1/c$", "R-3": r"$R\bar{3}$", "P2_1/m": r"$P2_1/m$",
         "Pn-3": r"$Pn\bar{3}$", "C2/m": r"$C2/m$", "P1": r"$P1$", "R3": r"$R3$", "P-1": r"$P\bar{1}$", "I4/mmm": r"$I4/mmm$"}
    return m.get(s, s)
order = ["MP-derived", "Glazer-generated"]
def bar_labels(ax, bars, fmt):
    for b in bars:
        ax.annotate(fmt(b.get_height()), (b.get_x()+b.get_width()/2, b.get_height()), xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, color=INK)
# ---- Fig 2a,b
fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.6), gridspec_kw=dict(wspace=0.35))
n = u.src.value_counts()[order]; N = n.sum()
bars = a.bar(range(2), n.values, width=0.6, color=[C[k] for k in order], linewidth=0)
for bb, k in zip(bars, order):
    a.annotate(f"{n[k]} ({100*n[k]/N:.1f} %)", (bb.get_x()+bb.get_width()/2, bb.get_height()), xytext=(0, 3),
               textcoords="offset points", ha="center", va="bottom", fontsize=9, color=INK)
a.set_xticks(range(2)); a.set_xticklabels(order); a.set_ylabel("Number of structures"); a.set_ylim(0, 400)
a.set_title(f"(a) Universe composition (N = {N} unique)", loc="left", fontsize=10, color=INK)
a.yaxis.grid(True, color=GRID); a.set_axisbelow(True)
zero = u.pbe_gap == 0
ct = pd.crosstab(u.src, zero).loc[order]
x = np.arange(2); w = 0.36
for i, k in enumerate(order):
    vals = [ct.loc[k, True], ct.loc[k, False]]
    bars = b.bar(x + (i-0.5)*w, vals, width=w-0.03, color=C[k], linewidth=0, label=k)
    bar_labels(b, bars, lambda v: f"{int(v)}")
b.set_xticks(x); b.set_xticklabels([r"PBE gap = 0", r"PBE gap > 0"]); b.set_ylabel("Number of structures"); b.set_ylim(0, 300)
b.set_title("(b) Zero vs. nonzero PBE bandgap", loc="left", fontsize=10, color=INK)
b.legend(loc="upper left"); b.yaxis.grid(True, color=GRID); b.set_axisbelow(True)
fig.savefig(f"{OUT}/fig2ab_composition.png", dpi=300, bbox_inches="tight"); plt.close(fig)
# ---- Fig 2c
top = u.spg_symbol.value_counts().head(10).index.tolist()
ct = pd.crosstab(u.spg_symbol, u.src).reindex(top).fillna(0)[order]
fig, ax = plt.subplots(figsize=(9, 3.4))
x = np.arange(len(top)); w = 0.4
for i, k in enumerate(order):
    bars = ax.bar(x + (i-0.5)*w, ct[k].values, width=w-0.03, color=C[k], linewidth=0, label=k)
    bar_labels(ax, [bb for bb in bars if bb.get_height() > 0], lambda v: f"{int(v)}")
ax.set_xticks(x); ax.set_xticklabels([tex_sg(s) for s in top]); ax.set_ylabel("Number of structures"); ax.set_ylim(0, 175)
ax.set_title(f"(c) Ten most frequent space groups (post-relaxation, {int(ct.values.sum())} of {len(u)} structures)", loc="left", fontsize=10, color=INK)
ax.legend(loc="upper right"); ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
fig.savefig(f"{OUT}/fig2c_spacegroups.png", dpi=300, bbox_inches="tight"); plt.close(fig)
# ---- Fig S4
g = u[u.src == "Glazer-generated"]
tmpl = ["Fm-3m", "I4/m", "R-3", "P2_1/c"]
other = [s for s in g.spg_symbol.value_counts().index if s not in tmpl]
cats = tmpl + other
cnt = g.spg_symbol.value_counts().reindex(cats).fillna(0)
fig, ax = plt.subplots(figsize=(7, 3.6))
cols = [C["Glazer-generated"] if s in tmpl else "#9a9a94" for s in cats]
bars = ax.bar(range(len(cats)), cnt.values, width=0.6, color=cols, linewidth=0)
bar_labels(ax, bars, lambda v: f"{int(v)}")
ax.set_xticks(range(len(cats))); ax.set_xticklabels([tex_sg(s) for s in cats]); ax.set_ylabel("Number of structures"); ax.set_ylim(0, 170)
ax.set_xlabel("Space group after PBE relaxation")
ax.set_title(f"Glazer-generated subset (n = {len(g)}); prototype templates in orange, other groups in grey", loc="left", fontsize=9.5, color=INK)
ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
fig.savefig(f"{OUT}/SI_glazer_prototype_usage.png", dpi=300, bbox_inches="tight"); plt.close(fig)
# ---- Fig S16
cc = pd.read_csv(CAL)
fig, axs = plt.subplots(1, 2, figsize=(8, 3.8), sharey=True, gridspec_kw=dict(wspace=0.12))
for ax, s, ttl, nn in [(axs[0], "val", "(A) Validation", 96), (axs[1], "test", "(B) Held-out test", 115)]:
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--", label="Perfect calibration")
    ax.plot(cc.expected, cc[f"{s}_raw"], color=C["MP-derived"], lw=2, marker="o", ms=4, label=r"Raw $\sigma_\Delta$")
    ax.plot(cc.expected, cc[f"{s}_sigcal"], color=C["Glazer-generated"], lw=2, marker="s", ms=4, label=r"Recalibrated $\sigma_{\mathrm{cal}}$")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_xlabel("Nominal fraction"); ax.set_title(f"{ttl} (n = {nn})", loc="left", fontsize=10, color=INK)
    ax.grid(True, color=GRID); ax.set_axisbelow(True)
axs[0].set_ylabel("Observed fraction"); axs[0].legend(loc="upper left", fontsize=8.5)
fig.savefig(f"{OUT}/SI_calibration_curve.pdf", bbox_inches="tight"); fig.savefig(f"{OUT}/SI_calibration_curve.png", dpi=200, bbox_inches="tight"); plt.close(fig)
print("done")
