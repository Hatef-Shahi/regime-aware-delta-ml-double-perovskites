"""Summary numbers quoted from the SHAP and ARD analysis (R2.2, R3.12, R3.13; ESI Table S10).

Reads <OUT>/shap_importance.csv and <OUT>/ard_length_scales_full.csv written by shap_gpr.py
and writes <OUT>/shap_summary.csv with the share of the total mean |SHAP| carried by the
top-5 and top-10 Delta-GPR features and by the B-site electronegativity, electron-affinity
and ionisation descriptors (names starting en_mulliken_B, en_pauling_B, ea_B, ie1_B or
dchi_B), the Spearman correlation between the GPR and random-forest attributions, the
top-10 overlap, and the number of ARD length scales >= 900, < 100 and < 10.

Usage (from the repository root):
    python revision/scripts/shap_derived.py revision/results
"""
import sys
import pandas as pd
from scipy.stats import spearmanr

OUT = sys.argv[1]
s = pd.read_csv(f"{OUT}/shap_importance.csv").sort_values("rank_gpr")
ls = pd.read_csv(f"{OUT}/ard_length_scales_full.csv").ard_length_scale
v = s.mean_abs_shap_gpr
fam = s.feature.str.match(r"(en_mulliken_B|en_pauling_B|ea_B|ie1_B|dchi_B)")
rows = [
    ("top5_share", v.head(5).sum() / v.sum()),
    ("top10_share", v.head(10).sum() / v.sum()),
    ("Bsite_en_ea_ie_share", v[fam].sum() / v.sum()),
    ("spearman_gpr_rf", spearmanr(s.mean_abs_shap_gpr, s.mean_abs_shap_rf).correlation),
    ("top10_overlap_gpr_rf", len(set(s.nsmallest(10, "rank_gpr").feature) & set(s.nsmallest(10, "rank_rf").feature))),
    ("n_features", len(ls)),
    ("ard_ge_900", int((ls >= 900).sum())),
    ("ard_lt_100", int((ls < 100).sum())),
    ("ard_lt_10", int((ls < 10).sum())),
]
S = pd.DataFrame(rows, columns=["quantity", "value"])
S.to_csv(f"{OUT}/shap_summary.csv", index=False)
print(S.to_string(index=False))
print("top-5 GPR features:", ", ".join(s.feature.head(5)))
