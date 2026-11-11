#!/usr/bin/env python3

from pathlib import Path
import re
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.stats import mannwhitneyu
except Exception:
    mannwhitneyu = None


REPO = Path(__file__).resolve().parents[2]

METHOD_DIRS = {
    "Gate-aware AL": REPO / "OFFICIAL-RUNS" / "01_gate_aware_delta_ml_dos_on",
    "No-gate full-pool AL": REPO / "OFFICIAL-RUNS" / "02_nogate_unified_delta_fullpool",
}

METHODS = ["Gate-aware AL", "No-gate full-pool AL"]

COLORS = {
    "Gate-aware AL": "#1f77b4",          # matplotlib default blue
    "No-gate full-pool AL": "#ff7f0e",   # matplotlib default orange
}

OUTDIR = REPO / "audits" / "nogate_vs_gate_fullpool"
OUTDIR.mkdir(parents=True, exist_ok=True)

CYCLE_MAX = 24
PBE_BANDGAP_ZERO_THR = 0.05


def cycle_from_name(path: Path):
    m = re.search(r"cycle(\d+)", path.name)
    return int(m.group(1)) if m else None


def find_col(df, candidates):
    norm = {re.sub(r"[^a-z0-9]+", "", c.lower()): c for c in df.columns}
    for cand in candidates:
        key = re.sub(r"[^a-z0-9]+", "", cand.lower())
        if key in norm:
            return norm[key]
    return None


def read_metrics():
    rows = []

    for method, root in METHOD_DIRS.items():
        for run_dir in sorted(root.glob("al*")):
            path = run_dir / "metrics" / "metrics.csv"
            if not path.exists():
                print(f"[WARN] missing {path}")
                continue

            df = pd.read_csv(path)
            if "cycle" not in df.columns:
                continue

            df["cycle"] = pd.to_numeric(df["cycle"], errors="coerce")
            df = df[df["cycle"].notna()].copy()
            df = df[df["cycle"].le(CYCLE_MAX)].copy()

            for _, r in df.iterrows():
                rows.append({
                    "method": method,
                    "run": run_dir.name,
                    "cycle": int(r["cycle"]),
                    "end2end_hse_MAE": pd.to_numeric(r.get("end2end_hse_MAE", np.nan), errors="coerce"),
                    "delta_val_MAE_nonmetal": pd.to_numeric(r.get("delta_val_MAE_nonmetal", np.nan), errors="coerce"),
                    "end2end_hse_R2": pd.to_numeric(r.get("end2end_hse_R2", np.nan), errors="coerce"),
                    "delta_val_R2_nonmetal": pd.to_numeric(r.get("delta_val_R2_nonmetal", np.nan), errors="coerce"),
                    "train_n_total": pd.to_numeric(r.get("train_n_total", np.nan), errors="coerce"),
                    "train_n_nonmetal_used": pd.to_numeric(r.get("train_n_nonmetal_used", np.nan), errors="coerce"),
                })

    out = pd.DataFrame(rows)
    out.to_csv(OUTDIR / "cycle_learn_curves_input.csv", index=False)
    return out


def read_selection_counts_directly():
    """
    Recompute cumulative selected PBE band-gap-zero candidates directly
    from selected_cycle*.csv files using numerical cycle order.
    """
    rows = []

    for method, root in METHOD_DIRS.items():
        for run_dir in sorted(root.glob("al*")):
            sel_files = sorted(
                (run_dir / "selections").glob("selected_cycle*.csv"),
                key=lambda p: cycle_from_name(p) if cycle_from_name(p) is not None else 10**9,
            )

            cum_zero = 0
            cum_total = 0

            for sf in sel_files:
                cycle = cycle_from_name(sf)
                if cycle is None or cycle > CYCLE_MAX:
                    continue

                df = pd.read_csv(sf)

                zero_col = find_col(df, ["is_pbe0"])
                pbe_col = find_col(df, ["pbe_gap_effective", "pbe_gap", "pbe_gap_ev"])

                if zero_col is not None:
                    is_zero = (
                        pd.to_numeric(df[zero_col], errors="coerce")
                        .fillna(0)
                        .astype(int)
                        .values == 1
                    )
                elif pbe_col is not None:
                    pbe = pd.to_numeric(df[pbe_col], errors="coerce")
                    is_zero = pbe.le(PBE_BANDGAP_ZERO_THR).fillna(False).values
                else:
                    print(f"[WARN] no PBE gap column in {sf}; treating selected zero-gap count as 0")
                    is_zero = np.zeros(len(df), dtype=bool)

                n_total = int(len(df))
                n_zero = int(np.sum(is_zero))

                cum_total += n_total
                cum_zero += n_zero

                rows.append({
                    "method": method,
                    "run": run_dir.name,
                    "cycle": int(cycle),
                    "n_selected": n_total,
                    "n_selected_pbe_bandgap_zero": n_zero,
                    "cum_selected": cum_total,
                    "cum_selected_pbe_bandgap_zero": cum_zero,
                })

    out = pd.DataFrame(rows)

    # Sanity check: cumulative count must not decrease within any run.
    bad = []
    for (method, run), sub in out.sort_values(["method", "run", "cycle"]).groupby(["method", "run"]):
        diffs = sub["cum_selected_pbe_bandgap_zero"].diff().dropna()
        if (diffs < 0).any():
            bad.append((method, run))

    if bad:
        raise RuntimeError(f"Cumulative PBE band-gap-zero count decreased for: {bad}")

    out.to_csv(OUTDIR / "selection_counts_by_cycle_recomputed_for_plot.csv", index=False)
    return out


def summarize_endpoint(metrics):
    sub = metrics[metrics["cycle"].eq(CYCLE_MAX)].copy()

    rows = []
    for metric, label in [
        ("end2end_hse_MAE", "End-to-end HSE MAE"),
        ("delta_val_MAE_nonmetal", "Non-ZBG Δ MAE"),
    ]:
        for method in METHODS:
            vals = pd.to_numeric(
                sub.loc[sub["method"].eq(method), metric],
                errors="coerce",
            ).dropna()

            rows.append({
                "metric": label,
                "method": method,
                "mean": float(vals.mean()) if len(vals) else np.nan,
                "std": float(vals.std(ddof=1)) if len(vals) > 1 else np.nan,
                "n": int(len(vals)),
            })

    endpoint = pd.DataFrame(rows)
    endpoint.to_csv(OUTDIR / "cycle24_endpoint_summary_for_plot.csv", index=False)
    return endpoint, sub


def mann_whitney_p(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]

    if len(x) < 2 or len(y) < 2:
        return np.nan

    if mannwhitneyu is not None:
        try:
            return float(mannwhitneyu(x, y, alternative="two-sided", method="exact").pvalue)
        except TypeError:
            return float(mannwhitneyu(x, y, alternative="two-sided").pvalue)

    # Fallback for complete separation without scipy.
    if np.max(x) < np.min(y) or np.max(y) < np.min(x):
        return float(2.0 / math.comb(len(x) + len(y), len(x)))

    return np.nan


def p_text(p):
    if not np.isfinite(p):
        return "n.s."
    if p < 0.001:
        return r"$p<0.001$"
    if p < 0.01:
        return r"$p<0.01$"
    return rf"$p={p:.3f}$"


def add_sig_bracket(ax, x1, x2, y, h, text):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="black", linewidth=0.8)
    ax.text((x1 + x2) / 2.0, y + h, text, ha="center", va="bottom", fontsize=8)


def plot_mean_sd(ax, df, value_col, ylabel, title):
    handles = []
    labels = []

    for method in METHODS:
        sub = df[df["method"].eq(method)].copy()
        if sub.empty:
            continue

        g = (
            sub
            .groupby("cycle")[value_col]
            .agg(["mean", "std"])
            .reset_index()
            .sort_values("cycle")
        )

        x = g["cycle"].to_numpy(dtype=float)
        y = g["mean"].to_numpy(dtype=float)
        s = g["std"].to_numpy(dtype=float)

        line, = ax.plot(
            x,
            y,
            marker="o",
            markersize=3,
            linewidth=1.8,
            label=method,
            color=COLORS[method],
        )

        handles.append(line)
        labels.append(method)

        if np.isfinite(s).any():
            ax.fill_between(
                x,
                y - s,
                y + s,
                alpha=0.18,
                color=COLORS[method],
                linewidth=0,
            )

    ax.set_title(title)
    ax.set_xlabel("AL cycle")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, CYCLE_MAX)
    ax.grid(True, alpha=0.25)

    return handles, labels


def main():
    metrics = read_metrics()
    selections = read_selection_counts_directly()
    endpoint, endpoint_runs = summarize_endpoint(metrics)

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0))
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    # A. End-to-end validation MAE learning curve
    handles, labels = plot_mean_sd(
        ax_a,
        metrics,
        "end2end_hse_MAE",
        "Validation MAE (eV)",
        "A. End-to-end HSE06 validation error",
    )

    # B. Non-ZBG Delta MAE learning curve
    plot_mean_sd(
        ax_b,
        metrics,
        "delta_val_MAE_nonmetal",
        "Validation MAE (eV)",
        "B. Non-ZBG Δ-learning validation error",
    )

    # C. Cycle-24 endpoint bars
    metric_labels = ["End-to-end HSE MAE", "Non-ZBG Δ MAE"]

    x = np.arange(len(metric_labels), dtype=float)
    width = 0.34

    bracket_records = []

    for i, method in enumerate(METHODS):
        means = []
        stds = []

        for metric in metric_labels:
            row = endpoint[
                endpoint["metric"].eq(metric)
                & endpoint["method"].eq(method)
            ]
            means.append(float(row["mean"].iloc[0]))
            stds.append(float(row["std"].iloc[0]))

        offset = (i - 0.5) * width
        ax_c.bar(
            x + offset,
            means,
            width,
            yerr=stds,
            capsize=4,
            label=method,
            alpha=0.85,
            color=COLORS[method],
        )

        for j, metric in enumerate(metric_labels):
            col = (
                "end2end_hse_MAE"
                if metric == "End-to-end HSE MAE"
                else "delta_val_MAE_nonmetal"
            )
            vals = endpoint_runs.loc[
                endpoint_runs["method"].eq(method),
                col,
            ].dropna().to_numpy(dtype=float)

            if len(vals):
                jitter = np.linspace(-0.045, 0.045, len(vals))
                ax_c.scatter(
                    np.full(len(vals), x[j] + offset) + jitter,
                    vals,
                    s=20,
                    zorder=3,
                    color=COLORS[method],
                    edgecolor="black",
                    linewidth=0.3,
                )

    # Significance brackets for Panel C
    y0, y1 = ax_c.get_ylim()
    yr = y1 - y0
    bracket_gap = 0.065 * yr
    bracket_h = 0.025 * yr

    for j, metric in enumerate(metric_labels):
        col = (
            "end2end_hse_MAE"
            if metric == "End-to-end HSE MAE"
            else "delta_val_MAE_nonmetal"
        )

        gate_vals = endpoint_runs.loc[
            endpoint_runs["method"].eq("Gate-aware AL"),
            col,
        ].dropna().to_numpy(dtype=float)

        nogate_vals = endpoint_runs.loc[
            endpoint_runs["method"].eq("No-gate full-pool AL"),
            col,
        ].dropna().to_numpy(dtype=float)

        p = mann_whitney_p(gate_vals, nogate_vals)

        ymax = np.nanmax(np.concatenate([gate_vals, nogate_vals]))
        y_bracket = ymax + bracket_gap

        x1 = x[j] - width / 2.0
        x2 = x[j] + width / 2.0

        add_sig_bracket(ax_c, x1, x2, y_bracket, bracket_h, p_text(p))
        bracket_records.append({
            "metric": metric,
            "mann_whitney_two_sided_p": p,
        })

    ax_c.set_ylim(top=max(ax_c.get_ylim()[1], max([r["mann_whitney_two_sided_p"] for r in bracket_records if np.isfinite(r["mann_whitney_two_sided_p"])] or [0]) + ax_c.get_ylim()[1]))

    # Reset ylim more sensibly after bracket annotation
    current_bottom, current_top = ax_c.get_ylim()
    max_data = max(
        float(endpoint_runs["end2end_hse_MAE"].max()),
        float(endpoint_runs["delta_val_MAE_nonmetal"].max()),
    )
    ax_c.set_ylim(current_bottom, max_data + 0.22 * (max_data - current_bottom))

    ax_c.set_title("C. Cycle-24 budget-matched endpoint")
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(metric_labels, rotation=10, ha="right")
    ax_c.set_ylabel("Validation MAE (eV)")
    ax_c.grid(True, axis="y", alpha=0.25)

    pd.DataFrame(bracket_records).to_csv(
        OUTDIR / "cycle24_endpoint_mannwhitney_pvalues.csv",
        index=False,
    )

    # D. Cumulative PBE band-gap-zero candidate selections
    for method in METHODS:
        sub = selections[selections["method"].eq(method)].copy()
        if sub.empty:
            continue

        g = (
            sub
            .groupby("cycle")["cum_selected_pbe_bandgap_zero"]
            .agg(["mean", "std"])
            .reset_index()
            .sort_values("cycle")
        )

        xcy = g["cycle"].to_numpy(dtype=float)
        y = g["mean"].to_numpy(dtype=float)
        s = g["std"].to_numpy(dtype=float)

        ax_d.plot(
            xcy,
            y,
            marker="o",
            markersize=3,
            linewidth=1.8,
            label=method,
            color=COLORS[method],
        )

        if np.isfinite(s).any():
            ax_d.fill_between(
                xcy,
                y - s,
                y + s,
                alpha=0.18,
                color=COLORS[method],
                linewidth=0,
            )

    ax_d.set_title(r"D. Cumulative $E_g^{\mathrm{PBE}}=0$ candidates selected")
    ax_d.set_xlabel("AL cycle")
    ax_d.set_ylabel(r"Cumulative selected $E_g^{\mathrm{PBE}}=0$ labels")
    ax_d.set_xlim(0, CYCLE_MAX)
    ax_d.grid(True, alpha=0.25)

    # One shared legend, not repeated in every panel.
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
    )

    fig.tight_layout(rect=[0, 0, 1, 0.955])

    png = OUTDIR / "fig_nogate_vs_gate_fullpool_ablation.png"
    pdf = OUTDIR / "fig_nogate_vs_gate_fullpool_ablation.pdf"
    svg = OUTDIR / "fig_nogate_vs_gate_fullpool_ablation.svg"

    fig.savefig(png, dpi=450, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")

    print("Wrote:")
    print(png)
    print(pdf)
    print(svg)

    print("\nEndpoint summary:")
    print(endpoint.to_string(index=False))

    print("\nMann-Whitney p-values:")
    print(pd.DataFrame(bracket_records).to_string(index=False))

    print("\nFinal cumulative selected Eg^PBE=0 labels:")
    last = (
        selections
        .sort_values(["method", "run", "cycle"])
        .groupby(["method", "run"])
        .tail(1)
    )
    print(last[["method", "run", "cycle", "cum_selected", "cum_selected_pbe_bandgap_zero"]].to_string(index=False))


if __name__ == "__main__":
    main()
