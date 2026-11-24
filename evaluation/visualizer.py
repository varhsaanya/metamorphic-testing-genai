
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import RESULTS_DIR
from metamorphic.executor import MRResult

plt.rcParams.update({
    "figure.dpi":     150,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "font.family":    "DejaVu Sans",
})
PALETTE = "Blues_d"


def _save(fig: plt.Figure, name: str) -> Path:
    path = RESULTS_DIR / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path

def plot_violation_rates(results: list[MRResult], save: bool = True) -> Path | None:
    names  = [r.mr_name.replace("_", "\n") for r in results]
    rates  = [r.violation_rate * 100 for r in results]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, rates, color=sns.color_palette(PALETTE, len(names)))
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=8)
    ax.set_ylabel("Violation Rate (%)")
    ax.set_title("Metamorphic Relation Violation Rates")
    ax.set_ylim(0, max(rates) * 1.2 + 1)
    ax.tick_params(axis="x", labelsize=7)
    fig.tight_layout()
    return _save(fig, "violation_rates.png") if save else None


def plot_fault_detection(comparison_df: pd.DataFrame, save: bool = True) -> Path | None:
    fig, ax = plt.subplots(figsize=(10, 5))
    methods = comparison_df["method"].str.replace("_", "\n")
    fdrs    = comparison_df["fdr_pct"]

    colors = ["#e74c3c" if "random" in m.lower() else "#2980b9"
              for m in comparison_df["method"]]
    bars = ax.bar(methods, fdrs, color=colors)
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=8)
    ax.set_ylabel("Fault Detection Rate (%)")
    ax.set_title("Fault Detection Rate: MRs vs Random Testing")
    ax.set_ylim(0, max(fdrs) * 1.25 + 1)
    ax.tick_params(axis="x", labelsize=7)

    from matplotlib.patches import Patch
    legend = [
        Patch(facecolor="#2980b9", label="Metamorphic Relation"),
        Patch(facecolor="#e74c3c", label="Random Testing"),
    ]
    ax.legend(handles=legend, loc="upper right")
    fig.tight_layout()
    return _save(fig, "fault_detection_rate.png") if save else None


def plot_prob_shift(results: list[MRResult], save: bool = True) -> Path | None:
    eligible = [r for r in results if r.prob_orig is not None]
    if not eligible:
        return None

    n = len(eligible)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, r in zip(axes, eligible):
        delta = r.prob_trans - r.prob_orig
        ax.hist(delta, bins=40, color="#3498db", edgecolor="white", alpha=0.85)
        ax.axvline(0, color="red", linestyle="--", linewidth=1.2)
        ax.set_title(r.mr_name.replace("_", "\n"), fontsize=8)
        ax.set_xlabel("Δ Prob (trans − orig)", fontsize=8)
        ax.set_ylabel("Count", fontsize=8)

    fig.suptitle("Probability Shift Distributions per MR", fontsize=11)
    fig.tight_layout()
    return _save(fig, "prob_shift_distributions.png") if save else None


def plot_exec_times(results: list[MRResult], save: bool = True) -> Path | None:
    names = [r.mr_name.replace("_", "\n") for r in results]
    times = [r.exec_time_s for r in results]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(names[::-1], times[::-1], color="#27ae60")
    ax.set_xlabel("Execution Time (s)")
    ax.set_title("MR Execution Times")
    fig.tight_layout()
    return _save(fig, "exec_times.png") if save else None


def plot_model_comparison_heatmap(
    data: dict[str, list[MRResult]], save: bool = True
) -> Path | None:
    rows = {}
    for model_name, results in data.items():
        rows[model_name] = {r.mr_name: r.violation_rate for r in results}

    df = pd.DataFrame(rows).T
    fig, ax = plt.subplots(figsize=(max(8, len(df.columns)), max(4, len(df))))
    sns.heatmap(
        df * 100, annot=True, fmt=".1f", cmap="YlOrRd",
        linewidths=0.4, ax=ax, cbar_kws={"label": "Violation Rate (%)"},
    )
    ax.set_title("MR Violation Rate Heatmap — Model Comparison")
    ax.set_ylabel("Model")
    ax.set_xlabel("Metamorphic Relation")
    fig.tight_layout()
    return _save(fig, "model_comparison_heatmap.png") if save else None