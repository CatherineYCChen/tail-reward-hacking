from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd


def _require_plotting():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to write PNG plots.") from exc
    return plt


def make_all_plots(
    bon_summary: pd.DataFrame,
    random_summary: pd.DataFrame,
    percentile_summary: pd.DataFrame,
    all_candidates: pd.DataFrame,
    selected_by_n: Dict[int, pd.DataFrame],
    output_dir: Path,
) -> None:
    plt = _require_plotting()

    bon_summary = bon_summary.sort_values("n")
    random_summary = random_summary.sort_values("n")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(bon_summary["n"], bon_summary["avg_proxy_reward_mean"], marker="o", label="Proxy reward")
    ax.plot(bon_summary["n"], bon_summary["avg_true_reward_mean"], marker="o", label="True reward")
    ax.set_xlabel("n")
    ax.set_ylabel("Average reward")
    ax.set_title("Best-of-n rewards")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "plot_bon_rewards.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(bon_summary["n"], bon_summary["mean_gap_mean"], marker="o", label="Mean gap")
    ax.plot(bon_summary["n"], bon_summary["cvar_gap_95_mean"], marker="o", label="CVaR_95 gap")
    ax.plot(bon_summary["n"], bon_summary["mean_gap_z_mean"], marker="o", label="Mean gap z")
    ax.plot(bon_summary["n"], bon_summary["cvar_gap_z_95_mean"], marker="o", label="CVaR_95 gap z")
    ax.set_xlabel("n")
    ax.set_ylabel("Gap metric")
    ax.set_title("Reward-gap tail growth")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "plot_gap_tail.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(percentile_summary["bin_midpoint"] * 100.0, percentile_summary["mean_gap"], marker="o", label="Mean gap")
    ax.plot(percentile_summary["bin_midpoint"] * 100.0, percentile_summary["mean_gap_z"], marker="o", label="Mean gap z")
    ax.set_xlabel("Proxy reward percentile")
    ax.set_ylabel("Average gap")
    ax.set_title("Gap by proxy percentile")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "plot_gap_by_proxy_percentile.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    axes[0].axhline(
        random_summary["mean_gap_mean"].iloc[0],
        color="#999999",
        linestyle="--",
        label="Random baseline",
    )
    axes[0].plot(bon_summary["n"], bon_summary["mean_gap_mean"], marker="o", label="BoN")
    axes[0].set_xlabel("n")
    axes[0].set_ylabel("Mean gap")
    axes[0].set_title("Random vs BoN mean gap")
    axes[0].legend()

    axes[1].axhline(
        random_summary["cvar_gap_95_mean"].iloc[0],
        color="#999999",
        linestyle="--",
        label="Random baseline",
    )
    axes[1].plot(bon_summary["n"], bon_summary["cvar_gap_95_mean"], marker="o", label="BoN")
    axes[1].set_xlabel("n")
    axes[1].set_ylabel("CVaR_95 gap")
    axes[1].set_title("Random vs BoN tail gap")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "plot_random_vs_bon_tail.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(all_candidates["proxy_reward"], all_candidates["true_reward"], alpha=0.35, s=16)
    ax.set_xlabel("Proxy reward")
    ax.set_ylabel("True reward")
    ax.set_title("Proxy vs true reward")
    fig.tight_layout()
    fig.savefig(output_dir / "scatter_proxy_vs_true.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(all_candidates["proxy_reward"], all_candidates["gap"], alpha=0.35, s=16)
    ax.set_xlabel("Proxy reward")
    ax.set_ylabel("Gap")
    ax.set_title("Proxy reward vs gap")
    fig.tight_layout()
    fig.savefig(output_dir / "scatter_proxy_vs_gap.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(all_candidates["gap"], bins=30, alpha=0.8, color="#4c78a8")
    ax.set_xlabel("Gap")
    ax.set_ylabel("Count")
    ax.set_title("Gap histogram across all candidates")
    fig.tight_layout()
    fig.savefig(output_dir / "histogram_gap_all_candidates.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for n, frame in sorted(selected_by_n.items()):
        ax.hist(frame["gap"], bins=15, alpha=0.45, label="n=%d" % n)
    ax.set_xlabel("Gap")
    ax.set_ylabel("Count")
    ax.set_title("Gap histogram for selected winners by n")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "histogram_gap_selected_by_n.png", dpi=180)
    plt.close(fig)
