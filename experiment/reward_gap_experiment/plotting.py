from __future__ import annotations

from pathlib import Path

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
    ax.set_xlabel("n")
    ax.set_ylabel("Proxy - true reward")
    ax.set_title("Reward-gap tail growth")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "plot_gap_tail.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(percentile_summary["bin_midpoint"] * 100.0, percentile_summary["mean_gap"], marker="o")
    ax.set_xlabel("Proxy reward percentile")
    ax.set_ylabel("Average gap")
    ax.set_title("Gap by proxy percentile")
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
