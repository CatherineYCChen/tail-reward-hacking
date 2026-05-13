from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


def cvar(values: np.ndarray, alpha: float) -> float:
    if len(values) == 0:
        return float("nan")
    threshold = np.quantile(values, alpha)
    tail = values[values >= threshold]
    return float(np.mean(tail))


def apply_candidate_permutation(candidates: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    shuffled_groups = []
    for _, frame in candidates.groupby("prompt_id", sort=True):
        order = rng.permutation(len(frame))
        shuffled = frame.iloc[order].copy().reset_index(drop=True)
        shuffled["analysis_candidate_id"] = np.arange(len(shuffled))
        shuffled_groups.append(shuffled)
    return pd.concat(shuffled_groups, ignore_index=True)


def summarize_seed(
    candidates: pd.DataFrame,
    n_values: Iterable[int],
    alpha: float,
    top_proxy_quantile: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    randomized = apply_candidate_permutation(candidates, seed=seed)
    failure_threshold = float(randomized["gap"].quantile(0.95))

    bon_rows = []
    sorted_candidates = randomized.sort_values(["prompt_id", "analysis_candidate_id"]).copy()
    for n in n_values:
        eligible = sorted_candidates[sorted_candidates["analysis_candidate_id"] < n]
        picked = (
            eligible.sort_values(["prompt_id", "proxy_reward"], ascending=[True, False])
            .groupby("prompt_id", as_index=False)
            .head(1)
            .copy()
        )
        bon_rows.append(_summarize_selection(
            picked=picked,
            strategy="bon",
            n=n,
            alpha=alpha,
            seed=seed,
            failure_threshold=failure_threshold,
            top_proxy_quantile=top_proxy_quantile,
        ))

    rng = np.random.default_rng(seed + 1000)
    random_picked = (
        randomized.groupby("prompt_id", group_keys=False)
        .apply(lambda frame: frame.sample(n=1, random_state=int(rng.integers(0, 2**31 - 1))))
        .reset_index(drop=True)
        .copy()
    )
    random_summary = pd.DataFrame([
        _summarize_selection(
            picked=random_picked,
            strategy="random",
            n=1,
            alpha=alpha,
            seed=seed,
            failure_threshold=failure_threshold,
            top_proxy_quantile=top_proxy_quantile,
        )
    ])
    return pd.DataFrame(bon_rows), random_summary, failure_threshold


def aggregate_seed_summaries(summary_by_seed: pd.DataFrame) -> pd.DataFrame:
    numeric_metrics = [
        "avg_proxy_reward",
        "avg_true_reward",
        "mean_gap",
        "cvar_gap_95",
        "failure_threshold",
        "top_proxy_threshold",
        "conditional_failure_rate",
        "selected_count",
    ]
    grouped = (
        summary_by_seed.groupby(["strategy", "n"], as_index=False)
        .agg({metric: ["mean", "std"] for metric in numeric_metrics})
    )
    grouped.columns = [
        "_".join([part for part in column if part]).rstrip("_")
        for column in grouped.columns.to_flat_index()
    ]
    grouped = grouped.rename(
        columns={
            "strategy_": "strategy",
            "n_": "n",
        }
    )
    grouped["num_seeds"] = summary_by_seed.groupby(["strategy", "n"]).size().values
    return grouped.sort_values(["strategy", "n"]).reset_index(drop=True)


def summarize_gap_by_proxy_percentile(
    candidates: pd.DataFrame,
    num_bins: int,
) -> pd.DataFrame:
    work = candidates.copy()
    work["proxy_percentile"] = work["proxy_reward"].rank(method="average", pct=True)
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    work["proxy_bin"] = pd.cut(work["proxy_percentile"], bins=bins, include_lowest=True)
    summary = (
        work.groupby("proxy_bin", observed=False)
        .agg(
            mean_gap=("gap", "mean"),
            median_gap=("gap", "median"),
            count=("gap", "size"),
            mean_proxy_reward=("proxy_reward", "mean"),
        )
        .reset_index()
    )
    summary["bin_midpoint"] = [
        interval.mid if pd.notna(interval) else np.nan for interval in summary["proxy_bin"]
    ]
    return summary


def _summarize_selection(
    picked: pd.DataFrame,
    strategy: str,
    n: int,
    alpha: float,
    seed: int,
    failure_threshold: float,
    top_proxy_quantile: float,
) -> dict:
    top_proxy_threshold = float(picked["proxy_reward"].quantile(top_proxy_quantile))
    conditional_mask = picked["proxy_reward"] >= top_proxy_threshold
    conditional_failure_rate = float(
        (picked.loc[conditional_mask, "gap"] > failure_threshold).mean()
    )

    return {
        "strategy": strategy,
        "seed": int(seed),
        "n": int(n),
        "avg_proxy_reward": float(picked["proxy_reward"].mean()),
        "avg_true_reward": float(picked["true_reward"].mean()),
        "mean_gap": float(picked["gap"].mean()),
        "cvar_gap_95": cvar(picked["gap"].to_numpy(), alpha),
        "failure_threshold": float(failure_threshold),
        "top_proxy_threshold": top_proxy_threshold,
        "conditional_failure_rate": conditional_failure_rate,
        "selected_count": int(len(picked)),
    }
