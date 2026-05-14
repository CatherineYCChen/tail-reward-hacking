from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


def cvar(values: np.ndarray, alpha: float) -> float:
    if len(values) == 0:
        return float("nan")
    threshold = np.quantile(values, alpha)
    tail = values[values >= threshold]
    return float(np.mean(tail))


def add_normalized_columns(candidates: pd.DataFrame) -> pd.DataFrame:
    work = candidates.copy()
    work["proxy_z"] = _zscore_series(work["proxy_reward"])
    work["true_z"] = _zscore_series(work["true_reward"])
    work["gap_z"] = work["proxy_z"] - work["true_z"]
    return work


def apply_candidate_permutation(candidates: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    shuffled_groups = []
    for _, frame in candidates.groupby("prompt_id", sort=True):
        order = rng.permutation(len(frame))
        shuffled = frame.iloc[order].copy().reset_index(drop=True)
        shuffled["analysis_candidate_id"] = np.arange(len(shuffled))
        shuffled_groups.append(shuffled)
    return pd.concat(shuffled_groups, ignore_index=True)


def select_best_of_n(
    candidates: pd.DataFrame,
    n: int,
) -> pd.DataFrame:
    eligible = candidates[candidates["analysis_candidate_id"] < n]
    return (
        eligible.sort_values(["prompt_id", "proxy_reward"], ascending=[True, False])
        .groupby("prompt_id", as_index=False)
        .head(1)
        .copy()
    )


def select_random_winners(
    candidates: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    random_rows = []
    for _, frame in candidates.groupby("prompt_id", sort=True):
        sampled = frame.sample(n=1, random_state=int(rng.integers(0, 2**31 - 1)))
        random_rows.append(sampled)
    return pd.concat(random_rows, ignore_index=True).copy()


def summarize_seed(
    candidates: pd.DataFrame,
    n_values: Iterable[int],
    alpha: float,
    top_proxy_quantile: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, float, Dict[int, pd.DataFrame]]:
    randomized = apply_candidate_permutation(candidates, seed=seed)
    failure_threshold = float(randomized["gap"].quantile(0.95))
    median_global_gap = float(randomized["gap"].median())

    bon_rows = []
    bon_selected_by_n = {}
    for n in n_values:
        picked = select_best_of_n(randomized, n=n)
        bon_selected_by_n[int(n)] = picked
        bon_rows.append(
            _summarize_selection(
                picked=picked,
                strategy="bon",
                n=n,
                alpha=alpha,
                seed=seed,
                failure_threshold=failure_threshold,
                median_global_gap=median_global_gap,
                top_proxy_quantile=top_proxy_quantile,
            )
        )

    random_picked = select_random_winners(randomized, seed=seed + 1000)
    random_summary = pd.DataFrame(
        [
            _summarize_selection(
                picked=random_picked,
                strategy="random",
                n=1,
                alpha=alpha,
                seed=seed,
                failure_threshold=failure_threshold,
                median_global_gap=median_global_gap,
                top_proxy_quantile=top_proxy_quantile,
            )
        ]
    )
    return pd.DataFrame(bon_rows), random_summary, failure_threshold, bon_selected_by_n


def aggregate_seed_summaries(summary_by_seed: pd.DataFrame) -> pd.DataFrame:
    numeric_metrics = [
        "avg_proxy_reward",
        "avg_true_reward",
        "mean_gap",
        "cvar_gap_95",
        "selected_gap_quantile_90",
        "selected_gap_quantile_95",
        "selected_gap_max",
        "top_proxy_mean_gap",
        "top_proxy_gap_rate",
        "corr_proxy_true",
        "corr_proxy_gap",
        "mean_gap_z",
        "cvar_gap_z_95",
        "selected_gap_z_quantile_95",
        "top_proxy_mean_gap_z",
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
            mean_gap_z=("gap_z", "mean"),
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
    median_global_gap: float,
    top_proxy_quantile: float,
) -> dict:
    top_proxy_threshold = float(picked["proxy_reward"].quantile(top_proxy_quantile))
    conditional_mask = picked["proxy_reward"] >= top_proxy_threshold
    conditional_failure_rate = float(
        (picked.loc[conditional_mask, "gap"] > failure_threshold).mean()
    )

    top_proxy_mask = picked["proxy_reward"] >= float(picked["proxy_reward"].quantile(0.80))
    top_proxy_slice = picked.loc[top_proxy_mask].copy()

    return {
        "strategy": strategy,
        "seed": int(seed),
        "n": int(n),
        "avg_proxy_reward": float(picked["proxy_reward"].mean()),
        "avg_true_reward": float(picked["true_reward"].mean()),
        "mean_gap": float(picked["gap"].mean()),
        "cvar_gap_95": cvar(picked["gap"].to_numpy(), alpha),
        "selected_gap_quantile_90": float(picked["gap"].quantile(0.90)),
        "selected_gap_quantile_95": float(picked["gap"].quantile(0.95)),
        "selected_gap_max": float(picked["gap"].max()),
        "top_proxy_mean_gap": float(top_proxy_slice["gap"].mean()),
        "top_proxy_gap_rate": float((top_proxy_slice["gap"] > median_global_gap).mean()),
        "corr_proxy_true": _safe_corr(picked["proxy_reward"], picked["true_reward"]),
        "corr_proxy_gap": _safe_corr(picked["proxy_reward"], picked["gap"]),
        "mean_gap_z": float(picked["gap_z"].mean()),
        "cvar_gap_z_95": cvar(picked["gap_z"].to_numpy(), alpha),
        "selected_gap_z_quantile_95": float(picked["gap_z"].quantile(0.95)),
        "top_proxy_mean_gap_z": float(top_proxy_slice["gap_z"].mean()),
        "failure_threshold": float(failure_threshold),
        "top_proxy_threshold": top_proxy_threshold,
        "conditional_failure_rate": conditional_failure_rate,
        "selected_count": int(len(picked)),
    }


def _safe_corr(left: pd.Series, right: pd.Series) -> float:
    left_std = float(left.std(ddof=0))
    right_std = float(right.std(ddof=0))
    if left_std == 0.0 or right_std == 0.0:
        return float("nan")
    return float(left.corr(right))


def _zscore_series(series: pd.Series) -> pd.Series:
    mean = float(series.mean())
    std = float(series.std(ddof=0))
    if std == 0.0:
        return pd.Series(np.zeros(len(series)), index=series.index, dtype=float)
    return (series - mean) / std
