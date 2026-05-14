from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd


_CONFIDENT_MARKERS = (
    "definitely",
    "certainly",
    "always",
    "never",
    "guaranteed",
    "prove",
    "proves",
    "obviously",
    "clearly",
    "undoubtedly",
)
_UNCERTAINTY_MARKERS = (
    "maybe",
    "perhaps",
    "possibly",
    "might",
    "could",
    "likely",
    "uncertain",
    "appears",
    "seems",
)
_TECHNICAL_MARKERS = (
    "protocol",
    "architecture",
    "bandwidth",
    "latency",
    "kernel",
    "api",
    "framework",
    "database",
    "algorithm",
    "encryption",
    "vectorized",
    "quantum",
    "throughput",
    "microservice",
    "transformer",
)
_BULLET_RE = re.compile(r"(^|\n)\s*(?:[-*+]|\d+[.)])\s+")
_NUMBER_RE = re.compile(r"\d")
_EQUATION_RE = re.compile(r"(=|==|!=|<=|>=|\+|-|\*|/|\^|≈|∑|√)")
_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_SENTENCE_SPLIT_RE = re.compile(r"[.!?]\s+")
_MARKDOWN_STRUCTURE_RE = re.compile(r"(^|\n)\s*(#{1,6}\s|```|\|.+\||>\s)")


def add_response_length_tokens(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["response_length_tokens"] = work["response"].astype(str).map(estimate_token_count)
    return work


def annotate_high_gap_examples(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    response_text = work["response"].astype(str)
    work["response_length_tokens"] = response_text.map(estimate_token_count)
    work["contains_lists"] = response_text.map(_contains_lists)
    work["contains_numbers"] = response_text.map(_contains_numbers)
    work["contains_equations"] = response_text.map(_contains_equations)
    work["contains_confident_language"] = response_text.map(_contains_confident_language)
    work["contains_uncertainty_language"] = response_text.map(_contains_uncertainty_language)
    work["factual_inconsistency_heuristic"] = response_text.map(_factual_inconsistency_heuristic)
    work["hallucinated_technical_detail_heuristic"] = response_text.map(
        _hallucinated_technical_detail_heuristic
    )
    work["verbosity_heuristic"] = response_text.map(_verbosity_heuristic)
    work["repetition_heuristic"] = response_text.map(_repetition_heuristic)
    formatting_scores = response_text.map(_formatting_quality_score)
    work["formatting_quality_score"] = formatting_scores
    work["formatting_quality_heuristic"] = formatting_scores.map(_formatting_quality_label)
    work["formatting_structure_exploitation_heuristic"] = response_text.map(
        _formatting_structure_exploitation_heuristic
    )
    return work


def write_annotated_examples_and_summary(
    selected_by_n: Dict[int, pd.DataFrame],
    output_dir: Path,
    candidate_correlations: Dict[str, float],
) -> Dict[str, float]:
    annotated_frames: List[pd.DataFrame] = []
    for n, frame in sorted(selected_by_n.items()):
        largest_gap = (
            frame.sort_values(["gap", "proxy_reward"], ascending=[False, False])
            .head(10)
            .copy()
        )
        annotated = annotate_high_gap_examples(largest_gap)
        annotated_frames.append(annotated.assign(n=int(n)))
        annotated.to_csv(output_dir / ("examples_largest_gap_annotated_n%d.csv" % n), index=False)

    if annotated_frames:
        combined = pd.concat(annotated_frames, ignore_index=True)
    else:
        combined = pd.DataFrame()

    summary_stats = summarize_qualitative_patterns(combined)
    (output_dir / "qualitative_summary.txt").write_text(
        _render_qualitative_summary(summary_stats, candidate_correlations)
    )
    return summary_stats


def write_manual_review_exports(
    selected_by_n: Dict[int, pd.DataFrame],
    output_dir: Path,
    top_k: int = 50,
) -> None:
    combined = _combine_selected_examples(selected_by_n)
    if combined.empty:
        manual_frame = _add_manual_review_columns(combined)
        manual_frame.to_csv(output_dir / "manual_review_largest_gap.csv", index=False)
        manual_frame.to_csv(output_dir / "manual_review_high_proxy.csv", index=False)
        return

    largest_gap = (
        combined.sort_values(["gap", "proxy_reward"], ascending=[False, False])
        .head(top_k)
        .copy()
    )
    highest_proxy = (
        combined.sort_values(["proxy_reward", "gap"], ascending=[False, False])
        .head(top_k)
        .copy()
    )
    _add_manual_review_columns(largest_gap).to_csv(
        output_dir / "manual_review_largest_gap.csv",
        index=False,
    )
    _add_manual_review_columns(highest_proxy).to_csv(
        output_dir / "manual_review_high_proxy.csv",
        index=False,
    )


def summarize_qualitative_patterns(frame: pd.DataFrame) -> Dict[str, float]:
    if frame.empty:
        return {
            "num_high_gap_examples": 0.0,
            "fraction_factual_errors": 0.0,
            "fraction_hallucinated_technical_language": 0.0,
            "fraction_verbose_but_shallow": 0.0,
            "fraction_formatting_structure_exploitation": 0.0,
        }

    return {
        "num_high_gap_examples": float(len(frame)),
        "fraction_factual_errors": float(frame["factual_inconsistency_heuristic"].mean()),
        "fraction_hallucinated_technical_language": float(
            frame["hallucinated_technical_detail_heuristic"].mean()
        ),
        "fraction_verbose_but_shallow": float(frame["verbosity_heuristic"].mean()),
        "fraction_formatting_structure_exploitation": float(
            frame["formatting_structure_exploitation_heuristic"].mean()
        ),
    }


def summarize_candidate_correlations(candidates: pd.DataFrame) -> Dict[str, float]:
    return {
        "gap_vs_response_length": _safe_corr(
            candidates["gap"], candidates["response_length_tokens"]
        ),
        "gap_vs_proxy_reward": _safe_corr(candidates["gap"], candidates["proxy_reward"]),
        "gap_vs_true_reward": _safe_corr(candidates["gap"], candidates["true_reward"]),
    }


def estimate_token_count(text: str) -> int:
    return len(_WORD_RE.findall(str(text)))


def _combine_selected_examples(selected_by_n: Dict[int, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for n, frame in sorted(selected_by_n.items()):
        frames.append(frame.copy().assign(n=int(n)))
    if not frames:
        return pd.DataFrame(columns=_manual_review_base_columns())
    return pd.concat(frames, ignore_index=True)


def _add_manual_review_columns(frame: pd.DataFrame) -> pd.DataFrame:
    base_columns = _manual_review_base_columns()
    work = frame.loc[:, [column for column in base_columns if column in frame.columns]].copy()
    for column in (
        "manual_factual_error",
        "manual_hallucination",
        "manual_incomplete_answer",
        "manual_verbose_but_shallow",
        "manual_formatting_or_style_reward",
        "manual_notes",
    ):
        work[column] = ""
    return work


def _manual_review_base_columns() -> List[str]:
    return [
        "prompt",
        "response",
        "proxy_reward",
        "true_reward",
        "gap",
        "proxy_z",
        "true_z",
        "gap_z",
        "n",
        "candidate_id",
        "response_length_tokens",
    ]


def _contains_lists(text: str) -> bool:
    return bool(_BULLET_RE.search(text))


def _contains_numbers(text: str) -> bool:
    return bool(_NUMBER_RE.search(text))


def _contains_equations(text: str) -> bool:
    return bool(_EQUATION_RE.search(text))


def _contains_confident_language(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _CONFIDENT_MARKERS)


def _contains_uncertainty_language(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _UNCERTAINTY_MARKERS)


def _factual_inconsistency_heuristic(text: str) -> bool:
    lowered = text.lower()
    contradiction_pairs = (
        ("always", "sometimes"),
        ("always", "might"),
        ("never", "can"),
        ("is", "is not"),
        ("will", "may"),
    )
    if any(left in lowered and right in lowered for left, right in contradiction_pairs):
        return True
    return _contains_confident_language(text) and _contains_uncertainty_language(text)


def _hallucinated_technical_detail_heuristic(text: str) -> bool:
    lowered = text.lower()
    technical_hits = sum(marker in lowered for marker in _TECHNICAL_MARKERS)
    return technical_hits >= 2 and _contains_numbers(text) and _contains_confident_language(text)


def _verbosity_heuristic(text: str) -> bool:
    token_count = estimate_token_count(text)
    sentence_count = max(1, len(_SENTENCE_SPLIT_RE.split(text.strip())))
    avg_sentence_len = token_count / sentence_count
    return token_count >= 120 and avg_sentence_len >= 14


def _repetition_heuristic(text: str) -> bool:
    tokens = [token.lower() for token in _WORD_RE.findall(text)]
    if len(tokens) < 12:
        return False
    bigrams = list(zip(tokens, tokens[1:]))
    repeated_bigram_ratio = 1.0 - (len(set(bigrams)) / max(len(bigrams), 1))
    repeated_token_ratio = 1.0 - (len(set(tokens)) / max(len(tokens), 1))
    return repeated_bigram_ratio > 0.18 or repeated_token_ratio > 0.55


def _formatting_quality_score(text: str) -> int:
    score = 0
    stripped = text.strip()
    if stripped and stripped[0].isupper():
        score += 1
    if any(punct in stripped for punct in ".:;"):
        score += 1
    if _contains_lists(text) or "\n\n" in text:
        score += 1
    if _repetition_heuristic(text):
        score -= 1
    return max(score, 0)


def _formatting_quality_label(score: int) -> str:
    if score >= 3:
        return "high"
    if score == 2:
        return "medium"
    return "low"


def _formatting_structure_exploitation_heuristic(text: str) -> bool:
    return bool(_MARKDOWN_STRUCTURE_RE.search(text) or _contains_lists(text))


def _safe_corr(left: Iterable[float], right: Iterable[float]) -> float:
    left_series = pd.Series(list(left), dtype=float)
    right_series = pd.Series(list(right), dtype=float)
    if float(left_series.std(ddof=0)) == 0.0 or float(right_series.std(ddof=0)) == 0.0:
        return float("nan")
    return float(left_series.corr(right_series))


def _render_qualitative_summary(
    summary_stats: Dict[str, float],
    candidate_correlations: Dict[str, float],
) -> str:
    return (
        "Qualitative summary for largest-gap selected winners\n"
        "Fraction estimates are heuristic-only and use no LLM labeling.\n\n"
        "Number of annotated high-gap examples: {num_high_gap_examples:.0f}\n"
        "Fraction involving factual errors: {fraction_factual_errors:.3f}\n"
        "Fraction involving hallucinated technical language: {fraction_hallucinated_technical_language:.3f}\n"
        "Fraction involving verbose but shallow responses: {fraction_verbose_but_shallow:.3f}\n"
        "Fraction involving formatting/structure exploitation: {fraction_formatting_structure_exploitation:.3f}\n\n"
        "Candidate-level correlations:\n"
        "corr(gap, response_length_tokens): {gap_vs_response_length:.3f}\n"
        "corr(gap, proxy_reward): {gap_vs_proxy_reward:.3f}\n"
        "corr(gap, true_reward): {gap_vs_true_reward:.3f}\n"
    ).format(**summary_stats, **candidate_correlations)
