from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd


REQUIRED_RAW_COLUMNS = {
    "prompt_id",
    "prompt",
    "candidate_id",
    "response",
    "generator_model",
    "temperature",
    "top_p",
    "max_new_tokens",
    "run_mode",
}

REQUIRED_SCORED_COLUMNS = {
    "prompt_id",
    "prompt",
    "candidate_id",
    "response",
    "proxy_reward",
    "true_reward",
    "gap",
    "proxy_model",
    "true_model",
    "run_mode",
}

_NON_WORD_RE = re.compile(r"[^A-Za-z0-9]+")
_PLACEHOLDER_MARKERS = (
    "synthetic",
    "placeholder",
    "lorem ipsum",
    "dummy response",
    "todo",
)


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_prompts(prompt_file: Path) -> List[Dict[str, object]]:
    if prompt_file.suffix != ".jsonl":
        raise ValueError("Prompt file must be .jsonl for the real experiment.")

    rows = []
    with prompt_file.open() as handle:
        for i, line in enumerate(handle):
            record = json.loads(line)
            prompt = record.get("prompt") or record.get("instruction") or record.get("question")
            if not prompt:
                raise ValueError("Missing prompt field on line %d of %s." % (i + 1, prompt_file))
            prompt_id = record.get("prompt_id", i)
            rows.append({"prompt_id": int(prompt_id), "prompt": str(prompt)})
    return rows


def write_prompts_jsonl(prompt_rows: Sequence[Dict[str, object]], prompt_file: Path) -> None:
    ensure_parent_dir(prompt_file)
    with prompt_file.open("w") as handle:
        for row in prompt_rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def append_jsonl_rows(rows: Sequence[Dict[str, object]], output_file: Path) -> None:
    if not rows:
        return
    ensure_parent_dir(output_file)
    with output_file.open("a") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _load_jsonl_rows(input_file: Path) -> List[Dict[str, object]]:
    rows = []
    with input_file.open() as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def load_candidate_rows(input_file: Path) -> List[Dict[str, object]]:
    if input_file.suffix != ".jsonl":
        raise ValueError("Raw candidate input must be a .jsonl file.")
    rows = _load_jsonl_rows(input_file)
    missing = REQUIRED_RAW_COLUMNS - set(rows[0].keys()) if rows else REQUIRED_RAW_COLUMNS
    if missing:
        raise ValueError("Candidate file is missing required columns: %s" % sorted(missing))
    return rows


def load_scored_candidates(input_file: Path) -> pd.DataFrame:
    if input_file.suffix != ".parquet":
        raise ValueError("Scored candidates input must be a .parquet file.")
    frame = pd.read_parquet(input_file)
    missing = REQUIRED_SCORED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError("Scored parquet is missing required columns: %s" % sorted(missing))
    return frame.copy()


def load_existing_generation_keys(output_file: Path) -> Set[Tuple[int, int]]:
    if not output_file.exists():
        return set()
    rows = _load_jsonl_rows(output_file)
    return {(int(row["prompt_id"]), int(row["candidate_id"])) for row in rows}


def load_existing_scored_keys(scored_file: Path) -> Set[Tuple[int, int]]:
    if not scored_file.exists():
        return set()
    frame = pd.read_parquet(scored_file, columns=["prompt_id", "candidate_id"])
    return set(zip(frame["prompt_id"].astype(int), frame["candidate_id"].astype(int)))


def save_scored_candidates(frame: pd.DataFrame, output_file: Path) -> None:
    ensure_parent_dir(output_file)
    frame.to_parquet(output_file, index=False)


def require_live_or_replay_real(run_mode: str) -> None:
    if run_mode not in {"live", "replay_real"}:
        raise ValueError("run_mode must be 'live' or 'replay_real'; synthetic mode is forbidden.")


def validate_natural_language_response(text: str) -> bool:
    stripped = text.strip()
    lowered = stripped.lower()
    if not lowered:
        return False
    if any(marker in lowered for marker in _PLACEHOLDER_MARKERS):
        return False
    normalized = _NON_WORD_RE.sub(" ", stripped).strip()
    tokens = [token for token in normalized.split() if token]
    alpha_tokens = [token for token in tokens if any(char.isalpha() for char in token)]
    if not alpha_tokens:
        return False
    if len(alpha_tokens) >= 2:
        return True

    # Allow concise but natural one-word or label-style answers such as
    # "Yes", "Read", or quoted taglines, while still rejecting empty/noisy text.
    alpha_chars = sum(1 for char in stripped if char.isalpha())
    return alpha_chars >= 3


def validate_scored_candidates(
    frame: pd.DataFrame,
    run_mode: str,
    minimum_prompt_count: int = 300,
    minimum_candidates_per_prompt: int = 20,
) -> None:
    require_live_or_replay_real(run_mode)
    if frame.empty:
        raise ValueError("No scored candidates found.")
    if frame["run_mode"].nunique() != 1 or str(frame["run_mode"].iloc[0]) != run_mode:
        raise ValueError("Scored data run_mode does not match the requested run_mode.")
    if frame["proxy_model"].nunique() != 1 or frame["true_model"].nunique() != 1:
        raise ValueError("Expected a single proxy model and a single true model in scored output.")
    if str(frame["proxy_model"].iloc[0]) == str(frame["true_model"].iloc[0]):
        raise ValueError("proxy_model and true_model must be different evaluators.")
    if frame[["proxy_reward", "true_reward", "gap"]].isnull().any().any():
        raise ValueError("Missing scores detected in candidates_scored.parquet.")
    if minimum_prompt_count < 1:
        raise ValueError("minimum_prompt_count must be at least 1.")
    if minimum_candidates_per_prompt < 1:
        raise ValueError("minimum_candidates_per_prompt must be at least 1.")

    prompt_count = int(frame["prompt_id"].nunique())
    if prompt_count < minimum_prompt_count:
        raise ValueError("Need at least %d prompts with scored candidates; found %d." % (minimum_prompt_count, prompt_count))

    prompt_sizes = frame.groupby("prompt_id")["candidate_id"].nunique()
    too_small = prompt_sizes[prompt_sizes < minimum_candidates_per_prompt]
    if not too_small.empty:
        raise ValueError(
            "Each prompt must have at least %d candidates; failing prompt_ids include %s."
            % (minimum_candidates_per_prompt, list(too_small.index[:10]))
        )

    invalid_mask = ~frame["response"].astype(str).map(validate_natural_language_response)
    if invalid_mask.any():
        sample_bad = frame.loc[invalid_mask, ["prompt_id", "candidate_id", "response"]].head(3)
        raise ValueError(
            "Found placeholder or non-natural-language responses in scored data: %s"
            % sample_bad.to_dict(orient="records")
        )


def load_real_dataset_prompts(
    dataset_name: str,
    num_prompts: int,
    seed: int,
    split: Optional[str] = None,
) -> List[Dict[str, object]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("datasets is required for prompt preparation.") from exc

    dataset_map = {
        "alpaca": ("tatsu-lab/alpaca", split or "train"),
        "hh-rlhf": ("Anthropic/hh-rlhf", split or "train"),
        "rewardbench": ("allenai/reward-bench", split or "filtered"),
    }
    if dataset_name not in dataset_map:
        raise ValueError("Unsupported dataset: %s" % dataset_name)

    dataset_id, dataset_split = dataset_map[dataset_name]
    dataset = load_dataset(dataset_id, split=dataset_split)
    shuffled = dataset.shuffle(seed=seed)

    extractor = {
        "alpaca": _extract_alpaca_prompt,
        "hh-rlhf": _extract_hh_prompt,
        "rewardbench": _extract_rewardbench_prompt,
    }[dataset_name]

    prompts = []
    seen = set()
    for row in shuffled:
        prompt = extractor(row)
        if not prompt:
            continue
        prompt_text = str(prompt).strip()
        if prompt_text in seen:
            continue
        seen.add(prompt_text)
        prompts.append({"prompt_id": len(prompts), "prompt": prompt_text, "source_dataset": dataset_name})
        if len(prompts) >= num_prompts:
            break

    if len(prompts) < num_prompts:
        raise ValueError(
            "Only extracted %d prompts from %s, fewer than requested %d."
            % (len(prompts), dataset_name, num_prompts)
        )
    return prompts


def _extract_alpaca_prompt(row: Dict[str, object]) -> Optional[str]:
    instruction = str(row.get("instruction", "")).strip()
    if not instruction:
        return None
    input_text = str(row.get("input", "")).strip()
    if input_text:
        return instruction + "\n\nInput:\n" + input_text
    return instruction


def _extract_hh_prompt(row: Dict[str, object]) -> Optional[str]:
    chosen = str(row.get("chosen", "")).strip()
    if not chosen:
        return None
    marker = "\n\nAssistant:"
    if marker not in chosen:
        return None
    prompt = chosen.split(marker, 1)[0].strip()
    prompt = prompt.replace("Human:", "").strip()
    return prompt or None


def _extract_rewardbench_prompt(row: Dict[str, object]) -> Optional[str]:
    for key in ("prompt", "instruction", "question"):
        value = row.get(key)
        if value:
            return str(value).strip()
    chosen = row.get("chosen")
    if isinstance(chosen, dict):
        content = chosen.get("content")
        if content:
            return str(content).strip()
    return None
