from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .data import (
    append_jsonl_rows,
    ensure_parent_dir,
    load_candidate_rows,
    load_existing_generation_keys,
    load_existing_scored_keys,
    load_prompts,
    load_real_dataset_prompts,
    load_scored_candidates,
    require_live_or_replay_real,
    save_scored_candidates,
    validate_scored_candidates,
    write_prompts_jsonl,
)
from .generation import HFGenerator, build_candidate_rows
from .metrics import (
    add_normalized_columns,
    aggregate_seed_summaries,
    summarize_gap_by_proxy_percentile,
    summarize_seed,
)
from .plotting import make_all_plots
from .qualitative import (
    add_response_length_tokens,
    summarize_candidate_correlations,
    write_annotated_examples_and_summary,
)
from .scorers import build_scorer


def prepare_real_prompts(args: Namespace) -> None:
    prompts = load_real_dataset_prompts(
        dataset_name=args.dataset,
        num_prompts=args.num_prompts,
        seed=args.seed,
        split=args.split,
    )
    write_prompts_jsonl(prompts, args.prompt_file)


def generate_real_candidates(args: Namespace) -> None:
    require_live_or_replay_real(args.run_mode)
    prompts = load_prompts(args.prompt_file)[: args.num_prompts]
    if len(prompts) < args.num_prompts:
        raise ValueError("Prompt file only contains %d prompts." % len(prompts))

    output_file = args.raw_output_file or (args.output_dir / "candidates_raw.jsonl")
    ensure_parent_dir(output_file)

    existing_keys = load_existing_generation_keys(output_file)
    generator = HFGenerator(
        model_name=args.generator_model,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
    )

    for prompt_row in prompts:
        prompt_id = int(prompt_row["prompt_id"])
        prompt = str(prompt_row["prompt"])
        existing_count = sum(1 for key in existing_keys if key[0] == prompt_id)
        if existing_count >= args.num_candidates:
            continue

        remaining = args.num_candidates - existing_count
        responses = generator.sample(prompt, num_candidates=remaining)
        new_rows = build_candidate_rows(
            prompt_id=prompt_id,
            prompt=prompt,
            responses=responses,
            generator_model=args.generator_model,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            run_mode=args.run_mode,
            start_candidate_id=existing_count,
        )
        append_jsonl_rows(new_rows, output_file)
        for row in new_rows:
            existing_keys.add((int(row["prompt_id"]), int(row["candidate_id"])))


def score_real_candidates(args: Namespace) -> None:
    require_live_or_replay_real(args.run_mode)
    input_file = args.input_file or (args.output_dir / "candidates_raw.jsonl")
    scored_output_file = args.scored_output_file or (args.output_dir / "candidates_scored.parquet")

    raw_rows = load_candidate_rows(input_file)
    if not raw_rows:
        raise ValueError("No raw candidates found to score.")

    if args.proxy_model_name == args.true_model_name:
        raise ValueError("proxy_model and true_model must be different evaluators.")

    existing_scored_keys = load_existing_scored_keys(scored_output_file)
    proxy_scorer = build_scorer(args.proxy_scorer_kind, args.proxy_model_name, args.device)
    true_scorer = build_scorer(args.true_scorer_kind, args.true_model_name, args.device)

    if getattr(proxy_scorer, "model_name", None) == getattr(true_scorer, "model_name", None):
        raise ValueError("proxy_model and true_model resolved to the same evaluator.")

    existing_frame = pd.read_parquet(scored_output_file) if scored_output_file.exists() else pd.DataFrame()
    new_records = []

    for row in raw_rows:
        key = (int(row["prompt_id"]), int(row["candidate_id"]))
        if key in existing_scored_keys:
            continue

        prompt = str(row["prompt"])
        response = str(row["response"])
        proxy_reward = proxy_scorer.score(prompt, response)
        true_reward = true_scorer.score(prompt, response)
        record = dict(row)
        record["proxy_reward"] = float(proxy_reward)
        record["true_reward"] = float(true_reward)
        record["gap"] = float(proxy_reward - true_reward)
        record["proxy_model"] = args.proxy_model_name
        record["true_model"] = args.true_model_name
        new_records.append(record)
        existing_scored_keys.add(key)

        if args.max_score_pairs is not None and len(new_records) >= args.max_score_pairs:
            break

        if len(new_records) >= 25:
            _flush_scored_records(existing_frame, new_records, scored_output_file)
            if scored_output_file.exists():
                existing_frame = pd.read_parquet(scored_output_file)
            new_records = []

    if new_records:
        _flush_scored_records(existing_frame, new_records, scored_output_file)

    scored_frame = load_scored_candidates(scored_output_file)
    validate_scored_candidates(
        scored_frame,
        run_mode=args.run_mode,
        minimum_prompt_count=args.min_prompt_count,
        minimum_candidates_per_prompt=args.min_candidates_per_prompt,
    )


def analyze_real_candidates(args: Namespace) -> None:
    require_live_or_replay_real(args.run_mode)
    input_file = args.input_file or (args.output_dir / "candidates_scored.parquet")
    ensure_parent_dir(args.output_dir / "summary.csv")

    candidates = load_scored_candidates(input_file)
    validate_scored_candidates(
        candidates,
        run_mode=args.run_mode,
        minimum_prompt_count=args.min_prompt_count,
        minimum_candidates_per_prompt=args.min_candidates_per_prompt,
    )
    candidates = add_normalized_columns(candidates)
    candidates = add_response_length_tokens(candidates)

    bon_seed_frames = []
    random_seed_frames = []
    failure_thresholds = {}
    example_seed = int(args.analysis_seeds[0])
    example_selected_by_n = {}
    for seed in args.analysis_seeds:
        bon_frame, random_frame, failure_threshold, selected_by_n = summarize_seed(
            candidates=candidates,
            n_values=args.n_values,
            alpha=args.alpha,
            top_proxy_quantile=args.top_proxy_quantile,
            seed=seed,
        )
        bon_seed_frames.append(bon_frame)
        random_seed_frames.append(random_frame)
        failure_thresholds[int(seed)] = float(failure_threshold)
        if int(seed) == example_seed:
            example_selected_by_n = selected_by_n

    bon_by_seed = pd.concat(bon_seed_frames, ignore_index=True)
    random_by_seed = pd.concat(random_seed_frames, ignore_index=True)
    summary_by_seed = pd.concat([random_by_seed, bon_by_seed], ignore_index=True)
    summary = aggregate_seed_summaries(summary_by_seed)
    percentile_summary = summarize_gap_by_proxy_percentile(
        candidates=candidates,
        num_bins=args.num_percentile_bins,
    )

    summary.to_csv(args.output_dir / "summary.csv", index=False)
    summary_by_seed.to_csv(args.output_dir / "summary_by_seed.csv", index=False)
    percentile_summary.to_csv(args.output_dir / "gap_by_proxy_percentile.csv", index=False)
    _write_example_exports(example_selected_by_n, args.output_dir)
    candidate_correlations = summarize_candidate_correlations(candidates)
    qualitative_summary = write_annotated_examples_and_summary(
        example_selected_by_n,
        args.output_dir,
        candidate_correlations,
    )

    with (args.output_dir / "summary.json").open("w") as handle:
        json.dump(
            {
                "run_mode": args.run_mode,
                "n_values": args.n_values,
                "alpha": args.alpha,
                "top_proxy_quantile": args.top_proxy_quantile,
                "analysis_seeds": args.analysis_seeds,
                "example_seed": example_seed,
                "min_prompt_count": args.min_prompt_count,
                "min_candidates_per_prompt": args.min_candidates_per_prompt,
                "failure_thresholds_by_seed": failure_thresholds,
                "num_prompts": int(candidates["prompt_id"].nunique()),
                "num_candidates_total": int(len(candidates)),
                "generator_model": str(candidates["generator_model"].iloc[0]),
                "proxy_model": str(candidates["proxy_model"].iloc[0]),
                "true_model": str(candidates["true_model"].iloc[0]),
                "candidate_correlations": candidate_correlations,
                "qualitative_summary": qualitative_summary,
                "summary": summary.to_dict(orient="records"),
                "summary_by_seed": summary_by_seed.to_dict(orient="records"),
            },
            handle,
            indent=2,
        )

    bon_summary = summary[summary["strategy"] == "bon"].copy()
    random_summary = summary[summary["strategy"] == "random"].copy()
    make_all_plots(
        bon_summary=bon_summary,
        random_summary=random_summary,
        percentile_summary=percentile_summary,
        all_candidates=candidates,
        selected_by_n=example_selected_by_n,
        output_dir=args.output_dir,
    )

    note_file = args.output_dir / "implementation_note_real_run.txt"
    note_file.write_text(_build_implementation_note(candidates))


def _flush_scored_records(
    existing_frame: pd.DataFrame,
    new_records: List[Dict[str, object]],
    scored_output_file: Path,
) -> None:
    new_frame = pd.DataFrame(new_records)
    if existing_frame.empty:
        combined = new_frame
    else:
        combined = pd.concat([existing_frame, new_frame], ignore_index=True)
        combined = combined.drop_duplicates(subset=["prompt_id", "candidate_id"], keep="last")
    save_scored_candidates(combined, scored_output_file)


def _build_implementation_note(candidates: pd.DataFrame) -> str:
    generator_model = str(candidates["generator_model"].iloc[0])
    proxy_model = str(candidates["proxy_model"].iloc[0])
    true_model = str(candidates["true_model"].iloc[0])
    return (
        "Real components used in this run:\n"
        "- Generator: %s\n"
        "- Proxy evaluator: %s\n"
        "- True evaluator: %s\n"
        "- Responses are generated by a real LLM and scored by real evaluators.\n"
        "- Synthetic rewards are not used anywhere in the main experiment.\n"
        "\n"
        "This counts as empirical evidence only if the generations came from the listed real LLM,\n"
        "the proxy scores came from the listed real proxy evaluator, the true scores came from the\n"
        "separate listed true evaluator, and no synthetic rewards were used.\n"
    ) % (generator_model, proxy_model, true_model)


def _write_example_exports(selected_by_n: Dict[int, pd.DataFrame], output_dir: Path) -> None:
    example_columns = [
        "prompt",
        "response",
        "proxy_reward",
        "true_reward",
        "gap",
        "proxy_z",
        "true_z",
        "gap_z",
        "candidate_id",
    ]
    for n, frame in selected_by_n.items():
        largest_gap = (
            frame.sort_values(["gap", "proxy_reward"], ascending=[False, False])
            .head(10)
            .loc[:, example_columns]
            .copy()
        )
        highest_proxy = (
            frame.sort_values(["proxy_reward", "gap"], ascending=[False, False])
            .head(10)
            .loc[:, example_columns]
            .copy()
        )
        largest_gap.to_csv(output_dir / ("examples_largest_gap_n%d.csv" % n), index=False)
        highest_proxy.to_csv(output_dir / ("examples_highest_proxy_n%d.csv" % n), index=False)
