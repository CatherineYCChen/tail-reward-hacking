from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import analyze_real_candidates, generate_real_candidates, prepare_real_prompts, score_real_candidates


DEFAULT_N_VALUES = [1, 2, 5, 10, 20, 50]
DEFAULT_ANALYSIS_SEEDS = [7, 13, 29]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a real best-of-n reward hacking experiment without synthetic fallbacks."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare-prompts",
        help="Download a real prompt dataset and save data/prompts.jsonl.",
    )
    prepare.add_argument(
        "--dataset",
        choices=["alpaca", "hh-rlhf", "rewardbench"],
        default="alpaca",
    )
    prepare.add_argument("--num-prompts", type=int, default=500)
    prepare.add_argument("--seed", type=int, default=7)
    prepare.add_argument("--split", type=str, default=None)
    prepare.add_argument("--prompt-file", type=Path, default=Path("data/prompts.jsonl"))

    generate = subparsers.add_parser(
        "generate",
        help="Generate real candidate responses with a Hugging Face instruct model.",
    )
    generate.add_argument("--run-mode", choices=["live", "replay_real"], default="live")
    generate.add_argument("--prompt-file", type=Path, default=Path("data/prompts.jsonl"))
    generate.add_argument("--output-dir", type=Path, default=Path("outputs/real_run"))
    generate.add_argument(
        "--generator-model",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
    )
    generate.add_argument("--num-prompts", type=int, default=500)
    generate.add_argument("--num-candidates", type=int, default=50)
    generate.add_argument("--seed", type=int, default=7)
    generate.add_argument("--temperature", type=float, default=0.8)
    generate.add_argument("--top-p", type=float, default=0.95)
    generate.add_argument("--max-new-tokens", type=int, default=256)
    generate.add_argument("--device", type=str, default="auto")
    generate.add_argument(
        "--raw-output-file",
        type=Path,
        default=None,
        help="Defaults to <output-dir>/candidates_raw.jsonl",
    )

    score = subparsers.add_parser(
        "score",
        help="Score generated candidates with a real proxy and a separate real true evaluator.",
    )
    score.add_argument("--run-mode", choices=["live", "replay_real"], default="live")
    score.add_argument("--output-dir", type=Path, default=Path("outputs/real_run"))
    score.add_argument(
        "--input-file",
        type=Path,
        default=None,
        help="Defaults to <output-dir>/candidates_raw.jsonl",
    )
    score.add_argument(
        "--scored-output-file",
        type=Path,
        default=None,
        help="Defaults to <output-dir>/candidates_scored.parquet",
    )
    score.add_argument(
        "--proxy-scorer-kind",
        choices=["hf_reward_model", "openai_judge"],
        default="hf_reward_model",
    )
    score.add_argument(
        "--proxy-model-name",
        type=str,
        default="OpenAssistant/reward-model-deberta-v3-large-v2",
    )
    score.add_argument(
        "--true-scorer-kind",
        choices=["hf_reward_model", "openai_judge"],
        default="openai_judge",
    )
    score.add_argument("--true-model-name", type=str, default="gpt-4.1")
    score.add_argument("--device", type=str, default="auto")
    score.add_argument("--max-score-pairs", type=int, default=None)

    analyze = subparsers.add_parser(
        "analyze",
        help="Run BoN and random-baseline analysis on scored real candidates.",
    )
    analyze.add_argument("--run-mode", choices=["live", "replay_real"], default="live")
    analyze.add_argument("--output-dir", type=Path, default=Path("outputs/real_run"))
    analyze.add_argument(
        "--input-file",
        type=Path,
        default=None,
        help="Defaults to <output-dir>/candidates_scored.parquet",
    )
    analyze.add_argument("--n-values", type=int, nargs="+", default=DEFAULT_N_VALUES)
    analyze.add_argument("--alpha", type=float, default=0.95)
    analyze.add_argument("--top-proxy-quantile", type=float, default=0.95)
    analyze.add_argument("--num-percentile-bins", type=int, default=20)
    analyze.add_argument("--analysis-seeds", type=int, nargs="+", default=DEFAULT_ANALYSIS_SEEDS)

    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.command == "prepare-prompts":
        prepare_real_prompts(args)
    elif args.command == "generate":
        generate_real_candidates(args)
    elif args.command == "score":
        score_real_candidates(args)
    elif args.command == "analyze":
        analyze_real_candidates(args)
    else:
        raise ValueError("Unknown command.")

    return 0
