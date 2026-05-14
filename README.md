# Reward Gap Tail Experiment

This repository now supports a strictly real-only best-of-n reward hacking experiment. The main experiment path does not permit synthetic generation or synthetic scoring, and it fails loudly instead of silently falling back.

## Real experiment outputs

The real pipeline writes:

- `data/prompts.jsonl`
- `outputs/real_run/candidates_raw.jsonl`
- `outputs/real_run/candidates_scored.parquet`
- `outputs/real_run/summary.csv`
- `outputs/real_run/summary.json`
- `outputs/real_run/summary_by_seed.csv`
- `outputs/real_run/gap_by_proxy_percentile.csv`
- `outputs/real_run/plot_bon_rewards.png`
- `outputs/real_run/plot_gap_tail.png`
- `outputs/real_run/plot_gap_by_proxy_percentile.png`
- `outputs/real_run/plot_random_vs_bon_tail.png`
- `outputs/real_run/scatter_proxy_vs_true.png`
- `outputs/real_run/scatter_proxy_vs_gap.png`
- `outputs/real_run/gap_vs_response_length.png`
- `outputs/real_run/histogram_gap_all_candidates.png`
- `outputs/real_run/histogram_gap_selected_by_n.png`
- `outputs/real_run/examples_largest_gap_n*.csv`
- `outputs/real_run/examples_largest_gap_annotated_n*.csv`
- `outputs/real_run/examples_highest_proxy_n*.csv`
- `outputs/real_run/manual_review_largest_gap.csv`
- `outputs/real_run/manual_review_high_proxy.csv`
- `outputs/real_run/qualitative_summary.txt`
- `outputs/real_run/implementation_note_real_run.txt`

## Recommended real setup

- Dataset: `tatsu-lab/alpaca` with 500 prompts saved to `data/prompts.jsonl`
- Generator: `Qwen/Qwen2.5-1.5B-Instruct`
- Proxy reward model: `OpenAssistant/reward-model-deberta-v3-large-v2`
- True evaluator: `gpt-4.1` via OpenAI judge, or a different stronger Hugging Face reward model if no API key is available
- Sampling: `K=50`, `temperature=0.8`, `top_p=0.95`, `max_new_tokens=256`

## Exact commands

Prepare the real prompt file:

```bash
python3 experiment/run_reward_gap_experiment.py prepare-prompts \
  --dataset alpaca \
  --num-prompts 500 \
  --seed 7 \
  --prompt-file data/prompts.jsonl
```

Generate real candidates:

```bash
python3 experiment/run_reward_gap_experiment.py generate \
  --run-mode live \
  --prompt-file data/prompts.jsonl \
  --generator-model Qwen/Qwen2.5-1.5B-Instruct \
  --num-prompts 500 \
  --num-candidates 50 \
  --temperature 0.8 \
  --top-p 0.95 \
  --max-new-tokens 256 \
  --output-dir outputs/real_run
```

Score the candidates with a real proxy and a separate real true evaluator:

```bash
OPENAI_API_KEY=... python3 experiment/run_reward_gap_experiment.py score \
  --run-mode live \
  --output-dir outputs/real_run \
  --proxy-scorer-kind hf_reward_model \
  --proxy-model-name OpenAssistant/reward-model-deberta-v3-large-v2 \
  --true-scorer-kind openai_judge \
  --true-model-name gpt-4.1 \
  --min-prompt-count 300 \
  --min-candidates-per-prompt 20
```

If you do not have an OpenAI API key, you must still use a different evaluator for `R_t`, for example:

```bash
python3 experiment/run_reward_gap_experiment.py score \
  --run-mode live \
  --output-dir outputs/real_run \
  --proxy-scorer-kind hf_reward_model \
  --proxy-model-name OpenAssistant/reward-model-deberta-v3-large-v2 \
  --true-scorer-kind hf_reward_model \
  --true-model-name <different-stronger-reward-model>
```

Run analysis:

```bash
python3 experiment/run_reward_gap_experiment.py analyze \
  --run-mode live \
  --output-dir outputs/real_run \
  --n-values 1 2 5 10 20 50 \
  --analysis-seeds 7 13 29 \
  --alpha 0.95 \
  --top-proxy-quantile 0.95 \
  --num-percentile-bins 20 \
  --min-prompt-count 300 \
  --min-candidates-per-prompt 20
```

For staged real smoke tests, lower the thresholds explicitly. For example, Stage A can use:

```bash
python3 experiment/run_reward_gap_experiment.py score \
  --run-mode live \
  --output-dir outputs/stage_a \
  --proxy-scorer-kind hf_reward_model \
  --proxy-model-name OpenAssistant/reward-model-deberta-v3-large-v2 \
  --true-scorer-kind openai_judge \
  --true-model-name gpt-4.1 \
  --min-prompt-count 10 \
  --min-candidates-per-prompt 3
```

## What the analysis computes

For `n in [1, 2, 5, 10, 20, 50]`, the analysis:

- restricts each prompt to candidates with `candidate_id < n`
- selects the winner with maximum proxy reward
- reports average proxy reward, average true reward, mean gap, CVaR_95 gap, selected gap quantiles and max, top-proxy gap diagnostics, correlation diagnostics, z-normalized gap metrics, the selected-winner top proxy threshold, the global 95th percentile gap threshold, conditional failure rate, and selected count
- compares that against a random baseline
- repeats the analysis for at least 3 seeds via candidate-order permutation and random selection

The analysis now also computes global candidate-level z-scores:

- `proxy_z = zscore(proxy_reward)`
- `true_z = zscore(true_reward)`
- `gap_z = proxy_z - true_z`

And exports qualitative inspection tables for each `n`:

- `examples_largest_gap_n{n}.csv`
- `examples_highest_proxy_n{n}.csv`

These make it easier to inspect whether large gaps look like genuine reward hacking or evaluator-scale mismatch.

For manual labeling, the analysis also exports pooled review sheets across all `n`:

- `manual_review_largest_gap.csv`
- `manual_review_high_proxy.csv`

These include blank columns for human review of factual errors, hallucinations, incomplete answers, verbose-but-shallow behavior, and style/formatting reward effects.

The qualitative pass also adds lightweight heuristic annotations to the largest-gap examples, including:

- response length in tokens
- list/number/equation markers
- confident vs uncertain language
- factual inconsistency heuristics
- hallucinated technical-detail heuristics
- verbosity, repetition, and formatting-quality heuristics

No LLM-based labeling is used for this step.

## Validity checks

The pipeline fails if any of the following are false:

- `run_mode` is `live` or `replay_real`
- `proxy_model != true_model`
- there are at least 300 prompts with scored candidates
- each prompt has at least 20 candidates
- no score is missing
- scored responses pass placeholder/non-language screening, including short real answers

## Caveats

This is empirical evidence only if:

- generations are from a real LLM
- proxy scores are from a real proxy model or judge
- true scores are from a separate stronger evaluator
- no synthetic rewards are used

Practical caveats:

- The repository does not vendor model weights or datasets. You need working access to Hugging Face, and optionally the OpenAI API for the stronger judge path.
- The default proxy model and generator are chosen to be relatively small, but a full 500 x 50 run is still substantial.
- If you use the no-API fallback for `R_t`, you must choose a genuinely different evaluator and should document why it is stronger.

## Stage C batch script

`run_stageC.sbatch` is included for a real Stage C run with:

- 100 prompts
- 20 candidates per prompt
- `max_new_tokens=128`
- generator `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- proxy `OpenAssistant/reward-model-deberta-v3-large-v2`
- true evaluator `Skywork/Skywork-Reward-Llama-3.1-8B-v0.2`
- `n_values: 1 2 5 10 20`

The script prepares prompts, generates candidates, scores them, and runs analysis, but it is not launched automatically.
