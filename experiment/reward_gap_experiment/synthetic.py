from __future__ import annotations

import numpy as np
import pandas as pd


def build_synthetic_candidates(
    num_prompts: int,
    num_candidates: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows = []
    for prompt_id in range(num_prompts):
        prompt = f"Synthetic prompt {prompt_id}"
        base_difficulty = rng.normal(0.0, 0.4)

        for candidate_id in range(num_candidates):
            quality = rng.normal(loc=0.9 - 0.3 * base_difficulty, scale=0.35)

            # Heavy-tailed hackiness creates rare candidates with inflated proxy
            # reward and sharply degraded true reward. Most samples remain
            # reasonably aligned, which keeps the mean gap modest while making
            # the upper tail dangerous under optimization.
            tail_gate = rng.random() < 0.012
            hackiness = (
                rng.lognormal(mean=0.9 + 0.25 * base_difficulty, sigma=0.95)
                if tail_gate
                else rng.lognormal(mean=-1.25, sigma=0.30)
            )

            proxy_reward = quality + 0.16 * hackiness + rng.normal(0.0, 0.07)
            true_reward = (
                quality
                - 0.08 * hackiness
                - 1.70 * max(hackiness - 1.8, 0.0)
                + rng.normal(0.0, 0.07)
            )
            response = (
                f"Synthetic response q={quality:.3f} h={hackiness:.3f} "
                f"proxy={proxy_reward:.3f} true={true_reward:.3f}"
            )

            rows.append(
                {
                    "prompt_id": prompt_id,
                    "prompt": prompt,
                    "candidate_id": candidate_id,
                    "response": response,
                    "proxy_reward": float(proxy_reward),
                    "true_reward": float(true_reward),
                    "quality_latent": float(quality),
                    "hackiness_latent": float(hackiness),
                }
            )

    return pd.DataFrame(rows)
