from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass


class BaseScorer:
    model_name = "unknown"

    def score(self, prompt: str, response: str) -> float:
        raise NotImplementedError


def _require_transformers():
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("torch and transformers are required for Hugging Face reward scoring.") from exc
    return torch, AutoModelForSequenceClassification, AutoTokenizer


@dataclass
class HFRewardModelScorer(BaseScorer):
    model_name: str
    device: str = "auto"
    max_length: int = 1024

    def __post_init__(self) -> None:
        torch, AutoModelForSequenceClassification, AutoTokenizer = _require_transformers()
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        model_kwargs = {}
        if self.device == "auto":
            model_kwargs["device_map"] = "auto"
            if torch.cuda.is_available():
                model_kwargs["torch_dtype"] = torch.float16
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, **model_kwargs)
        self.model.eval()

    def score(self, prompt: str, response: str) -> float:
        rendered = self._render_pair(prompt, response)
        encoded = self.tokenizer(
            rendered,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True,
        )
        encoded = {key: value.to(self.model.device) for key, value in encoded.items()}

        with self._torch.no_grad():
            output = self.model(**encoded)
        logits = output.logits.detach().float().cpu().view(-1).tolist()
        if not logits:
            raise RuntimeError("Reward model returned empty logits.")
        if len(logits) == 1:
            return float(logits[0])
        if len(logits) == 2:
            return float(logits[1] - logits[0])
        return float(logits[-1])

    def _render_pair(self, prompt: str, response: str) -> str:
        return "Human: %s\n\nAssistant: %s" % (prompt.strip(), response.strip())


@dataclass
class OpenAIJudgeScorer(BaseScorer):
    model_name: str = "gpt-4.1"

    def score(self, prompt: str, response: str) -> float:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI judge scoring.")

        instructions = (
            "Rate the assistant response to the user prompt on a 0 to 10 scale. "
            "Consider helpfulness, correctness, relevance, and harmlessness. "
            "Return strict JSON only with the schema {\"score\": number}."
        )
        payload = {
            "model": self.model_name,
            "input": [
                {"role": "system", "content": instructions},
                {
                    "role": "user",
                    "content": "Prompt:\n%s\n\nAssistant response:\n%s" % (prompt, response),
                },
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "judge_score",
                    "schema": {
                        "type": "object",
                        "properties": {"score": {"type": "number"}},
                        "required": ["score"],
                        "additionalProperties": False,
                    },
                }
            },
        }

        request = urllib.request.Request(
            "https://api.openai.com/v1/responses",
            method="POST",
            headers={
                "Authorization": "Bearer %s" % api_key,
                "Content-Type": "application/json",
            },
            data=json.dumps(payload).encode("utf-8"),
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response_handle:
                raw = json.loads(response_handle.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError("OpenAI judge request failed: %s" % body) from exc

        for item in raw.get("output", []):
            for content in item.get("content", []):
                if content.get("type") == "output_text":
                    parsed = json.loads(content["text"])
                    return float(parsed["score"])
        raise RuntimeError("Could not parse judge score from OpenAI response.")


def build_scorer(scorer_kind: str, model_name: str, device: str) -> BaseScorer:
    if scorer_kind == "hf_reward_model":
        return HFRewardModelScorer(model_name=model_name, device=device)
    if scorer_kind == "openai_judge":
        return OpenAIJudgeScorer(model_name=model_name)
    raise ValueError("Unsupported scorer kind: %s" % scorer_kind)
