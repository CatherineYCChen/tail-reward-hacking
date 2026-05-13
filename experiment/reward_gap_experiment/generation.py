from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence


def _require_torch_and_transformers():
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("torch and transformers are required for real generation.") from exc
    return torch, AutoModelForCausalLM, AutoTokenizer


@dataclass
class HFGenerator:
    model_name: str
    device: str = "auto"
    max_new_tokens: int = 256
    temperature: float = 0.8
    top_p: float = 0.95
    seed: int = 7

    def __post_init__(self) -> None:
        torch, AutoModelForCausalLM, AutoTokenizer = _require_torch_and_transformers()
        self._torch = torch
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {}
        if self.device == "auto":
            model_kwargs["device_map"] = "auto"
            if torch.cuda.is_available():
                model_kwargs["torch_dtype"] = torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        self.model.eval()

    def sample(self, prompt: str, num_candidates: int) -> List[str]:
        rendered_prompt = self._render_prompt(prompt)
        inputs = self.tokenizer(rendered_prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs["attention_mask"].to(self.model.device)

        with self._torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                max_new_tokens=self.max_new_tokens,
                num_return_sequences=num_candidates,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        prompt_len = input_ids.shape[1]
        responses = []
        for output_ids in outputs:
            text = self.tokenizer.decode(output_ids[prompt_len:], skip_special_tokens=True).strip()
            if not text:
                raise RuntimeError("Generation produced an empty response for prompt: %s" % prompt[:80])
            responses.append(text)
        return responses

    def _render_prompt(self, prompt: str) -> str:
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            messages = [{"role": "user", "content": prompt}]
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return prompt


def build_candidate_rows(
    prompt_id: int,
    prompt: str,
    responses: Sequence[str],
    generator_model: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    run_mode: str,
    start_candidate_id: int = 0,
) -> List[Dict[str, object]]:
    rows = []
    for offset, response in enumerate(responses):
        rows.append(
            {
                "prompt_id": int(prompt_id),
                "prompt": prompt,
                "candidate_id": int(start_candidate_id + offset),
                "response": response,
                "generator_model": generator_model,
                "temperature": float(temperature),
                "top_p": float(top_p),
                "max_new_tokens": int(max_new_tokens),
                "run_mode": run_mode,
            }
        )
    return rows
