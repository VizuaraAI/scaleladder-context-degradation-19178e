"""Model runners.

Two implementations behind one interface:

  HFRunner   loads an open-weights checkpoint locally, which is where all of
             the scale-ladder evidence comes from.
  APIRunner  calls a hosted frontier model, used only to draw the reference
             curve. Its parameter count is unknown, so it never enters the
             regression on log10(params).

Keeping the interface this thin matters: the same prompt string must reach
every model through the same chat template path, otherwise prompt formatting
becomes a confound with scale.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol, Sequence

DTYPES = {"bfloat16": "bfloat16", "float16": "float16", "float32": "float32"}


class Runner(Protocol):
    name: str

    def generate(self, prompts: Sequence[str], max_new_tokens: int) -> list[str]:
        ...


@dataclass
class HFRunner:
    """Local Hugging Face causal LM, greedy decoding by default."""

    model_id: str
    dtype: str = "bfloat16"
    device_map: str = "auto"
    temperature: float = 0.0
    batch_size: int = 4
    trust_remote_code: bool = False

    def __post_init__(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self.dtype not in DTYPES:
            raise ValueError(f"dtype must be one of {sorted(DTYPES)}, got '{self.dtype}'")

        self.name = self.model_id
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=self.trust_remote_code
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=getattr(torch, DTYPES[self.dtype]),
            device_map=self.device_map,
            trust_remote_code=self.trust_remote_code,
        )
        self.model.eval()

    def _wrap(self, prompt: str) -> str:
        """Apply the model's own chat template, which is part of its recipe."""
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        return prompt

    def generate(self, prompts: Sequence[str], max_new_tokens: int = 96) -> list[str]:
        torch = self._torch
        outputs: list[str] = []
        for start in range(0, len(prompts), self.batch_size):
            batch = [self._wrap(p) for p in prompts[start : start + self.batch_size]]
            encoded = self.tokenizer(batch, return_tensors="pt", padding=True).to(
                self.model.device
            )
            with torch.no_grad():
                generated = self.model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=self.temperature > 0.0,
                    temperature=self.temperature if self.temperature > 0.0 else None,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            prompt_len = encoded["input_ids"].shape[1]
            for row in generated[:, prompt_len:]:
                outputs.append(self.tokenizer.decode(row, skip_special_tokens=True).strip())
        return outputs


@dataclass
class APIRunner:
    """Hosted reference model. Used for the frontier curve only."""

    model_id: str
    provider: str = "anthropic"
    temperature: float = 0.0

    def __post_init__(self) -> None:
        self.name = f"{self.provider}:{self.model_id}"
        if self.provider == "anthropic":
            from anthropic import Anthropic

            self._client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        elif self.provider == "openai":
            from openai import OpenAI

            self._client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        else:
            raise ValueError(f"Unsupported provider '{self.provider}'")

    def generate(self, prompts: Sequence[str], max_new_tokens: int = 96) -> list[str]:
        outputs: list[str] = []
        for prompt in prompts:
            if self.provider == "anthropic":
                response = self._client.messages.create(
                    model=self.model_id,
                    max_tokens=max_new_tokens,
                    temperature=self.temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
                outputs.append("".join(block.text for block in response.content).strip())
            else:
                response = self._client.chat.completions.create(
                    model=self.model_id,
                    max_completion_tokens=max_new_tokens,
                    temperature=self.temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
                outputs.append((response.choices[0].message.content or "").strip())
        return outputs


def build_runner(spec, generation) -> Runner:
    """Dispatch on the model id: 'api:provider:model' goes to APIRunner."""
    if spec.id.startswith("api:"):
        _, provider, model_id = spec.id.split(":", 2)
        return APIRunner(model_id=model_id, provider=provider, temperature=generation.temperature)
    return HFRunner(
        model_id=spec.id,
        dtype=generation.dtype,
        device_map=generation.device_map,
        temperature=generation.temperature,
        batch_size=generation.batch_size,
        trust_remote_code=generation.trust_remote_code,
    )


def load_tokenizer(model_id: str, trust_remote_code: bool = False):
    """Tokenizer only, for building prompts without paying for model weights."""
    from transformers import AutoTokenizer

    if model_id.startswith("api:"):
        # Reference models do not expose a tokenizer; use a public stand-in so
        # that target lengths stay comparable, and record that in the results.
        model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    return AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
