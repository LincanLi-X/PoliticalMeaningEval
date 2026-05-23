"""Model clients for PPS runs.

Only the mock client is used by local tests. API-backed clients import optional
packages lazily and read credentials from environment variables.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

from .schema import CONDITION_IA, LABEL_FALSE, LABEL_HALF_TRUE, LABEL_TRUE, PPSSample


@dataclass
class GenerationConfig:
    model: str
    temperature: float = 0.0
    max_tokens: int = 700


class BaseModelClient:
    provider = "base"

    def generate(self, prompt: str, sample: Optional[PPSSample] = None, condition: str = "") -> str:
        raise NotImplementedError


class MockModelClient(BaseModelClient):
    """Deterministic fake model used for dry runs and tests."""

    provider = "mock"

    def __init__(self, model: str = "mock-pps"):
        self.config = GenerationConfig(model=model)

    def generate(self, prompt: str, sample: Optional[PPSSample] = None, condition: str = "") -> str:
        if sample is None:
            final_label = LABEL_FALSE
            intent = ""
            risk = "unclear"
            support = "unclear"
        elif sample.label == LABEL_HALF_TRUE:
            final_label = LABEL_TRUE if condition == "LE" else LABEL_HALF_TRUE
            intent = sample.inferred_intent_gold if condition == CONDITION_IA else "surface reading of the claim"
            risk = "low" if condition == "LE" else "high"
            support = "supported"
        elif sample.label == LABEL_TRUE:
            final_label = LABEL_TRUE
            intent = sample.inferred_intent_gold if condition == CONDITION_IA else "the claim is supported"
            risk = "low"
            support = "supported"
        else:
            final_label = LABEL_FALSE
            intent = sample.inferred_intent_gold if condition == CONDITION_IA else "the claim is contradicted"
            risk = "medium"
            support = "unsupported"

        return json.dumps(
            {
                "literal_support": support,
                "missing_context_risk": risk,
                "inferred_intent": intent,
                "final_label": final_label,
                "short_explanation": f"Mock {condition} judgment for dry-run testing.",
                "intent_recovered": bool(sample and condition == CONDITION_IA),
            }
        )


class OpenAICompatibleClient(BaseModelClient):
    provider = "openai"

    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 700):
        from openai import OpenAI

        api_key = os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("OPENAI_BASE_URL")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY must be set for provider=openai")
        self.client = OpenAI(api_key=api_key, base_url=base_url or None)
        self.config = GenerationConfig(model=model, temperature=temperature, max_tokens=max_tokens)

    def generate(self, prompt: str, sample: Optional[PPSSample] = None, condition: str = "") -> str:
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        return response.choices[0].message.content or ""


class AnthropicClient(BaseModelClient):
    provider = "anthropic"

    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 700):
        import anthropic

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY must be set for provider=anthropic")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.config = GenerationConfig(model=model, temperature=temperature, max_tokens=max_tokens)

    def generate(self, prompt: str, sample: Optional[PPSSample] = None, condition: str = "") -> str:
        response = self.client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(getattr(block, "text", "") for block in response.content)


class GeminiClient(BaseModelClient):
    provider = "gemini"

    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 700):
        import google.generativeai as genai

        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError("GOOGLE_API_KEY must be set for provider=gemini")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
        self.config = GenerationConfig(model=model, temperature=temperature, max_tokens=max_tokens)

    def generate(self, prompt: str, sample: Optional[PPSSample] = None, condition: str = "") -> str:
        response = self.model.generate_content(
            prompt,
            generation_config={
                "temperature": self.config.temperature,
                "max_output_tokens": self.config.max_tokens,
            },
        )
        return getattr(response, "text", "") or ""


class LocalHFClient(BaseModelClient):
    provider = "local_hf"

    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 700):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else "auto",
            device_map="auto",
        )
        self.config = GenerationConfig(model=model, temperature=temperature, max_tokens=max_tokens)

    def generate(self, prompt: str, sample: Optional[PPSSample] = None, condition: str = "") -> str:
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        output_ids = self.model.generate(
            **inputs,
            do_sample=self.config.temperature > 0,
            temperature=max(self.config.temperature, 1e-6),
            max_new_tokens=self.config.max_tokens,
        )
        generated = output_ids[:, inputs.input_ids.shape[-1] :]
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0]


def build_client(
    provider: str,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 700,
) -> BaseModelClient:
    provider = provider.lower()
    if provider == "mock":
        return MockModelClient(model=model)
    if provider in {"openai", "openai_compatible"}:
        return OpenAICompatibleClient(model=model, temperature=temperature, max_tokens=max_tokens)
    if provider == "anthropic":
        return AnthropicClient(model=model, temperature=temperature, max_tokens=max_tokens)
    if provider == "gemini":
        return GeminiClient(model=model, temperature=temperature, max_tokens=max_tokens)
    if provider in {"local_hf", "hf", "transformers"}:
        return LocalHFClient(model=model, temperature=temperature, max_tokens=max_tokens)
    raise ValueError(f"unsupported provider: {provider}")
