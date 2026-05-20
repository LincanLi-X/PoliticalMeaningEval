"""LLM client abstraction with OpenAI live mode and deterministic mock mode."""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any


def load_dotenv(path: str | Path = ".env") -> None:
    """Load simple KEY=VALUE lines without overriding existing environment variables."""
    path = Path(path)
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


class LLMClient:
    """Generate text for agents and evaluators."""

    def __init__(self, config: dict[str, Any], purpose: str = "generation"):
        load_dotenv()
        models = config.get("models", {})
        self.mode = models.get("mode", "mock")
        self.purpose = purpose
        self.model = models.get("generation_model" if purpose == "generation" else "evaluation_model", "gpt-5-mini")
        max_output_tokens = models.get("max_output_tokens", {})
        default_key = "generation" if purpose == "generation" else "evaluation"
        self.default_max_tokens = int(max_output_tokens.get(default_key, 300))
        self._client = None
        if self.mode == "live":
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError("The openai package is required for live mode.") from exc
            if not os.getenv("OPENAI_API_KEY"):
                raise RuntimeError("OPENAI_API_KEY is not available in the environment or .env file.")
            timeout = float(models.get("request_timeout_seconds", 60))
            max_retries = int(models.get("max_retries", 2))
            self._client = OpenAI(timeout=timeout, max_retries=max_retries)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        role: str | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        max_tokens = self.default_max_tokens if max_tokens is None else max_tokens
        if self.mode == "mock":
            return self._mock_generate(system_prompt, user_prompt, role=role)
        if self.mode != "live":
            raise ValueError(f"Unsupported LLM mode: {self.mode}")
        return self._openai_generate(system_prompt, user_prompt, temperature, max_tokens, response_format)

    def _openai_generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        assert self._client is not None
        request = {
            "model": self.model,
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_output_tokens": max_tokens,
        }
        if response_format is not None:
            request["text"] = {"format": response_format}
        # Some GPT-5 family snapshots reject temperature; use the model default.
        if not self.model.startswith("gpt-5"):
            request["temperature"] = temperature
        response = self._client.responses.create(**request)
        text = getattr(response, "output_text", None)
        if text:
            return text.strip()
        chunks: list[str] = []
        for item in getattr(response, "output", []) or []:
            for content in getattr(item, "content", []) or []:
                value = getattr(content, "text", None)
                if value:
                    chunks.append(value)
        return "\n".join(chunks).strip()

    def _mock_generate(self, system_prompt: str, user_prompt: str, role: str | None = None) -> str:
        """Deterministic synthetic outputs for offline smoke tests."""
        digest = hashlib.sha1((system_prompt + user_prompt + str(role)).encode("utf-8")).hexdigest()[:8]
        original_match = re.search(r"Original verified report:\s*(.+)", user_prompt)
        report = original_match.group(1).strip() if original_match else "the reported event"
        report = report[:220].rstrip()

        if role == "partisan_amplifier":
            return (
                f"This is not just an isolated update: {report} The bigger issue is who benefits when institutions "
                f"avoid accountability. [{digest}]"
            )
        if role == "political_commentator":
            return (
                f"The key political lesson is that this event sits inside a broader fight over trust, authority, "
                f"and public accountability. The verified core remains: {report} [{digest}]"
            )
        if role == "institutional_verifier":
            return (
                f"Verification note: the circulating interpretation should stay anchored to the verified report: "
                f"{report} Avoid adding motives, numbers, or blame not supported by the evidence. [{digest}]"
            )
        if role == "ordinary_observer":
            return (
                f"I am trying to make sense of this, but the basic report is worrying: {report} It feels like people "
                f"will argue over what it says about the system. [{digest}]"
            )
        return (
            '{"factual_deviation":{"score":0.25,"label":"low","rationale":"Mock evaluator output.",'
            '"evidence_span":""},"ideological_drift":{"score":0.35,"label":"medium","direction":"mixed",'
            '"rationale":"Mock evaluator output.","evidence_span":""},"blame_reassignment":{"score":0.3,'
            '"label":"low","target_shift":"limited","rationale":"Mock evaluator output.","evidence_span":""},'
            '"agenda_shift":{"score":0.3,"label":"low","new_agenda":"limited","rationale":"Mock evaluator output.",'
            '"evidence_span":""},"polarization_intensity":{"score":0.4,"label":"medium",'
            '"rationale":"Mock evaluator output.","evidence_span":""}}'
        )
