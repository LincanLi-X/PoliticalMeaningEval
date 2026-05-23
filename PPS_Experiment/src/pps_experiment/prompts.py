"""Prompt rendering for PPS evidence conditions."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from .schema import CONDITION_HE, CONDITION_IA, CONDITION_LE, CONDITIONS, PPSSample


TEMPLATE_FILES: Dict[str, str] = {
    CONDITION_LE: "le.txt",
    CONDITION_HE: "he.txt",
    CONDITION_IA: "ia.txt",
}


def format_evidence(sentences: list[str], empty_text: str = "None provided.") -> str:
    if not sentences:
        return empty_text
    return "\n".join(f"{index + 1}. {sentence}" for index, sentence in enumerate(sentences))


class PromptRenderer:
    """Load and render prompt templates from text files."""

    def __init__(self, prompt_dir: Path):
        self.prompt_dir = prompt_dir
        self.templates = {
            condition: (prompt_dir / filename).read_text(encoding="utf-8")
            for condition, filename in TEMPLATE_FILES.items()
        }

    def render(self, sample: PPSSample, condition: str) -> str:
        if condition not in CONDITIONS:
            raise ValueError(f"unknown condition: {condition}")
        template = self.templates[condition]
        replacements = {
            "{{sample_id}}": sample.sample_id,
            "{{claim}}": sample.claim,
            "{{presented_evidence}}": format_evidence(sample.presented_evidence),
            "{{hidden_evidence}}": format_evidence(sample.hidden_evidence),
            "{{domain}}": sample.domain,
        }
        prompt = template
        for token, value in replacements.items():
            prompt = prompt.replace(token, value)
        return prompt
