"""Agent role definitions for the NPS multi-agent news propagation experiment."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROLE_PROMPT_FILES = {
    "partisan_amplifier": "partisan_amplifier.txt",
    "political_commentator": "political_commentator.txt",
    "institutional_verifier": "institutional_verifier.txt",
    "ordinary_observer": "ordinary_observer.txt",
}


ROLE_DISPLAY_NAMES = {
    "partisan_amplifier": "Partisan Amplifier",
    "political_commentator": "Political Commentator",
    "institutional_verifier": "Institutional Verifier",
    "ordinary_observer": "Ordinary Observer",
}


@dataclass
class BaseAgent:
    """Base class for all NPS community agents."""

    agent_id: str
    role: str
    ideology: str | None = None
    style: str | None = None
    memory_size: int = 5
    prompt_template: str = ""
    memory: list[dict[str, Any]] = field(default_factory=list)

    @property
    def display_role(self) -> str:
        return ROLE_DISPLAY_NAMES.get(self.role, self.role)

    def observe(self, message: dict[str, Any], context: dict[str, Any] | None = None) -> None:
        """Store a message in bounded memory."""
        self.update_memory(message)

    def update_memory(self, message: dict[str, Any]) -> None:
        self.memory.append(
            {
                "round_id": message.get("round_id"),
                "agent_id": message.get("agent_id"),
                "agent_role": message.get("agent_role"),
                "text": message.get("text", ""),
            }
        )
        if len(self.memory) > self.memory_size:
            self.memory = self.memory[-self.memory_size :]

    def build_system_prompt(self) -> str:
        identity = [
            self.prompt_template.strip(),
            f"Agent ID: {self.agent_id}",
            f"Role: {self.display_role}",
        ]
        if self.ideology:
            identity.append(f"Ideological orientation/style: {self.ideology}")
        if self.style:
            identity.append(f"Communication style: {self.style}")
        return "\n\n".join(identity)

    def build_user_prompt(self, incoming_messages: list[dict[str, Any]], seed: dict[str, Any]) -> str:
        incoming = "\n".join(
            f"- Round {m.get('round_id')} | {m.get('agent_role')} {m.get('agent_id')}: {m.get('text')}"
            for m in incoming_messages
        )
        memory_text = "\n".join(
            f"- Round {m.get('round_id')} | {m.get('agent_role')} {m.get('agent_id')}: {m.get('text')}"
            for m in self.memory[-self.memory_size :]
        )
        if not memory_text:
            memory_text = "- No prior memory."

        return (
            "Verified seed anchor:\n"
            f"Domain: {seed.get('domain')}\n"
            f"Original verified report: {seed.get('original_verified_report')}\n"
            f"Supporting fact-check evidence: {seed.get('supporting_fact_check_evidence')}\n"
            f"Known distorted/reframed variant: {seed.get('real_world_distorted_or_reframed_variant')}\n\n"
            "Incoming messages from neighbors:\n"
            f"{incoming or '- No incoming messages.'}\n\n"
            "Your recent memory:\n"
            f"{memory_text}\n\n"
            "Now produce your message for the next propagation step."
        )

    def respond(
        self,
        incoming_messages: list[dict[str, Any]],
        seed: dict[str, Any],
        llm_client: Any,
        temperature: float = 0.7,
        max_tokens: int = 240,
    ) -> str:
        """Generate one role-conditioned message."""
        system_prompt = self.build_system_prompt()
        user_prompt = self.build_user_prompt(incoming_messages, seed)
        return llm_client.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            role=self.role,
        ).strip()


class PartisanAmplifier(BaseAgent):
    pass


class PoliticalCommentator(BaseAgent):
    pass


class InstitutionalVerifier(BaseAgent):
    def build_user_prompt(self, incoming_messages: list[dict[str, Any]], seed: dict[str, Any]) -> str:
        base = super().build_user_prompt(incoming_messages, seed)
        return (
            base
            + "\n\nVerification task:\n"
            "1. State whether the incoming message preserves the verified political meaning.\n"
            "2. Identify any unsupported exaggeration, blame shift, or omitted context.\n"
            "3. Provide a corrected version anchored in the verified report."
        )


class OrdinaryObserver(BaseAgent):
    pass


AGENT_CLASSES = {
    "partisan_amplifier": PartisanAmplifier,
    "political_commentator": PoliticalCommentator,
    "institutional_verifier": InstitutionalVerifier,
    "ordinary_observer": OrdinaryObserver,
}


def load_prompt(role: str, prompts_dir: str | Path) -> str:
    prompts_dir = Path(prompts_dir)
    filename = ROLE_PROMPT_FILES[role]
    return (prompts_dir / filename).read_text(encoding="utf-8")

