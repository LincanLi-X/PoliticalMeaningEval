"""Factory helpers for constructing configured NPS agents."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .agent_roles import AGENT_CLASSES, BaseAgent, load_prompt


ROLE_ORDER = [
    "partisan_amplifier",
    "political_commentator",
    "institutional_verifier",
    "ordinary_observer",
]


IDEOLOGY_PROFILES = {
    "partisan_amplifier": [
        ("left_populist", "high moral urgency; emphasizes corporate power, inequality, and elite betrayal"),
        ("left_institutional", "defends democratic institutions but criticizes right-wing threats"),
        ("right_populist", "emphasizes government overreach, border/security fears, and elite betrayal"),
        ("right_nationalist", "emphasizes sovereignty, law and order, and national decline"),
        ("anti_elite_swing", "distrusts both parties and frames institutions as self-protective"),
        ("left_activist", "emphasizes vulnerable groups, rights, and systemic injustice"),
        ("right_libertarian", "emphasizes bureaucracy, censorship, and individual liberty"),
        ("cultural_conservative", "emphasizes social order and cultural threat"),
        ("progressive_reformer", "emphasizes accountability and policy repair"),
        ("hardline_security", "emphasizes enforcement, punishment, and public threat"),
    ],
    "political_commentator": [
        ("center_left_analyst", "policy-focused, cautious but institutionally skeptical"),
        ("center_right_analyst", "policy-focused, cautious but skeptical of bureaucracy"),
        ("populism_analyst", "interprets events through anti-elite pressure and public backlash"),
        ("democracy_analyst", "interprets events through democratic norms and institutional resilience"),
        ("security_analyst", "interprets events through risk, deterrence, and public order"),
        ("media_analyst", "interprets events through framing, agenda-setting, and attention dynamics"),
        ("legal_analyst", "interprets events through courts, authority, and procedure"),
        ("economic_analyst", "interprets events through distribution, budgets, and incentives"),
        ("foreign_policy_analyst", "interprets events through alliances, sovereignty, and conflict risk"),
        ("public_opinion_analyst", "interprets events through voter perception and trust"),
    ],
    "institutional_verifier": [
        ("fact_checking", "strictly compares claims to evidence and flags unsupported assertions"),
        ("legal_procedural", "emphasizes legal/procedural distinctions and institutional authority"),
        ("data_quality", "emphasizes statistics, denominators, baselines, and uncertainty"),
        ("source_audit", "emphasizes provenance, source reliability, and media context"),
        ("harm_reduction", "corrects misinformation while avoiding unnecessary amplification"),
        ("public_records", "anchors claims in official records and documented procedures"),
        ("scientific_evidence", "anchors claims in technical evidence and expert consensus"),
        ("election_admin", "anchors claims in election law and administrative safeguards"),
        ("budget_audit", "anchors claims in funding streams and accounting categories"),
        ("security_context", "anchors claims in verified threat and law-enforcement evidence"),
    ],
    "ordinary_observer": [
        ("low_information_concerned", "concerned citizen; repeats salient claims cautiously"),
        ("politically_tired", "cynical about politics and institutions"),
        ("community_focused", "focuses on local consequences and social trust"),
        ("headline_scanner", "reacts to the strongest headline frame"),
        ("skeptical_but_uncertain", "asks questions but may accept repeated claims"),
        ("family_safety_focused", "focuses on risks to family and everyday life"),
        ("taxpayer_focused", "focuses on waste, fairness, and who pays"),
        ("rights_focused", "focuses on civil liberties and unequal treatment"),
        ("media_skeptical", "distrusts mainstream and social media in different ways"),
        ("pragmatic_moderate", "prefers practical explanations and avoids extreme claims"),
    ],
}


def build_role_agents(
    role: str,
    count: int,
    memory_size: int,
    prompts_dir: str | Path,
) -> list[BaseAgent]:
    """Build a fixed number of agents for one role."""
    prompt = load_prompt(role, prompts_dir)
    profiles = IDEOLOGY_PROFILES[role]
    cls = AGENT_CLASSES[role]
    agents: list[BaseAgent] = []
    for idx in range(count):
        ideology, style = profiles[idx % len(profiles)]
        agents.append(
            cls(
                agent_id=f"{role}_{idx + 1:02d}",
                role=role,
                ideology=ideology,
                style=style,
                memory_size=memory_size,
                prompt_template=prompt,
            )
        )
    return agents


def build_agents(config: dict[str, Any], prompts_dir: str | Path = "NPS_exp/prompts") -> list[BaseAgent]:
    """Build the full NPS community from config."""
    community_config = config.get("community", {})
    count = int(community_config.get("agents_per_role", 5))
    memory_size = int(community_config.get("memory_size", 5))
    agents: list[BaseAgent] = []
    for role in ROLE_ORDER:
        agents.extend(build_role_agents(role, count, memory_size, prompts_dir))
    return agents

