"""Single-seed and batch trajectory simulation for NPS."""

from __future__ import annotations

import json
import random
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .agent_roles import BaseAgent
from .community import Network, get_neighbors, select_active_agents


@dataclass
class Message:
    message_id: str
    seed_id: str
    round_id: int
    agent_id: str
    agent_role: str
    ideology: str | None
    parent_message_id: str | None
    text: str
    timestamp: str


@dataclass
class SimulationTrajectory:
    seed_id: str
    domain: str
    original_report: str
    supporting_evidence: str
    known_distorted_variant: str
    config_snapshot: dict[str, Any]
    network: Network
    messages: list[Message] = field(default_factory=list)
    final_messages: list[Message] = field(default_factory=list)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def message_to_dict(message: Message) -> dict[str, Any]:
    return asdict(message)


def trajectory_to_dict(trajectory: SimulationTrajectory) -> dict[str, Any]:
    data = asdict(trajectory)
    data["messages"] = [message_to_dict(message) for message in trajectory.messages]
    data["final_messages"] = [message_to_dict(message) for message in trajectory.final_messages]
    return data


def initialize_seed_message(seed: dict[str, Any]) -> Message:
    """Create the round-0 anchor message from the verified seed."""
    text = (
        f"VERIFIED REPORT: {seed['original_verified_report']}\n"
        f"SUPPORTING EVIDENCE: {seed['supporting_fact_check_evidence']}\n"
        f"KNOWN DISTORTED VARIANT: {seed['real_world_distorted_or_reframed_variant']}"
    )
    return Message(
        message_id=f"{seed['seed_id']}_round0_seed",
        seed_id=seed["seed_id"],
        round_id=0,
        agent_id="seed_source",
        agent_role="verified_seed",
        ideology=None,
        parent_message_id=None,
        text=text,
        timestamp=utc_now_iso(),
    )


def _messages_by_agent(messages: list[Message]) -> dict[str, list[Message]]:
    grouped: dict[str, list[Message]] = {}
    for message in messages:
        grouped.setdefault(message.agent_id, []).append(message)
    return grouped


def _select_initial_agents(
    agents: list[BaseAgent],
    config: dict[str, Any],
    rng: random.Random,
) -> list[BaseAgent]:
    sim_config = config.get("simulation", {})
    roles = set(sim_config.get("initial_receiver_roles", ["ordinary_observer", "political_commentator"]))
    per_role = int(sim_config.get("initial_receivers_per_role", 2))
    selected: list[BaseAgent] = []
    for role in sorted(roles):
        role_agents = [agent for agent in agents if agent.role == role]
        rng.shuffle(role_agents)
        selected.extend(role_agents[:per_role])
    return selected


def _incoming_for_agent(
    agent: BaseAgent,
    previous_round_messages: list[Message],
    network: Network,
    max_context_messages: int,
) -> list[dict[str, Any]]:
    neighbor_ids = set(get_neighbors(agent.agent_id, network))
    candidates = [
        message
        for message in previous_round_messages
        if message.agent_id in neighbor_ids or message.agent_role == "verified_seed"
    ]
    if not candidates:
        candidates = previous_round_messages
    candidates = candidates[-max_context_messages:]
    return [message_to_dict(message) for message in candidates]


def _temperature_for(agent: BaseAgent, config: dict[str, Any]) -> float:
    temperatures = config.get("models", {}).get("temperature", {})
    return float(temperatures.get(agent.role, 0.7))


def _max_generation_tokens(config: dict[str, Any]) -> int:
    return int(config.get("models", {}).get("max_output_tokens", {}).get("generation", 240))


def _max_concurrent_agent_calls(config: dict[str, Any]) -> int:
    return max(1, int(config.get("simulation", {}).get("max_concurrent_agent_calls", 1)))


def _make_message(
    seed_id: str,
    round_id: int,
    agent: BaseAgent,
    parent_message_id: str | None,
    text: str,
) -> Message:
    return Message(
        message_id=f"{seed_id}_r{round_id}_{agent.agent_id}_{uuid.uuid4().hex[:8]}",
        seed_id=seed_id,
        round_id=round_id,
        agent_id=agent.agent_id,
        agent_role=agent.role,
        ideology=agent.ideology,
        parent_message_id=parent_message_id,
        text=text,
        timestamp=utc_now_iso(),
    )


def run_round(
    round_id: int,
    previous_round_messages: list[Message],
    agents: list[BaseAgent],
    network: Network,
    seed: dict[str, Any],
    config: dict[str, Any],
    llm_client: Any,
) -> list[Message]:
    """Run one propagation round."""
    max_context = int(config.get("simulation", {}).get("max_context_messages", 4))
    active_agents = select_active_agents(round_id, agents, network)

    if round_id == 1:
        stable_seed_offset = sum(ord(ch) for ch in seed["seed_id"])
        rng = random.Random(int(config.get("dataset", {}).get("random_seed", 42)) + stable_seed_offset)
        active_agents = _select_initial_agents(active_agents, config, rng)

    agent_progress = bool(config.get("logging", {}).get("agent_progress", False))
    max_workers = _max_concurrent_agent_calls(config)

    def call_agent(agent: BaseAgent) -> Message | None:
        if agent_progress:
            print(f"  seed={seed['seed_id']} round={round_id} agent={agent.agent_id}", flush=True)
        incoming = _incoming_for_agent(agent, previous_round_messages, network, max_context)
        if not incoming:
            return None
        parent_message_id = incoming[-1]["message_id"]
        text = agent.respond(
            incoming_messages=incoming,
            seed=seed,
            llm_client=llm_client,
            temperature=_temperature_for(agent, config),
            max_tokens=_max_generation_tokens(config),
        )
        message = _make_message(seed["seed_id"], round_id, agent, parent_message_id, text)
        agent.observe(message_to_dict(message))
        return message

    new_messages: list[Message] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(call_agent, agent): agent for agent in active_agents}
        for future in as_completed(futures):
            message = future.result()
            if message is not None:
                new_messages.append(message)
    new_messages.sort(key=lambda message: message.agent_id)
    return new_messages


def apply_verifier_intervention(
    round_id: int,
    previous_round_messages: list[Message],
    verifiers: list[BaseAgent],
    seed: dict[str, Any],
    config: dict[str, Any],
    llm_client: Any,
) -> list[Message]:
    """Run extra verifier notes at configured intervention rounds."""
    if not verifiers:
        return []
    max_context = int(config.get("simulation", {}).get("max_context_messages", 4))
    agent_progress = bool(config.get("logging", {}).get("agent_progress", False))
    max_workers = _max_concurrent_agent_calls(config)

    def call_verifier(verifier: BaseAgent) -> Message:
        if agent_progress:
            print(f"  seed={seed['seed_id']} round={round_id} verifier={verifier.agent_id}", flush=True)
        incoming = [message_to_dict(message) for message in previous_round_messages[-max_context:]]
        text = verifier.respond(
            incoming_messages=incoming,
            seed=seed,
            llm_client=llm_client,
            temperature=_temperature_for(verifier, config),
            max_tokens=_max_generation_tokens(config),
        )
        parent_message_id = incoming[-1]["message_id"] if incoming else None
        message = _make_message(seed["seed_id"], round_id, verifier, parent_message_id, text)
        verifier.observe(message_to_dict(message))
        return message

    intervention_messages: list[Message] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(call_verifier, verifier) for verifier in verifiers]
        for future in as_completed(futures):
            intervention_messages.append(future.result())
    intervention_messages.sort(key=lambda message: message.agent_id)
    return intervention_messages


def run_single_simulation(
    seed: dict[str, Any],
    agents: list[BaseAgent],
    network: Network,
    config: dict[str, Any],
    llm_client: Any,
) -> SimulationTrajectory:
    """Run a complete propagation trajectory for one verified seed."""
    num_rounds = int(config.get("simulation", {}).get("num_rounds", 10))
    verifier_rounds = set(config.get("simulation", {}).get("verifier_intervention_rounds", [3, 6, 9]))
    verifiers = [agent for agent in agents if agent.role == "institutional_verifier"]

    seed_message = initialize_seed_message(seed)
    trajectory = SimulationTrajectory(
        seed_id=seed["seed_id"],
        domain=seed["domain"],
        original_report=seed["original_verified_report"],
        supporting_evidence=seed["supporting_fact_check_evidence"],
        known_distorted_variant=seed["real_world_distorted_or_reframed_variant"],
        config_snapshot={
            "community": config.get("community", {}),
            "simulation": config.get("simulation", {}),
            "models": {
                key: value
                for key, value in config.get("models", {}).items()
                if key not in {"api_key", "organization"}
            },
        },
        network=network,
        messages=[seed_message],
    )

    previous_round_messages = [seed_message]
    for round_id in range(1, num_rounds + 1):
        new_messages = run_round(
            round_id=round_id,
            previous_round_messages=previous_round_messages,
            agents=agents,
            network=network,
            seed=seed,
            config=config,
            llm_client=llm_client,
        )
        if round_id in verifier_rounds:
            new_messages.extend(
                apply_verifier_intervention(
                    round_id=round_id,
                    previous_round_messages=new_messages or previous_round_messages,
                    verifiers=verifiers,
                    seed=seed,
                    config=config,
                    llm_client=llm_client,
                )
            )
        trajectory.messages.extend(new_messages)
        previous_round_messages = new_messages or previous_round_messages

    trajectory.final_messages = [message for message in trajectory.messages if message.round_id == num_rounds]
    return trajectory


def save_trajectory(trajectory: SimulationTrajectory, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(trajectory_to_dict(trajectory), ensure_ascii=False, indent=2), encoding="utf-8")


def load_trajectory(path: str | Path) -> SimulationTrajectory:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    messages = [Message(**message) for message in data.pop("messages")]
    final_messages = [Message(**message) for message in data.pop("final_messages")]
    return SimulationTrajectory(**data, messages=messages, final_messages=final_messages)
