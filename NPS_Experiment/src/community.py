"""Agent community and propagation-network construction."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

from .agent_factory import build_agents as factory_build_agents
from .agent_roles import BaseAgent


Network = dict[str, list[str]]


def build_agents(config: dict[str, Any]) -> list[BaseAgent]:
    """Thin wrapper used by the run pipeline."""
    return factory_build_agents(config)


def _add_edge(edges: dict[str, set[str]], a: str, b: str) -> None:
    if a == b:
        return
    edges[a].add(b)
    edges[b].add(a)


def _finalize(edges: dict[str, set[str]], agent_ids: list[str]) -> Network:
    return {agent_id: sorted(edges.get(agent_id, set())) for agent_id in sorted(agent_ids)}


def build_random_network(agents: list[BaseAgent], degree: int = 4, seed: int = 42) -> Network:
    """Build an undirected random network with approximate fixed degree."""
    rng = random.Random(seed)
    ids = [agent.agent_id for agent in agents]
    edges: dict[str, set[str]] = defaultdict(set)
    for agent_id in ids:
        candidates = [other for other in ids if other != agent_id]
        rng.shuffle(candidates)
        for other in candidates[:degree]:
            _add_edge(edges, agent_id, other)
    return _finalize(edges, ids)


def build_small_world_network(agents: list[BaseAgent], k: int = 4, rewire_prob: float = 0.2, seed: int = 42) -> Network:
    """Build a simple Watts-Strogatz-style undirected network."""
    rng = random.Random(seed)
    ids = [agent.agent_id for agent in agents]
    n = len(ids)
    edges: dict[str, set[str]] = defaultdict(set)
    half_k = max(1, k // 2)

    for i, agent_id in enumerate(ids):
        for offset in range(1, half_k + 1):
            _add_edge(edges, agent_id, ids[(i + offset) % n])

    for i, agent_id in enumerate(ids):
        for neighbor in list(edges[agent_id]):
            if rng.random() < rewire_prob:
                edges[agent_id].discard(neighbor)
                edges[neighbor].discard(agent_id)
                candidates = [candidate for candidate in ids if candidate != agent_id and candidate not in edges[agent_id]]
                if candidates:
                    _add_edge(edges, agent_id, rng.choice(candidates))

    return _finalize(edges, ids)


def build_scale_free_network(agents: list[BaseAgent], m: int = 3, seed: int = 42) -> Network:
    """Build a Barabasi-Albert-style network with preferential attachment."""
    rng = random.Random(seed)
    ids = [agent.agent_id for agent in agents]
    if len(ids) <= 1:
        return _finalize(defaultdict(set), ids)

    edges: dict[str, set[str]] = defaultdict(set)
    initial_size = min(max(2, m + 1), len(ids))
    initial_nodes = ids[:initial_size]
    for i, agent_id in enumerate(initial_nodes):
        for other in initial_nodes[i + 1 :]:
            _add_edge(edges, agent_id, other)

    repeated_nodes: list[str] = []
    for agent_id in initial_nodes:
        repeated_nodes.extend([agent_id] * max(1, len(edges[agent_id])))

    for agent_id in ids[initial_size:]:
        targets: set[str] = set()
        while len(targets) < min(m, len([other for other in ids if other != agent_id])):
            if repeated_nodes:
                targets.add(rng.choice(repeated_nodes))
            else:
                targets.add(rng.choice([other for other in ids if other != agent_id]))
        for target in targets:
            _add_edge(edges, agent_id, target)
        repeated_nodes.extend([agent_id] * max(1, len(edges[agent_id])))
        for target in targets:
            repeated_nodes.append(target)

    return _finalize(edges, ids)


def _ideological_cluster(agent: BaseAgent) -> str:
    ideology = agent.ideology or ""
    if "right" in ideology or "conservative" in ideology or "security" in ideology:
        return "right"
    if "left" in ideology or "progressive" in ideology or "democracy" in ideology:
        return "left"
    if agent.role == "institutional_verifier":
        return "bridge"
    return "center"


def build_polarized_clusters_network(agents: list[BaseAgent], seed: int = 42) -> Network:
    """Build echo-chamber-like clusters with verifier bridge nodes."""
    rng = random.Random(seed)
    ids = [agent.agent_id for agent in agents]
    by_id = {agent.agent_id: agent for agent in agents}
    clusters: dict[str, list[str]] = defaultdict(list)
    for agent in agents:
        clusters[_ideological_cluster(agent)].append(agent.agent_id)

    edges: dict[str, set[str]] = defaultdict(set)

    for cluster_name in ("left", "right", "center"):
        cluster = clusters.get(cluster_name, [])
        for i, agent_id in enumerate(cluster):
            for other in cluster[i + 1 :]:
                if rng.random() < 0.75:
                    _add_edge(edges, agent_id, other)

    bridge_nodes = clusters.get("bridge", [])
    non_bridge = [agent_id for agent_id in ids if agent_id not in bridge_nodes]
    for verifier in bridge_nodes:
        rng.shuffle(non_bridge)
        for other in non_bridge[: min(6, len(non_bridge))]:
            _add_edge(edges, verifier, other)

    left = clusters.get("left", [])
    right = clusters.get("right", [])
    center = clusters.get("center", [])
    for agent_id in left:
        if right and rng.random() < 0.25:
            _add_edge(edges, agent_id, rng.choice(right))
    for agent_id in right:
        if left and rng.random() < 0.25:
            _add_edge(edges, agent_id, rng.choice(left))
    for agent_id in center:
        targets = left + right + bridge_nodes
        rng.shuffle(targets)
        for other in targets[: min(4, len(targets))]:
            _add_edge(edges, agent_id, other)

    # Ensure every node has at least two neighbors for propagation.
    for agent_id in ids:
        while len(edges[agent_id]) < min(2, len(ids) - 1):
            candidate = rng.choice([other for other in ids if other != agent_id])
            _add_edge(edges, agent_id, candidate)

    return _finalize(edges, ids)


def build_network(agents: list[BaseAgent], topology: str = "small_world", seed: int = 42) -> Network:
    """Build one of the supported propagation networks."""
    if topology == "random":
        return build_random_network(agents, seed=seed)
    if topology in {"small_world", "high_clustering"}:
        return build_small_world_network(agents, seed=seed)
    if topology in {"scale_free", "scale-free"}:
        return build_scale_free_network(agents, seed=seed)
    if topology == "polarized_clusters":
        return build_polarized_clusters_network(agents, seed=seed)
    raise ValueError(f"Unsupported topology: {topology}")


def get_neighbors(agent_id: str, network: Network) -> list[str]:
    return network.get(agent_id, [])


def select_active_agents(
    round_id: int,
    agents: list[BaseAgent],
    network: Network,
    strategy: str = "neighbor_propagation",
) -> list[BaseAgent]:
    """Return active agents for a round. Current default activates all non-isolated agents."""
    if strategy != "neighbor_propagation":
        raise ValueError(f"Unsupported activation strategy: {strategy}")
    return [agent for agent in agents if network.get(agent.agent_id)]


def network_summary(network: Network, agents: list[BaseAgent]) -> dict[str, Any]:
    by_id = {agent.agent_id: agent for agent in agents}
    degrees = {agent_id: len(neighbors) for agent_id, neighbors in network.items()}
    role_degree: dict[str, list[int]] = defaultdict(list)
    for agent_id, degree in degrees.items():
        role_degree[by_id[agent_id].role].append(degree)
    return {
        "num_agents": len(network),
        "num_edges": sum(degrees.values()) // 2,
        "min_degree": min(degrees.values()) if degrees else 0,
        "max_degree": max(degrees.values()) if degrees else 0,
        "avg_degree": round(sum(degrees.values()) / len(degrees), 3) if degrees else 0,
        "avg_degree_by_role": {
            role: round(sum(values) / len(values), 3) for role, values in sorted(role_degree.items())
        },
    }
