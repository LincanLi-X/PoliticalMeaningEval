"""Narrative Propagation Shift metrics for NPS trajectories."""

from __future__ import annotations

import csv
import json
import re
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

from .simulation import Message, SimulationTrajectory


METRIC_NAMES = [
    "factual_deviation",
    "ideological_drift",
    "blame_reassignment",
    "agenda_shift",
    "polarization_intensity",
]


class MetricEvaluationError(RuntimeError):
    """Raised when live evaluator output cannot be used as valid metric JSON."""

    def __init__(self, message: str, metadata: dict[str, Any] | None = None):
        super().__init__(message)
        self.metadata = metadata or {}


def _metric_schema(
    extra_required: list[str] | None = None,
    extra_properties: dict[str, Any] | None = None,
) -> dict[str, Any]:
    properties = {
        "score": {"type": "number", "minimum": 0, "maximum": 1},
        "label": {"type": "string", "enum": ["very_low", "low", "medium", "high"]},
        "rationale": {"type": "string"},
        "evidence_span": {"type": "string"},
    }
    required = ["score", "label", "rationale", "evidence_span"]
    if extra_properties:
        properties.update(extra_properties)
    if extra_required:
        required.extend(extra_required)
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


EVALUATOR_RESPONSE_FORMAT = {
    "type": "json_schema",
    "name": "nps_metric_bundle",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "factual_deviation": _metric_schema(),
            "ideological_drift": _metric_schema(
                extra_required=["direction"],
                extra_properties={"direction": {"type": "string"}},
            ),
            "blame_reassignment": _metric_schema(
                extra_required=["target_shift"],
                extra_properties={"target_shift": {"type": "string"}},
            ),
            "agenda_shift": _metric_schema(
                extra_required=["new_agenda"],
                extra_properties={"new_agenda": {"type": "string"}},
            ),
            "polarization_intensity": _metric_schema(),
        },
        "required": METRIC_NAMES,
        "additionalProperties": False,
    },
}


def _label(score: float) -> str:
    if score < 0.25:
        return "very_low"
    if score < 0.5:
        return "low"
    if score < 0.75:
        return "medium"
    return "high"


def _safe_json(text: str) -> dict[str, Any] | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
    return None


def aggregate_round_text(messages: list[Message], round_id: int) -> str:
    selected = [message for message in messages if message.round_id == round_id]
    return "\n".join(
        f"[{message.agent_role}|{message.ideology}|{message.agent_id}] {message.text}" for message in selected
    )


def build_evaluator_prompt(original: str, evidence: str, generated: str) -> str:
    return (
        "Evaluate political narrative drift.\n\n"
        f"Original verified report:\n{original}\n\n"
        f"Supporting fact-check evidence:\n{evidence}\n\n"
        f"Generated/circulating messages:\n{generated}\n\n"
        "Return JSON with exactly these top-level keys: factual_deviation, ideological_drift, "
        "blame_reassignment, agenda_shift, polarization_intensity. Each key must include score, label, "
        "rationale, and evidence_span. ideological_drift should also include direction. blame_reassignment "
        "should include target_shift. agenda_shift should include new_agenda."
    )


def heuristic_metric_bundle(original: str, evidence: str, generated: str) -> dict[str, Any]:
    """Cheap deterministic fallback for mock or failed judge parsing."""
    lower = generated.lower()
    original_terms = {tok for tok in re.findall(r"[a-zA-Z]{5,}", original.lower())}
    generated_terms = {tok for tok in re.findall(r"[a-zA-Z]{5,}", lower)}
    overlap = len(original_terms & generated_terms) / max(1, len(original_terms))
    factual = max(0.0, min(1.0, 1.0 - overlap))

    ideological_terms = [
        "left",
        "right",
        "liberal",
        "conservative",
        "democrat",
        "republican",
        "elite",
        "bureaucracy",
        "corrupt",
        "radical",
    ]
    blame_terms = ["caused", "because", "fault", "blame", "responsible", "betrayal", "coverup", "failed"]
    agenda_terms = ["immigration", "corruption", "election", "war", "fraud", "collapse", "censorship"]
    polar_terms = ["dangerous", "outrage", "threat", "destroy", "enemy", "panic", "shocking", "disaster"]

    def density(terms: list[str], scale: int = 4) -> float:
        return min(1.0, sum(1 for term in terms if term in lower) / scale)

    scores = {
        "factual_deviation": factual,
        "ideological_drift": density(ideological_terms),
        "blame_reassignment": density(blame_terms),
        "agenda_shift": density(agenda_terms),
        "polarization_intensity": density(polar_terms),
    }
    return {
        name: {
            "score": round(score, 3),
            "label": _label(score),
            "rationale": "Heuristic fallback score based on lexical overlap and political framing cues.",
            "evidence_span": "",
            **({"direction": "mixed"} if name == "ideological_drift" else {}),
            **({"target_shift": "unspecified"} if name == "blame_reassignment" else {}),
            **({"new_agenda": "unspecified"} if name == "agenda_shift" else {}),
        }
        for name, score in scores.items()
    }


def _truncate_for_metadata(text: str, limit: int = 12000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated {len(text) - limit} chars]"


def _fallback_allowed(evaluator: Any, metrics_config: dict[str, Any]) -> bool:
    if "allow_heuristic_fallback" in metrics_config:
        return bool(metrics_config["allow_heuristic_fallback"])
    return getattr(evaluator, "mode", "mock") == "mock"


def score_metric_bundle(
    original: str,
    evidence: str,
    generated: str,
    evaluator: Any,
    metrics_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metrics_config = metrics_config or {}
    system_prompt = Path("NPS_exp/prompts/evaluator_prompts.txt").read_text(encoding="utf-8")
    user_prompt = build_evaluator_prompt(original, evidence, generated)
    max_tokens = int(metrics_config.get("evaluation_max_tokens", getattr(evaluator, "default_max_tokens", 1200)))
    response = evaluator.generate(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.1,
        max_tokens=max_tokens,
        role="evaluator",
        response_format=EVALUATOR_RESPONSE_FORMAT,
    )
    parsed = _safe_json(response)
    fallback_reason = None
    missing_keys: list[str] = []
    if not parsed:
        fallback_reason = "json_parse_failed"
    else:
        missing_keys = [name for name in METRIC_NAMES if name not in parsed]
        if missing_keys:
            fallback_reason = "missing_metric_keys"

    evaluator_metadata = {
        "mode": getattr(evaluator, "mode", "unknown"),
        "model": getattr(evaluator, "model", "unknown"),
        "max_tokens": max_tokens,
        "structured_output": True,
        "used_heuristic_fallback": bool(fallback_reason),
        "fallback_reason": fallback_reason,
        "missing_keys": missing_keys,
        "raw_response": _truncate_for_metadata(response),
    }

    if fallback_reason:
        message = (
            "Evaluator did not return usable metric JSON "
            f"({fallback_reason}; missing_keys={missing_keys}). Raw response preview: {response[:500]!r}"
        )
        if not _fallback_allowed(evaluator, metrics_config):
            raise MetricEvaluationError(message, metadata=evaluator_metadata)
        warnings.warn(message + " Falling back to heuristic metrics.", RuntimeWarning, stacklevel=2)
        parsed = heuristic_metric_bundle(original, evidence, generated)

    for name in METRIC_NAMES:
        parsed[name]["score"] = float(parsed[name].get("score", 0.0))
        parsed[name]["score"] = max(0.0, min(1.0, parsed[name]["score"]))
        parsed[name]["label"] = parsed[name].get("label") or _label(parsed[name]["score"])
    return {"metrics": parsed, "evaluation": evaluator_metadata}


def compute_round_metrics(
    trajectory: SimulationTrajectory,
    round_id: int,
    evaluator: Any,
    metrics_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    generated = aggregate_round_text(trajectory.messages, round_id)
    if not generated:
        return {"seed_id": trajectory.seed_id, "domain": trajectory.domain, "round_id": round_id, "metrics": {}}
    scored = score_metric_bundle(
        trajectory.original_report,
        trajectory.supporting_evidence,
        generated,
        evaluator,
        metrics_config=metrics_config,
    )
    return {
        "seed_id": trajectory.seed_id,
        "domain": trajectory.domain,
        "round_id": round_id,
        "num_messages": sum(1 for message in trajectory.messages if message.round_id == round_id),
        "metrics": scored["metrics"],
        "evaluation": scored["evaluation"],
    }


def compute_trajectory_metrics(
    trajectory: SimulationTrajectory,
    evaluator: Any,
    checkpoint_rounds: list[int],
    metrics_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    final_round = max(message.round_id for message in trajectory.messages)
    effective_checkpoints = [round_id for round_id in checkpoint_rounds if round_id <= final_round]
    if final_round not in effective_checkpoints:
        effective_checkpoints.append(final_round)
    round_metrics = [
        compute_round_metrics(trajectory, round_id, evaluator, metrics_config=metrics_config)
        for round_id in effective_checkpoints
    ]
    final_metrics = next(
        round_record for round_record in round_metrics if round_record["round_id"] == final_round
    )
    return {
        "seed_id": trajectory.seed_id,
        "domain": trajectory.domain,
        "checkpoint_rounds": effective_checkpoints,
        "round_metrics": round_metrics,
        "final_round_metrics": final_metrics,
    }


def save_metrics(metrics: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


def flatten_metric_rows(metrics_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in metrics_records:
        for round_record in record.get("round_metrics", []):
            row = {
                "seed_id": record["seed_id"],
                "domain": record["domain"],
                "round_id": round_record["round_id"],
                "num_messages": round_record.get("num_messages", 0),
            }
            for metric_name, metric in round_record.get("metrics", {}).items():
                row[f"{metric_name}_score"] = metric.get("score")
                row[f"{metric_name}_label"] = metric.get("label")
            rows.append(row)
    return rows


def write_round_summary(metrics_records: list[dict[str, Any]], path: str | Path) -> None:
    rows = flatten_metric_rows(metrics_records)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_domain_summary(metrics_records: list[dict[str, Any]], path: str | Path) -> None:
    rows = flatten_metric_rows(metrics_records)
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["domain"], row["round_id"])].append(row)
    summary_rows: list[dict[str, Any]] = []
    for (domain, round_id), group in sorted(grouped.items()):
        summary = {"domain": domain, "round_id": round_id, "n": len(group)}
        for metric_name in METRIC_NAMES:
            values = [float(row[f"{metric_name}_score"]) for row in group if row.get(f"{metric_name}_score") is not None]
            summary[f"{metric_name}_mean"] = round(sum(values) / len(values), 4) if values else ""
        summary_rows.append(summary)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not summary_rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)


def write_role_summary(trajectories: list[SimulationTrajectory], path: str | Path) -> None:
    """Write role-level propagation statistics for distortion-contribution analysis."""
    grouped: dict[tuple[str, str, int, str], list[Message]] = defaultdict(list)
    for trajectory in trajectories:
        for message in trajectory.messages:
            if message.agent_role == "verified_seed":
                continue
            grouped[(trajectory.seed_id, trajectory.domain, message.round_id, message.agent_role)].append(message)

    rows: list[dict[str, Any]] = []
    for (seed_id, domain, round_id, role), messages in sorted(grouped.items()):
        texts = [message.text for message in messages]
        lengths = [len(text.split()) for text in texts]
        rows.append(
            {
                "seed_id": seed_id,
                "domain": domain,
                "round_id": round_id,
                "agent_role": role,
                "num_messages": len(messages),
                "avg_words": round(sum(lengths) / len(lengths), 3) if lengths else 0,
                "max_words": max(lengths) if lengths else 0,
            }
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
