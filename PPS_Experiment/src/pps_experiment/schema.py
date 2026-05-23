"""Shared schema objects for the PPS experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


LABEL_TRUE = "true"
LABEL_HALF_TRUE = "half-true"
LABEL_FALSE = "false"
LABELS: Tuple[str, str, str] = (LABEL_TRUE, LABEL_HALF_TRUE, LABEL_FALSE)

CONDITION_LE = "LE"
CONDITION_HE = "HE"
CONDITION_IA = "IA"
CONDITIONS: Tuple[str, str, str] = (CONDITION_LE, CONDITION_HE, CONDITION_IA)

DOMAIN_OTHER = "Other"


@dataclass
class PPSSample:
    """Unified PPS sample built from TRACER / PolitiFact-Hidden records."""

    sample_id: str
    claim: str
    presented_evidence: List[str]
    hidden_evidence: List[str]
    inferred_intent_gold: str
    label: str
    domain: str = DOMAIN_OTHER
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PPSPrediction:
    """Structured model judgment for one sample and evidence condition."""

    sample_id: str
    condition: str
    provider: str
    model: str
    gold_label: str
    domain: str
    final_label: str
    literal_support: str = "unclear"
    missing_context_risk: str = "unclear"
    inferred_intent: str = ""
    inferred_intent_gold: str = ""
    short_explanation: str = ""
    raw_response: str = ""
    prompt_hash: str = ""
    error: Optional[str] = None
    intent_recovered: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def make_prediction_key(record: Dict[str, Any]) -> str:
    """Return the checkpoint key for a prediction-like record."""

    return "|".join(
        [
            str(record.get("provider", "")),
            str(record.get("model", "")),
            str(record.get("condition", "")),
            str(record.get("sample_id", "")),
        ]
    )
