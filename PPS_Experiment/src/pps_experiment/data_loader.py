"""Load TRACER / PolitiFact-Hidden data into the PPS schema."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .schema import DOMAIN_OTHER, PPSSample
from .utils import normalize_label, read_json


DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "Budget/Economy": [
        "budget",
        "tax",
        "taxes",
        "economy",
        "economic",
        "jobs",
        "unemployment",
        "inflation",
        "debt",
        "deficit",
        "spending",
        "wage",
        "oil",
        "gas",
        "utility",
        "prices",
    ],
    "Immigration": [
        "immigration",
        "immigrant",
        "migrant",
        "border",
        "asylum",
        "deport",
        "visa",
        "undocumented",
        "illegal",
    ],
    "Public Safety": [
        "crime",
        "police",
        "gun",
        "shooting",
        "prison",
        "jail",
        "safety",
        "violence",
        "drug",
        "opioid",
    ],
    "Elections": [
        "election",
        "vote",
        "voter",
        "ballot",
        "campaign",
        "poll",
        "mail-in",
        "absentee",
        "electoral",
    ],
    "Institutional Trust": [
        "congress",
        "court",
        "supreme court",
        "agency",
        "fbi",
        "justice",
        "senate",
        "house",
        "governor",
        "president",
        "administration",
        "government",
    ],
    "Healthcare/Social Policy": [
        "health",
        "healthcare",
        "medicare",
        "medicaid",
        "insurance",
        "covid",
        "coronavirus",
        "school",
        "education",
        "welfare",
        "social security",
    ],
}


def infer_domain(record: Dict[str, Any], evidence_limit: int = 8) -> str:
    """Infer manuscript issue area when the dataset lacks a domain field."""

    explicit = record.get("domain") or record.get("topic") or record.get("issue_area")
    if explicit:
        return str(explicit)

    evidence = record.get("evidence") or []
    text_parts = [
        str(record.get("claim", "")),
        str(record.get("ruling", "")),
        " ".join(str(item) for item in evidence[:evidence_limit]),
    ]
    text = " ".join(text_parts).lower()

    best_domain = DOMAIN_OTHER
    best_count = 0
    for domain, keywords in DOMAIN_KEYWORDS.items():
        count = sum(1 for keyword in keywords if keyword in text)
        if count > best_count:
            best_count = count
            best_domain = domain
    return best_domain


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if str(value).strip():
        return [str(value)]
    return []


def split_evidence(
    record: Dict[str, Any],
    presented_label: int = 0,
    hidden_label: int = 1,
) -> tuple[List[str], List[str]]:
    """Return presented and hidden evidence for a TRACER-like record."""

    explicit_presented = _as_list(
        record.get("presented_evidence")
        or record.get("presentedEvidence")
        or record.get("pe")
    )
    explicit_hidden = _as_list(
        record.get("hidden_evidence")
        or record.get("hiddenEvidence")
        or record.get("he")
    )
    if explicit_presented or explicit_hidden:
        return explicit_presented, explicit_hidden

    evidence = _as_list(record.get("evidence"))
    annotation = record.get("annotation") or record.get("evidence_alignment")
    if isinstance(annotation, list) and len(annotation) == len(evidence):
        presented: List[str] = []
        hidden: List[str] = []
        for sentence, label in zip(evidence, annotation):
            try:
                label_int = int(label)
            except (TypeError, ValueError):
                label_int = -1
            if label_int == presented_label:
                presented.append(sentence)
            elif label_int == hidden_label:
                hidden.append(sentence)
        return presented, hidden

    return evidence, []


def load_tracer_samples(
    data_path: Path,
    limit: Optional[int] = None,
    presented_label: int = 0,
    hidden_label: int = 1,
) -> List[PPSSample]:
    """Load a TRACER JSON split into unified PPS samples."""

    raw_data = read_json(data_path)
    if not isinstance(raw_data, list):
        raise ValueError(f"expected a list of records in {data_path}")

    samples: List[PPSSample] = []
    for record in raw_data:
        if limit is not None and len(samples) >= limit:
            break
        if not isinstance(record, dict):
            continue

        presented, hidden = split_evidence(
            record,
            presented_label=presented_label,
            hidden_label=hidden_label,
        )
        label = normalize_label(record.get("veracity") or record.get("label") or record.get("rating"))
        sample_id = str(record.get("example_id") or record.get("id") or len(samples))
        intent_gold = str(record.get("intent_gold") or record.get("intent") or "")
        metadata = {
            "speaker": record.get("speaker"),
            "date": record.get("date"),
            "source": record.get("source"),
            "ruling": record.get("ruling"),
            "enrich_ruling": record.get("enrich_ruling"),
            "data_path": str(data_path),
            "raw_annotation_presented_label": presented_label,
            "raw_annotation_hidden_label": hidden_label,
        }
        samples.append(
            PPSSample(
                sample_id=sample_id,
                claim=str(record.get("claim") or ""),
                presented_evidence=presented,
                hidden_evidence=hidden,
                inferred_intent_gold=intent_gold,
                label=label,
                domain=infer_domain(record),
                metadata=metadata,
            )
        )
    return samples


def load_many(paths: Iterable[Path], limit: Optional[int] = None) -> List[PPSSample]:
    samples: List[PPSSample] = []
    for path in paths:
        remaining = None if limit is None else max(0, limit - len(samples))
        if remaining == 0:
            break
        samples.extend(load_tracer_samples(path, limit=remaining))
    return samples
