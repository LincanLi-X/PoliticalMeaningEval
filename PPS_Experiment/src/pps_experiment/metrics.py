"""Metrics for PPS omission-sensitive verification."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Tuple

from .schema import CONDITION_HE, CONDITION_IA, CONDITION_LE, LABEL_HALF_TRUE, LABEL_TRUE, LABELS
from .utils import normalize_label, normalize_text_tokens


def precision_recall_f1(y_true: List[str], y_pred: List[str], label: str) -> Dict[str, float]:
    tp = sum(1 for truth, pred in zip(y_true, y_pred) if truth == label and pred == label)
    fp = sum(1 for truth, pred in zip(y_true, y_pred) if truth != label and pred == label)
    fn = sum(1 for truth, pred in zip(y_true, y_pred) if truth == label and pred != label)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "support": float(sum(1 for y in y_true if y == label))}


def condition_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    usable = [
        record
        for record in records
        if normalize_label(record.get("gold_label")) in LABELS
        and normalize_label(record.get("final_label")) in LABELS
    ]
    y_true = [normalize_label(record.get("gold_label")) for record in usable]
    y_pred = [normalize_label(record.get("final_label")) for record in usable]
    total = len(usable)
    correct = sum(1 for truth, pred in zip(y_true, y_pred) if truth == pred)
    per_label = {label: precision_recall_f1(y_true, y_pred, label) for label in LABELS}
    macro_f1 = sum(values["f1"] for values in per_label.values()) / len(LABELS)
    return {
        "count": total,
        "accuracy": correct / total if total else 0.0,
        "macro_f1": macro_f1,
        "half_truth_f1": per_label[LABEL_HALF_TRUE]["f1"],
        "per_label": per_label,
        "parse_error_count": sum(1 for record in records if record.get("error")),
    }


def intent_matches(record: Dict[str, Any], threshold: float = 0.45) -> bool:
    explicit = record.get("intent_recovered")
    if isinstance(explicit, bool):
        return explicit
    predicted = str(record.get("inferred_intent") or "")
    gold = str(record.get("inferred_intent_gold") or "")
    pred_tokens = normalize_text_tokens(predicted)
    gold_tokens = normalize_text_tokens(gold)
    if not pred_tokens or not gold_tokens:
        return False
    overlap = len(pred_tokens & gold_tokens) / max(1, len(gold_tokens))
    return overlap >= threshold


def intent_recovery_accuracy(records: List[Dict[str, Any]]) -> float:
    if not records:
        return 0.0
    return sum(1 for record in records if intent_matches(record)) / len(records)


def by_condition(records: Iterable[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("condition"))].append(record)
    return dict(grouped)


def _records_by_id(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(record.get("sample_id")): record for record in records}


def omission_sensitivity(
    le_records: List[Dict[str, Any]],
    target_records: List[Dict[str, Any]],
) -> float:
    le_by_id = _records_by_id(le_records)
    target_by_id = _records_by_id(target_records)
    denominator = 0
    numerator = 0
    for sample_id, le_record in le_by_id.items():
        target = target_by_id.get(sample_id)
        if target is None:
            continue
        if normalize_label(le_record.get("gold_label")) != LABEL_HALF_TRUE:
            continue
        if normalize_label(le_record.get("final_label")) != LABEL_TRUE:
            continue
        denominator += 1
        if normalize_label(target.get("final_label")) == LABEL_HALF_TRUE:
            numerator += 1
    return numerator / denominator if denominator else 0.0


def over_trust_rate(records: List[Dict[str, Any]]) -> float:
    half_truth = [
        record
        for record in records
        if normalize_label(record.get("gold_label")) == LABEL_HALF_TRUE
    ]
    if not half_truth:
        return 0.0
    over_trusted = [
        record
        for record in half_truth
        if normalize_label(record.get("final_label")) == LABEL_TRUE
    ]
    return len(over_trusted) / len(half_truth)


def reassessment_gain(
    condition_summary: Dict[str, Dict[str, Any]],
    left: str,
    right: str,
) -> Dict[str, float]:
    left_metrics = condition_summary.get(left, {})
    right_metrics = condition_summary.get(right, {})
    return {
        "half_truth_f1": float(right_metrics.get("half_truth_f1", 0.0))
        - float(left_metrics.get("half_truth_f1", 0.0)),
        "macro_f1": float(right_metrics.get("macro_f1", 0.0))
        - float(left_metrics.get("macro_f1", 0.0)),
    }


def compute_pps_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    grouped = by_condition(records)
    condition_summary = {
        condition: {
            **condition_metrics(condition_records),
            "intent_recovery_accuracy": intent_recovery_accuracy(condition_records),
        }
        for condition, condition_records in grouped.items()
    }
    le_records = grouped.get(CONDITION_LE, [])
    he_records = grouped.get(CONDITION_HE, [])
    ia_records = grouped.get(CONDITION_IA, [])
    return {
        "conditions": condition_summary,
        "omission_sensitivity": {
            CONDITION_HE: omission_sensitivity(le_records, he_records),
            CONDITION_IA: omission_sensitivity(le_records, ia_records),
        },
        "over_trust_rate": {
            CONDITION_HE: over_trust_rate(he_records),
            CONDITION_IA: over_trust_rate(ia_records),
        },
        "reassessment_gain": {
            "HE_minus_LE": reassessment_gain(condition_summary, CONDITION_LE, CONDITION_HE),
            "IA_minus_HE": reassessment_gain(condition_summary, CONDITION_HE, CONDITION_IA),
        },
    }


def group_key(record: Dict[str, Any]) -> Tuple[str, str]:
    return str(record.get("provider", "")), str(record.get("model", ""))


def summarize_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_model: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_model[group_key(record)].append(record)

    model_summary = {
        f"{provider}/{model}": compute_pps_metrics(model_records)
        for (provider, model), model_records in by_model.items()
    }

    domain_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        domain_groups[str(record.get("domain") or "Other")].append(record)
    domain_summary = {
        domain: compute_pps_metrics(domain_records)
        for domain, domain_records in sorted(domain_groups.items())
    }

    return {
        "record_count": len(records),
        "models": model_summary,
        "domains": domain_summary,
    }
