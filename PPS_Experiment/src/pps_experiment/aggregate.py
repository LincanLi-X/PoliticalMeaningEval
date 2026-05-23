"""Results aggregation entry points."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .metrics import summarize_records
from .utils import iter_jsonl, write_json


def aggregate_predictions(predictions_path: Path, out_path: Path) -> Dict[str, Any]:
    records = list(iter_jsonl(predictions_path))
    summary = summarize_records(records)
    write_json(out_path, summary)
    return summary
