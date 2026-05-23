#!/usr/bin/env python3
"""Aggregate PPS predictions into metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pps_experiment.aggregate import aggregate_predictions  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="Path to predictions JSONL.")
    parser.add_argument("--out", required=True, help="Path to write summary JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = aggregate_predictions(Path(args.predictions), Path(args.out))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
