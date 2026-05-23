#!/usr/bin/env python3
"""Run the PPS omission-sensitive verification experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pps_experiment.runner import load_config, run_experiment  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to JSON config.")
    parser.add_argument("--data-path", help="Override data JSON path.")
    parser.add_argument("--split", help="Override split name when using dataset_root.")
    parser.add_argument("--output-dir", help="Override output directory.")
    parser.add_argument("--provider", help="Override provider: mock, openai, anthropic, gemini, local_hf.")
    parser.add_argument("--model", help="Override model name.")
    parser.add_argument("--conditions", nargs="+", help="Subset/order of conditions: LE HE IA.")
    parser.add_argument("--limit", type=int, help="Limit number of samples.")
    parser.add_argument("--batch-size", type=int, help="Override batch size.")
    parser.add_argument("--prompt-dir", help="Override prompt template directory.")
    parser.add_argument("--resume", action="store_true", help="Skip already completed sample-condition records.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing predictions.jsonl in output dir.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    overrides = {
        "data_path": args.data_path,
        "split": args.split,
        "output_dir": args.output_dir,
        "provider": args.provider,
        "model": args.model,
        "conditions": args.conditions,
        "limit": args.limit,
        "batch_size": args.batch_size,
        "prompt_dir": args.prompt_dir,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value

    paths = run_experiment(
        config,
        overwrite=args.overwrite,
        resume=args.resume or not args.overwrite,
    )
    print(f"Predictions: {paths['predictions']}")
    print(f"Run config: {paths['run_config']}")


if __name__ == "__main__":
    main()
