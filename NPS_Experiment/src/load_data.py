"""Dataset loading, validation, merging and sampling utilities for NPS."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


REQUIRED_FIELDS = {
    "seed_id",
    "domain",
    "event_date",
    "country_region",
    "election_context",
    "original_verified_report",
    "original_report_source",
    "supporting_fact_check_evidence",
    "fact_check_source",
    "real_world_distorted_or_reframed_variant",
    "distortion_type",
    "nps_use_note",
}


DEFAULT_SOURCE_FILES = [
    "../nps_elections_verified_news_seeds_2024_2026_v2.jsonl",
    "../nps_immigration_verified_news_seeds_2024_2026.jsonl",
    "../nps_public_safety_verified_news_seeds_2024_2026.jsonl",
    "../nps_international_conflict_verified_news_seeds_2024_2026.jsonl",
    "../nps_institutional_trust_verified_news_seeds_2024_2026.jsonl",
]


class DatasetValidationError(ValueError):
    """Raised when a seed dataset is malformed."""


def load_jsonl(path: str | Path) -> list[dict]:
    """Load a JSONL file into a list of dictionaries."""
    path = Path(path)
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetValidationError(f"{path}:{line_no} is not valid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise DatasetValidationError(f"{path}:{line_no} must be a JSON object.")
            records.append(record)
    return records


def save_jsonl(records: Iterable[dict], path: str | Path) -> None:
    """Save dictionaries as UTF-8 JSONL."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def validate_seed_schema(seeds: list[dict]) -> None:
    """Validate required fields, duplicate IDs, dates and core value types."""
    if not seeds:
        raise DatasetValidationError("Seed dataset is empty.")

    seen_ids: set[str] = set()
    for idx, seed in enumerate(seeds):
        missing = REQUIRED_FIELDS - seed.keys()
        if missing:
            raise DatasetValidationError(f"Record {idx} missing fields: {sorted(missing)}")
        extra = seed.keys() - REQUIRED_FIELDS
        if extra:
            raise DatasetValidationError(f"Record {idx} has unexpected fields: {sorted(extra)}")

        seed_id = seed["seed_id"]
        if not isinstance(seed_id, str) or not seed_id:
            raise DatasetValidationError(f"Record {idx} has invalid seed_id: {seed_id!r}")
        if seed_id in seen_ids:
            raise DatasetValidationError(f"Duplicate seed_id: {seed_id}")
        seen_ids.add(seed_id)

        event_date = seed["event_date"]
        if not isinstance(event_date, str) or len(event_date) < 4:
            raise DatasetValidationError(f"{seed_id} has invalid event_date: {event_date!r}")
        year = event_date[:4]
        if year not in {"2024", "2025", "2026"}:
            raise DatasetValidationError(f"{seed_id} is outside 2024-2026: {event_date}")

        for field in ("original_report_source", "fact_check_source", "distortion_type"):
            if not isinstance(seed[field], list):
                raise DatasetValidationError(f"{seed_id}.{field} must be a list.")

        for field in (
            "domain",
            "country_region",
            "original_verified_report",
            "supporting_fact_check_evidence",
            "real_world_distorted_or_reframed_variant",
            "nps_use_note",
        ):
            if not isinstance(seed[field], str) or not seed[field].strip():
                raise DatasetValidationError(f"{seed_id}.{field} must be a non-empty string.")


def merge_seed_files(paths: Iterable[str | Path]) -> list[dict]:
    """Load, concatenate and validate multiple seed files."""
    merged: list[dict] = []
    for path in paths:
        merged.extend(load_jsonl(path))
    validate_seed_schema(merged)
    return merged


def sample_by_domain(seeds: list[dict], n_per_domain: int = 40, random_seed: int = 42) -> list[dict]:
    """Sample up to n_per_domain seeds per domain with deterministic ordering."""
    validate_seed_schema(seeds)
    rng = random.Random(random_seed)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for seed in seeds:
        grouped[seed["domain"]].append(seed)

    sampled: list[dict] = []
    for domain in sorted(grouped):
        records = list(grouped[domain])
        if len(records) < n_per_domain:
            raise DatasetValidationError(
                f"Domain {domain!r} has {len(records)} records, fewer than requested {n_per_domain}."
            )
        rng.shuffle(records)
        sampled.extend(sorted(records[:n_per_domain], key=lambda r: r["seed_id"]))
    validate_seed_schema(sampled)
    return sampled


def summarize(seeds: list[dict]) -> dict:
    """Return compact counts for reporting and smoke tests."""
    validate_seed_schema(seeds)
    return {
        "total": len(seeds),
        "by_domain": dict(sorted(Counter(seed["domain"] for seed in seeds).items())),
        "by_year": dict(sorted(Counter(seed["event_date"][:4] for seed in seeds).items())),
    }


def _resolve_default_paths(base_dir: Path) -> list[Path]:
    return [(base_dir / rel).resolve() for rel in DEFAULT_SOURCE_FILES]


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge and sample NPS seed JSONL files.")
    parser.add_argument("--source", action="append", dest="sources", help="Source JSONL path. Repeatable.")
    parser.add_argument("--merged-output", default="NPS_exp/data/merged_nps_news_seeds.jsonl")
    parser.add_argument("--sampled-output", default="NPS_exp/data/nps_200_sampled_seeds.jsonl")
    parser.add_argument("--n-per-domain", type=int, default=40)
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    cwd = Path.cwd()
    nps_dir = Path(__file__).resolve().parents[1]
    sources = [Path(p).resolve() for p in args.sources] if args.sources else _resolve_default_paths(nps_dir)

    merged = merge_seed_files(sources)
    sampled = sample_by_domain(merged, args.n_per_domain, args.random_seed)

    save_jsonl(merged, cwd / args.merged_output)
    save_jsonl(sampled, cwd / args.sampled_output)

    print(json.dumps({"merged": summarize(merged), "sampled": summarize(sampled)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

