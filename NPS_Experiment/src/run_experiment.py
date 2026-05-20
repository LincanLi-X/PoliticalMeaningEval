"""Batch entry point for the NPS multi-agent experiment."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import Any

import yaml

from .agent_factory import build_agents
from .community import build_network, network_summary
from .llm_client import LLMClient
from .load_data import (
    load_jsonl,
    merge_seed_files,
    sample_by_domain,
    save_jsonl,
    summarize,
    validate_seed_schema,
)
from .logging_utils import ensure_dir, write_json
from .metrics import (
    MetricEvaluationError,
    compute_trajectory_metrics,
    save_metrics,
    write_domain_summary,
    write_role_summary,
    write_round_summary,
)
from .simulation import run_single_simulation, save_trajectory


def load_config(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def prepare_dataset(config: dict[str, Any]) -> list[dict[str, Any]]:
    dataset_config = config.get("dataset", {})
    source_files = [Path(path) for path in dataset_config.get("source_files", [])]
    merged_path = Path(dataset_config.get("merged_path", "NPS_exp/data/merged_nps_news_seeds.jsonl"))
    sampled_path = Path(dataset_config.get("sampled_path", "NPS_exp/data/nps_200_sampled_seeds.jsonl"))

    if source_files:
        merged = merge_seed_files(source_files)
        save_jsonl(merged, merged_path)
    elif merged_path.exists():
        merged = load_jsonl(merged_path)
        validate_seed_schema(merged)
    else:
        raise FileNotFoundError("No source_files configured and merged dataset does not exist.")

    mode = dataset_config.get("mode", "sample_40_per_domain")
    random_seed = int(dataset_config.get("random_seed", 42))
    if mode.startswith("sample_") and mode.endswith("_per_domain"):
        n_per_domain = int(dataset_config.get("n_per_domain", mode.removeprefix("sample_").removesuffix("_per_domain")))
        seeds = sample_by_domain(merged, n_per_domain=n_per_domain, random_seed=random_seed)
    elif mode == "sample_n_total":
        n_total = int(dataset_config.get("n_total", 20))
        rng = random.Random(random_seed)
        seeds = list(merged)
        rng.shuffle(seeds)
        seeds = sorted(seeds[:n_total], key=lambda row: row["seed_id"])
        validate_seed_schema(seeds)
    elif mode == "use_all":
        seeds = merged
    else:
        raise ValueError(f"Unsupported dataset mode: {mode}")

    save_jsonl(seeds, sampled_path)
    return seeds


def maybe_filter_and_limit(
    seeds: list[dict[str, Any]],
    domain: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    if domain:
        seeds = [seed for seed in seeds if seed["domain"].lower() == domain.lower()]
    if limit is not None:
        seeds = seeds[:limit]
    validate_seed_schema(seeds)
    return seeds


def copy_config(config_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output_dir / config_path.name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run NPS multi-agent propagation experiments.")
    parser.add_argument("--config", default="NPS_exp/configs/experiment.yaml")
    parser.add_argument("--mode", choices=["mock", "live"], help="Override models.mode from config.")
    parser.add_argument("--limit", type=int, help="Limit number of seeds after sampling/filtering.")
    parser.add_argument("--domain", help="Optional domain filter, e.g. Elections.")
    parser.add_argument("--skip-metrics", action="store_true")
    parser.add_argument("--num-rounds", type=int, help="Override simulation.num_rounds.")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    if args.mode:
        config.setdefault("models", {})["mode"] = args.mode
    if args.num_rounds is not None:
        config.setdefault("simulation", {})["num_rounds"] = args.num_rounds

    seeds = prepare_dataset(config)
    seeds = maybe_filter_and_limit(seeds, domain=args.domain, limit=args.limit)

    output_config = config.get("outputs", {})
    trajectories_dir = ensure_dir(output_config.get("trajectories_dir", "NPS_exp/outputs/trajectories"))
    metrics_dir = ensure_dir(output_config.get("metrics_dir", "NPS_exp/outputs/metrics"))
    summaries_dir = ensure_dir(output_config.get("summaries_dir", "NPS_exp/outputs/summaries"))
    copy_config(config_path, summaries_dir)

    generation_client = LLMClient(config, purpose="generation")
    evaluator_client = LLMClient(config, purpose="evaluation")

    metrics_records: list[dict[str, Any]] = []
    trajectories = []
    run_metadata: dict[str, Any] = {
        "dataset_summary": summarize(seeds),
        "mode": config.get("models", {}).get("mode", "mock"),
        "num_rounds": config.get("simulation", {}).get("num_rounds"),
        "checkpoint_rounds": config.get("simulation", {}).get("checkpoint_rounds", []),
        "completed_seeds": [],
    }

    for idx, seed in enumerate(seeds, start=1):
        agents = build_agents(config)
        network = build_network(
            agents,
            topology=config.get("community", {}).get("topology", "small_world"),
            seed=int(config.get("community", {}).get("network_seed", 42)),
        )
        if idx == 1:
            run_metadata["network_summary"] = network_summary(network, agents)

        trajectory = run_single_simulation(seed, agents, network, config, generation_client)
        trajectories.append(trajectory)
        save_trajectory(trajectory, trajectories_dir / f"{seed['seed_id']}.json")

        if not args.skip_metrics:
            checkpoint_rounds = list(config.get("simulation", {}).get("checkpoint_rounds", []))
            metrics_config = dict(config.get("metrics", {}))
            metrics_config.setdefault(
                "evaluation_max_tokens",
                config.get("models", {}).get("max_output_tokens", {}).get("evaluation", evaluator_client.default_max_tokens),
            )
            try:
                metrics = compute_trajectory_metrics(
                    trajectory,
                    evaluator_client,
                    checkpoint_rounds,
                    metrics_config=metrics_config,
                )
            except MetricEvaluationError as exc:
                error_record = {
                    "seed_id": seed["seed_id"],
                    "domain": seed["domain"],
                    "error": "metric_evaluation_failed",
                    "message": str(exc),
                    "evaluation": exc.metadata,
                }
                save_metrics(error_record, metrics_dir / f"{seed['seed_id']}.json")
                raise
            save_metrics(metrics, metrics_dir / f"{seed['seed_id']}.json")
            metrics_records.append(metrics)

        run_metadata["completed_seeds"].append(seed["seed_id"])
        print(f"[{idx}/{len(seeds)}] completed {seed['seed_id']} ({seed['domain']})")

    if metrics_records:
        write_round_summary(metrics_records, summaries_dir / "round_level_drift.csv")
        write_domain_summary(metrics_records, summaries_dir / "domain_level_metrics.csv")
    write_role_summary(trajectories, summaries_dir / "role_level_metrics.csv")
    write_json(run_metadata, summaries_dir / "run_metadata.json")
    print(f"Done. Trajectories: {trajectories_dir}")
    print(f"Done. Metrics: {metrics_dir}")
    print(f"Done. Summaries: {summaries_dir}")


if __name__ == "__main__":
    main()
