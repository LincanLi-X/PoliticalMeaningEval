"""Checkpointed PPS experiment runner."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .data_loader import load_tracer_samples
from .llm_client import BaseModelClient, build_client
from .prompts import PromptRenderer
from .schema import CONDITIONS, PPSPrediction, PPSSample, make_prediction_key
from .utils import (
    append_jsonl,
    chunked,
    ensure_dir,
    iter_jsonl,
    normalize_label,
    parse_json_object,
    read_json,
    resolve_project_path,
    stable_hash,
    utc_timestamp,
    write_json,
)


def load_config(config_path: Path) -> Dict[str, Any]:
    config = read_json(config_path)
    if not isinstance(config, dict):
        raise ValueError(f"config must be a JSON object: {config_path}")
    project_root = config_path.parent.parent if config_path.parent.name == "configs" else Path.cwd()
    config.setdefault("project_root", str(project_root.resolve()))
    return config


def resolve_data_path(config: Dict[str, Any]) -> Path:
    project_root = Path(config["project_root"])
    if config.get("data_path"):
        data_path = resolve_project_path(project_root, config["data_path"])
        assert data_path is not None
        return data_path
    dataset_root = resolve_project_path(project_root, config.get("dataset_root", "TRACER/dataset"))
    split = config.get("split", "test")
    assert dataset_root is not None
    return dataset_root / f"{split}.json"


def parse_prediction_response(
    raw_response: str,
    sample: PPSSample,
    condition: str,
    provider: str,
    model: str,
    prompt_hash: str,
) -> PPSPrediction:
    error: Optional[str] = None
    data: Dict[str, Any] = {}
    try:
        data = parse_json_object(raw_response)
    except Exception as exc:  # noqa: BLE001 - preserve model response error.
        error = str(exc)

    final_label = normalize_label(data.get("final_label") or data.get("label"))
    return PPSPrediction(
        sample_id=sample.sample_id,
        condition=condition,
        provider=provider,
        model=model,
        gold_label=sample.label,
        domain=sample.domain,
        final_label=final_label,
        literal_support=str(data.get("literal_support") or "unclear"),
        missing_context_risk=str(data.get("missing_context_risk") or "unclear"),
        inferred_intent=str(data.get("inferred_intent") or ""),
        inferred_intent_gold=sample.inferred_intent_gold,
        short_explanation=str(data.get("short_explanation") or ""),
        raw_response=raw_response,
        prompt_hash=prompt_hash,
        error=error,
        intent_recovered=data.get("intent_recovered")
        if isinstance(data.get("intent_recovered"), bool)
        else None,
    )


def completed_keys(prediction_path: Path) -> set[str]:
    return {make_prediction_key(record) for record in iter_jsonl(prediction_path)}


def run_experiment(
    config: Dict[str, Any],
    client: Optional[BaseModelClient] = None,
    overwrite: bool = False,
    resume: bool = True,
) -> Dict[str, Path]:
    """Run PPS predictions and return output file paths."""

    project_root = Path(config["project_root"])
    provider = str(config.get("provider", "mock"))
    model = str(config.get("model", "mock-pps"))
    conditions = list(config.get("conditions") or CONDITIONS)
    for condition in conditions:
        if condition not in CONDITIONS:
            raise ValueError(f"unknown condition in config: {condition}")

    output_dir = resolve_project_path(project_root, config.get("output_dir", "outputs/run"))
    assert output_dir is not None
    ensure_dir(output_dir)
    prediction_path = output_dir / "predictions.jsonl"
    run_config_path = output_dir / "run_config.json"

    if overwrite and prediction_path.exists():
        prediction_path.unlink()

    data_path = resolve_data_path(config)
    prompt_dir = resolve_project_path(project_root, config.get("prompt_dir", "prompts"))
    assert prompt_dir is not None
    samples = load_tracer_samples(
        data_path=data_path,
        limit=config.get("limit"),
        presented_label=int(config.get("annotation_presented_label", 0)),
        hidden_label=int(config.get("annotation_hidden_label", 1)),
    )
    renderer = PromptRenderer(prompt_dir)
    model_client = client or build_client(
        provider=provider,
        model=model,
        temperature=float(config.get("temperature", 0.0)),
        max_tokens=int(config.get("max_tokens", 700)),
    )

    config_to_write = dict(config)
    config_to_write["resolved_data_path"] = str(data_path)
    config_to_write["started_at_utc"] = utc_timestamp()
    config_to_write["sample_count"] = len(samples)
    write_json(run_config_path, config_to_write)

    done = completed_keys(prediction_path) if resume else set()
    batch_size = int(config.get("batch_size", 1))
    delay = float(config.get("request_delay_seconds", 0.0))

    tasks = list(_build_tasks(samples, conditions))
    for batch in chunked(tasks, batch_size):
        for sample, condition in batch:
            key_record = {
                "provider": provider,
                "model": model,
                "condition": condition,
                "sample_id": sample.sample_id,
            }
            key = make_prediction_key(key_record)
            if key in done:
                continue
            prompt = renderer.render(sample, condition)
            prompt_hash = stable_hash(prompt)
            try:
                raw_response = model_client.generate(prompt, sample=sample, condition=condition)
                prediction = parse_prediction_response(
                    raw_response=raw_response,
                    sample=sample,
                    condition=condition,
                    provider=provider,
                    model=model,
                    prompt_hash=prompt_hash,
                )
            except Exception as exc:  # noqa: BLE001 - checkpoint failed sample.
                prediction = PPSPrediction(
                    sample_id=sample.sample_id,
                    condition=condition,
                    provider=provider,
                    model=model,
                    gold_label=sample.label,
                    domain=sample.domain,
                    final_label="",
                    inferred_intent_gold=sample.inferred_intent_gold,
                    prompt_hash=prompt_hash,
                    error=str(exc),
                )
            append_jsonl(prediction_path, prediction.to_dict())
            done.add(key)
            if delay > 0:
                time.sleep(delay)

    return {
        "predictions": prediction_path,
        "run_config": run_config_path,
        "output_dir": output_dir,
    }


def _build_tasks(
    samples: Iterable[PPSSample],
    conditions: Iterable[str],
) -> Iterable[tuple[PPSSample, str]]:
    for sample in samples:
        for condition in conditions:
            yield sample, condition
