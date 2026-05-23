# PoliticalMeaningEval

> Official Code and dataset resources for the Political-LLM manuscript:

**Do Large Language Models Measure, Transform, and Circulate Political Meaning?**

`PoliticalMeaningEval` contains two complementary experiment suites for evaluating LLMs as political meaning infrastructures:

- **PPS Experiment**: omission-sensitive political fact verification with hidden evidence and inferred intent.
- **NPS Experiment**: multi-agent political news propagation simulation for measuring narrative drift and democratic communication risk.

The repository is designed for local mock testing, API-based model evaluation, and GPU-server execution. API keys are never hardcoded; use environment variables or local ignored configuration files.

## Repository Layout

```text
PoliticalMeaningEval/
├── PPS_Experiment/
│   ├── PPS_Codes_develope.md
│   ├── README.md
│   ├── TRACER/
│   │   └── dataset/
│   │       ├── train.json
│   │       ├── dev.json
│   │       └── test.json
│   ├── configs/
│   │   ├── pps_mock.json
│   │   └── server_example.json
│   ├── prompts/
│   │   ├── le.txt
│   │   ├── he.txt
│   │   └── ia.txt
│   ├── scripts/
│   │   ├── run_pps_experiment.py
│   │   └── aggregate_results.py
│   ├── src/
│   │   └── pps_experiment/
│   ├── tests/
│   └── outputs/
├── NPS_Experiment/
│   ├── NPS_Codes_develope.md
│   ├── README.md
│   ├── configs/
│   │   ├── smoke_mock.yaml
│   │   ├── pilot_20_seeds.yaml
│   │   ├── pilot_20_elections.yaml
│   │   └── experiment.yaml
│   ├── prompts/
│   ├── src/
│   ├── data/
│   └── outputs/
├── NPS_news_corpora_dataset/
    ├── nps_elections_verified_news_seeds_2023_2026.jsonl
    ├── nps_elections_verified_news_seeds_2024_2026_v2.jsonl
    ├── nps_immigration_verified_news_seeds_2024_2026.jsonl
    ├── nps_institutional_trust_verified_news_seeds_2024_2026.jsonl
    ├── nps_international_conflict_verified_news_seeds_2024_2026.jsonl
    └── nps_public_safety_verified_news_seeds_2024_2026.jsonl

```

## Experiments

### PPS: Positive Political Science

The PPS experiment evaluates whether LLMs preserve political meaning when hidden context changes the implication of a political claim. It adapts the TRACER / PolitiFact-Hidden dataset to three evidence conditions:

- `LE` or Literal-Evidence: the model receives the claim and presented evidence only.
- `HE` or Hidden-Evidence: the model receives the claim, presented evidence, and hidden evidence.
- `IA` or Intent-Aware: the model receives all evidence and is explicitly asked to infer the implied conclusion before assigning a label.

Each model returns a structured judgment:

```json
{
  "literal_support": "supported|partly_supported|unsupported|unclear",
  "missing_context_risk": "low|medium|high",
  "inferred_intent": "short text",
  "final_label": "true|half-true|false",
  "short_explanation": "short text"
}
```

Main PPS metrics:

- Accuracy
- Macro-F1
- Half-Truth F1
- Omission Sensitivity
- Intent Recovery Accuracy
- Reassessment Gain
- Over-Trust Rate

The PPS data source is included under:

```text
PPS_Experiment/TRACER/dataset/
```

The loader converts TRACER records into a unified schema with `claim`, `presented_evidence`, `hidden_evidence`, `inferred_intent`, `label`, and `domain`.

### NPS: Normative Political Science

The NPS experiment studies how political meaning changes when verified political news is repeatedly interpreted and transmitted by an LLM-agent community.

The simulation includes four political communication roles:

- Partisan Amplifiers
- Political Commentators
- Institutional Verifiers
- Ordinary Observers

The simulation supports:

- Multi-round propagation
- Agent memory
- Random, scale-free, high-clustering, and polarized-cluster networks
- Corrective verifier interventions
- Mock/dry-run mode for offline testing

Main NPS metrics:

- Factual Deviation
- Ideological Drift
- Blame Reassignment
- Agenda Shift
- Polarization Intensity

The NPS seed corpora are included under:

```text
NPS_news_corpora_dataset/
```

The main NPS configs merge and sample these JSONL files into working files under `NPS_Experiment/data/`.

## Setup

Python 3.10+ is recommended.

Create and activate an environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install the packages needed by the local code paths:

```bash
python -m pip install pyyaml
```

Optional packages are needed only for specific providers:

```bash
python -m pip install openai anthropic google-generativeai transformers torch
```

No real API is required for the mock tests below.

## Quick Start: Offline Checks

### PPS mock test

```bash
cd PPS_Experiment
python -m unittest discover -s tests -p "test_*.py"
```

### PPS mock run

```bash
cd PPS_Experiment
python scripts/run_pps_experiment.py \
  --config configs/pps_mock.json \
  --limit 6 \
  --overwrite

python scripts/aggregate_results.py \
  --predictions outputs/mock_run/predictions.jsonl \
  --out outputs/mock_run/summary.json
```

### NPS smoke run

Run from the repository root:

```bash
python -m NPS_Experiment.src.run_experiment \
  --config NPS_Experiment/configs/smoke_mock.yaml \
  --dry-run
```

For the smallest wiring check:

```bash
python -m NPS_Experiment.src.run_experiment \
  --config NPS_Experiment/configs/smoke_mock.yaml \
  --dry-run \
  --limit 1 \
  --num-rounds 1
```

These commands use deterministic mock behavior and do not call any real LLM API.

## Running PPS with API or Local Model Server

Set credentials through environment variables:

```bash
export OPENAI_API_KEY="..."
export CUDA_VISIBLE_DEVICES="0"
```

Run the test split:

```bash
cd PPS_Experiment
python scripts/run_pps_experiment.py \
  --config configs/server_example.json \
  --split test \
  --conditions LE HE IA \
  --batch-size 8 \
  --resume

python scripts/aggregate_results.py \
  --predictions outputs/server_run/predictions.jsonl \
  --out outputs/server_run/summary.json
```

For an OpenAI-compatible local endpoint such as vLLM:

```bash
cd PPS_Experiment
export OPENAI_API_KEY="EMPTY"
export OPENAI_BASE_URL="http://127.0.0.1:8000/v1"
export CUDA_VISIBLE_DEVICES="0,1"

python scripts/run_pps_experiment.py \
  --config configs/server_example.json \
  --provider openai \
  --model Qwen3-32B \
  --split test \
  --conditions LE HE IA \
  --batch-size 16 \
  --resume
```

## Running NPS

Start with the pilot config:

```bash
python -m NPS_Experiment.src.run_experiment \
  --config NPS_Experiment/configs/pilot_20_seeds.yaml \
  --dry-run
```

For live API mode:

```bash
export OPENAI_API_KEY="..."

python -m NPS_Experiment.src.run_experiment \
  --config NPS_Experiment/configs/pilot_20_seeds.yaml \
  --mode live
```

For the larger configured experiment:

```bash
python -m NPS_Experiment.src.run_experiment \
  --config NPS_Experiment/configs/experiment.yaml \
  --mode live
```

Use `--limit`, `--domain`, and `--num-rounds` for small controlled runs:

```bash
python -m NPS_Experiment.src.run_experiment \
  --config NPS_Experiment/configs/experiment.yaml \
  --dry-run \
  --domain Elections \
  --limit 2 \
  --num-rounds 2
```

## Output Files

PPS writes:

```text
PPS_Experiment/outputs/<run_name>/
  predictions.jsonl
  run_config.json
  summary.json
```

NPS writes:

```text
NPS_Experiment/outputs/
  trajectories/
  metrics/
  summaries/
```

Typical NPS summary files include:

- `round_level_drift.csv`
- `domain_level_metrics.csv`
- `role_level_metrics.csv`
- `run_metadata.json`

## Server Notes

The intended GPU server environment includes:

- 1 NVIDIA A100 80GB
- 2 NVIDIA RTX 6000
- 2 NVIDIA RTX 6000 Pro

Recommended usage:

- Use the A100 80GB for the largest local models or high-throughput vLLM service.
- Use RTX 6000 / RTX 6000 Pro GPUs for smaller local models or parallel model-serving workers.
- Keep PPS and NPS runs checkpointed and write outputs incrementally.
- Start with mock or pilot runs before launching full experiments.
- For hosted APIs, tune concurrency conservatively to avoid rate limits.

## Reproducibility Checklist

For paper-grade runs, record:

- Git commit hash
- Model provider and exact model snapshot
- Prompt files and prompt hashes
- Config files
- Dataset split and sample IDs
- Random seeds
- Decoding parameters
- Run date
- Raw predictions and evaluator outputs
- Aggregation script version

The current codebase is structured so that PPS predictions are JSONL records and NPS trajectories/metrics are saved per seed, making interrupted jobs easier to audit and resume.

## Data Notes

PPS uses the bundled TRACER / PolitiFact-Hidden data under `PPS_Experiment/TRACER/dataset/`.

NPS uses the curated verified-news seed corpora under `NPS_news_corpora_dataset/`. Each JSONL record is expected to include fields such as:

- `seed_id`
- `domain`
- `event_date`
- `country_region`
- `original_verified_report`
- `supporting_fact_check_evidence`
- `real_world_distorted_or_reframed_variant`
- `distortion_type`
- `nps_use_note`

Please check the original data licenses and source requirements before redistribution outside the review package.

## Citation

If you use this repository, please cite the Political-LLM manuscript:

```bibtex
@misc{political_llm_2026,
  title = {Do Large Language Models Measure, Transform, and Circulate Political Meaning?},
  author = {Anonymous Authors},
  year = {2026},
  note = {Under Review}
}
```

## Contact

For anonymous review, please use the submission system rather than direct contact.
