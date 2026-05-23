# NPS Experiment

This directory contains the Narrative Propagation Shift (NPS) multi-agent simulation used for the Political-LLM NPS experiment. It supports the five JSONL seed corpora in `NPS_news_corpora_dataset/`, four political communication roles, multi-round propagation with bounded memory, corrective verifier interventions, and the five manuscript metrics:

- `factual_deviation`
- `ideological_drift`
- `blame_reassignment`
- `agenda_shift`
- `polarization_intensity`

## Offline Smoke Test

Run from the repository root:

```bash
python -m NPS_Experiment.src.run_experiment \
  --config NPS_Experiment/configs/smoke_mock.yaml \
  --dry-run
```

The smoke config uses deterministic mock LLM outputs, two seeds, two rounds, and writes to `NPS_Experiment/outputs/smoke_mock/`. It does not call any real API.

For an even smaller wiring check:

```bash
python -m NPS_Experiment.src.run_experiment \
  --config NPS_Experiment/configs/smoke_mock.yaml \
  --dry-run \
  --limit 1 \
  --num-rounds 1
```

## Dataset Loading

The main configs read the five 2024-2026 JSONL files from `NPS_news_corpora_dataset/` and write merged/sampled working files under `NPS_Experiment/data/`. `source_files` entries may be explicit paths, and `source_glob` may be used for controlled glob patterns. Avoid globbing both elections files together unless duplicate `seed_id` values are intentionally resolved first.

## Live Model Configuration

Live mode uses the OpenAI client only when `models.mode: live` or `--mode live` is set. API credentials must be supplied through environment variables or a local `.env` file, for example:

```bash
export OPENAI_API_KEY=...
```

No API key should be committed to configs or source files.

## GPU Server Run Notes

The default experiment config is CPU/API-client oriented and safe for one A100 80GB plus two RTX 6000 and two RTX 6000 Pro hosts. Recommended starting settings:

- Keep `max_concurrent_agent_calls: 10` for hosted API generation; raise gradually only after checking rate limits.
- For local inference servers, run one model server per GPU class and expose it behind an OpenAI-compatible endpoint before switching `LLMClient` to that provider.
- Use `topology: polarized_clusters` for the manuscript-style echo-chamber setting; use `random`, `scale_free`, or `high_clustering` for topology ablations.
- Start with `pilot_20_seeds.yaml` before the 200-seed `experiment.yaml`.

## Main Entry Points

```bash
python -m NPS_Experiment.src.load_data
python -m NPS_Experiment.src.run_experiment --config NPS_Experiment/configs/experiment.yaml --dry-run
```

Outputs are written as trajectories, per-seed metric JSON, and CSV summaries under `NPS_Experiment/outputs/`.
