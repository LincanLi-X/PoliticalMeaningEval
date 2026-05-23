# PPS Experiment

This folder contains the Political-LLM PPS experiment scaffold for omission-sensitive political fact verification. It adapts the bundled TRACER / PolitiFact-Hidden data to three evidence conditions:

- `LE`: claim plus presented evidence.
- `HE`: claim plus presented and hidden evidence.
- `IA`: hidden-evidence setting with explicit inferred-intent reasoning.

The model output schema is JSON with `literal_support`, `missing_context_risk`, `inferred_intent`, `final_label`, and `short_explanation`.

## Local Mock Test

No real LLM API is called:

```bash
cd PPS_Experiment
python -m unittest discover -s tests -p "test_*.py"
```

## Mock Dry Run

```bash
cd PPS_Experiment
python scripts/run_pps_experiment.py --config configs/pps_mock.json --limit 6 --overwrite
python scripts/aggregate_results.py --predictions outputs/mock_run/predictions.jsonl --out outputs/mock_run/summary.json
```

## API or Server Run

Set keys through environment variables only:

```bash
cd PPS_Experiment
export OPENAI_API_KEY="..."
export CUDA_VISIBLE_DEVICES="0"
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
  --batch-size 16 \
  --resume
```

## Notes

- The loader treats TRACER `annotation == 0` as presented evidence and `annotation == 1` as hidden evidence, matching the bundled data.
- Domain labels are inferred with a keyword heuristic when the source data has no explicit domain.
- Metrics include accuracy, macro-F1, Half-Truth F1, Omission Sensitivity, Intent Recovery Accuracy, Reassessment Gain, and Over-Trust Rate.
