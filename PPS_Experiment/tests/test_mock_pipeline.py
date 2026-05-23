"""Tiny mock tests for the PPS pipeline."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pps_experiment.aggregate import aggregate_predictions  # noqa: E402
from pps_experiment.data_loader import load_tracer_samples  # noqa: E402
from pps_experiment.runner import run_experiment  # noqa: E402
from pps_experiment.utils import iter_jsonl  # noqa: E402


class PPSMockPipelineTest(unittest.TestCase):
    def test_loader_and_mock_run(self) -> None:
        fixture = PROJECT_ROOT / "tests" / "fixtures" / "tiny_tracer.json"
        samples = load_tracer_samples(fixture)
        self.assertEqual(len(samples), 3)
        self.assertEqual(samples[1].presented_evidence[0], "A Texas official used a 35 percent figure that included direct, indirect, and induced impacts.")
        self.assertEqual(samples[1].hidden_evidence[0], "Direct oil and gas output accounted for a much smaller share of state gross product.")

        out_dir = PROJECT_ROOT / "outputs" / "unit_test_run"
        config = {
            "project_root": str(PROJECT_ROOT),
            "data_path": "tests/fixtures/tiny_tracer.json",
            "output_dir": "outputs/unit_test_run",
            "provider": "mock",
            "model": "mock-pps",
            "conditions": ["LE", "HE", "IA"],
            "prompt_dir": "prompts",
            "batch_size": 2,
        }
        paths = run_experiment(config, overwrite=True, resume=False)
        records = list(iter_jsonl(paths["predictions"]))
        self.assertEqual(len(records), 9)

        summary_path = out_dir / "summary.json"
        summary = aggregate_predictions(paths["predictions"], summary_path)
        model_summary = summary["models"]["mock/mock-pps"]
        self.assertEqual(model_summary["omission_sensitivity"]["HE"], 1.0)
        self.assertEqual(model_summary["over_trust_rate"]["HE"], 0.0)
        self.assertGreater(
            model_summary["reassessment_gain"]["HE_minus_LE"]["half_truth_f1"],
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
