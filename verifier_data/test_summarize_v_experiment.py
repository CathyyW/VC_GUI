import tempfile
import unittest
from pathlib import Path

from summarize_v_experiment import (
    CORE_RUNS,
    build_experiment_summary,
    calculate_key_metrics,
    metric_row,
    write_csv,
)


def summary(
    accuracy: float,
    errors: int,
    mean_margin: float,
    min_margin: float,
    back_errors: int | None = None,
) -> dict:
    result = {
        "evaluated": 159 if accuracy > 0.8 and errors > 1 else 50,
        "accuracy_chosen_gt_rejected": accuracy,
        "chosen_not_preferred_or_tied_count": errors,
        "mean_margin": mean_margin,
        "min_margin": min_margin,
        "p10_margin": 10.0,
        "p50_margin": 30.0,
        "p90_margin": 60.0,
    }
    if back_errors is not None:
        result["rejected_navigate_back_preferred_or_tied_count"] = back_errors
    return result


class SummarizeVExperimentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.runs = {
            "original_full_test": summary(0.89937106918239, 16, 44.0449, -27.7454),
            "trained_full_test": summary(1.0, 0, 43.1145, 8.7599),
            "original_test_back_real_c": summary(0.68, 16, 21.3787, -27.7454, 16),
            "original_test_back_empty_c": summary(0.70, 15, 20.6430, -28.4246, 15),
            "trained_test_back_real_c": summary(0.98, 1, 28.5438, -14.2698, 1),
            "trained_test_back_empty_c": summary(0.98, 1, 24.1637, -6.3143, 1),
        }

    def test_key_metrics_match_experiment_results(self) -> None:
        key = calculate_key_metrics(self.runs)
        self.assertAlmostEqual(
            key["full_test"]["accuracy_gain_percentage_points"],
            10.062893081761,
        )
        self.assertEqual(key["full_test"]["original_errors"], 16)
        self.assertEqual(key["full_test"]["trained_errors"], 0)
        self.assertAlmostEqual(
            key["navigate_back_diagnostic"]["accuracy_gain_percentage_points"],
            30.0,
        )
        self.assertEqual(
            key["navigate_back_diagnostic"]["trained_real_c_back_errors"],
            1,
        )
        self.assertAlmostEqual(
            key["critic_ablation_after_training"]["mean_margin_difference"],
            4.3801,
        )

    def test_report_and_csv_are_generated(self) -> None:
        key = calculate_key_metrics(self.runs)
        rows = [metric_row(name, self.runs[name]) for name in CORE_RUNS]
        report = build_experiment_summary(
            rows,
            key,
            {
                "train_runtime": 8696.4777,
                "train_loss": 0.0836981121,
                "eval_accuracy": 1.0,
                "args": {
                    "base_model_path": "/models/base",
                    "lora_path": "/models/original-v",
                    "epoch": 1,
                    "learning_rate": 2e-5,
                },
            },
            {
                "pairs": {"train": 1335, "val": 165, "test": 159},
                "steps_with_pairs": {"train": 445, "val": 55, "test": 53},
                "max_rejected_per_step": 3,
            },
            [],
        )
        self.assertIn("89.94%", report)
        self.assertIn("100.00%", report)
        self.assertIn("16 → 1", report)
        self.assertIn("尚不单独宣称", report)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.csv"
            write_csv(
                path,
                rows,
                ["run_name", "accuracy_percent", "mean_margin"],
            )
            content = path.read_text(encoding="utf-8-sig")
            self.assertIn("original_full_test", content)
            self.assertIn("89.937106918239", content)


if __name__ == "__main__":
    unittest.main()
