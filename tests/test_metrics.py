import unittest

from robustness.evaluate import calculate_classification_metrics


class MetricTests(unittest.TestCase):
    def test_perfect_predictions(self) -> None:
        metrics = calculate_classification_metrics(
            [0, 0, 1, 1, 2, 2], [0, 0, 1, 1, 2, 2], ["a", "b", "c"]
        )
        self.assertEqual(metrics["sample_count"], 6)
        self.assertAlmostEqual(metrics["accuracy"], 1.0)
        self.assertAlmostEqual(metrics["macro_f1"], 1.0)
        self.assertEqual(metrics["confusion_matrix"], [[2, 0, 0], [0, 2, 0], [0, 0, 2]])

    def test_macro_metrics_include_unpredicted_class_without_division_error(self) -> None:
        metrics = calculate_classification_metrics(
            [0, 1, 2], [0, 0, 0], ["a", "b", "c"]
        )
        self.assertAlmostEqual(metrics["accuracy"], 1 / 3)
        self.assertEqual(metrics["per_class"]["b"]["precision"], 0.0)
        self.assertEqual(metrics["per_class"]["c"]["recall"], 0.0)

    def test_rejects_empty_or_mismatched_inputs(self) -> None:
        with self.assertRaises(ValueError):
            calculate_classification_metrics([], [], ["a", "b"])
        with self.assertRaises(ValueError):
            calculate_classification_metrics([0], [0, 1], ["a", "b"])


if __name__ == "__main__":
    unittest.main()

