import unittest

import torch
from torch import nn

from robustness.benchmark import benchmark_model


class BenchmarkTests(unittest.TestCase):
    def test_reports_explicit_shape_batch_and_units(self) -> None:
        model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 2))
        inputs = torch.zeros(2, 3, 8, 8)
        result = benchmark_model(
            model,
            inputs,
            torch.device("cpu"),
            "tiny",
            warmup_iterations=0,
            timed_iterations=2,
        )
        self.assertEqual(result.batch_size, 2)
        self.assertEqual(result.input_shape, (2, 3, 8, 8))
        self.assertGreaterEqual(result.median_batch_latency_ms, 0)
        self.assertGreater(result.throughput_images_per_second, 0)


if __name__ == "__main__":
    unittest.main()

