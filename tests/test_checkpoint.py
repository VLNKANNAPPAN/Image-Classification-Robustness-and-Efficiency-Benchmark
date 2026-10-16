from pathlib import Path
import tempfile
import unittest

import torch

from robustness.models import SmallCNN
from robustness.train import load_checkpoint


class CheckpointTests(unittest.TestCase):
    def test_load_restores_saved_parameters(self) -> None:
        source = SmallCNN(2, [4], dropout=0.0)
        target = SmallCNN(2, [4], dropout=0.0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            torch.save({"model_state_dict": source.state_dict(), "epoch": 1}, path)
            metadata = load_checkpoint(target, path, torch.device("cpu"))
        self.assertEqual(metadata["epoch"], 1)
        for source_parameter, target_parameter in zip(source.parameters(), target.parameters()):
            self.assertTrue(torch.equal(source_parameter, target_parameter))

    def test_rejects_invalid_checkpoint(self) -> None:
        model = SmallCNN(2, [4], dropout=0.0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.pt"
            torch.save({"unexpected": "content"}, path)
            with self.assertRaises(ValueError):
                load_checkpoint(model, path, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()

