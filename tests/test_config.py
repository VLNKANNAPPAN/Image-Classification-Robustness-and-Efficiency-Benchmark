import tempfile
import unittest
from pathlib import Path

from robustness.config import load_config


class ConfigTests(unittest.TestCase):
    def test_loads_nested_values_and_dotted_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text("seed: 7\ndata:\n  batch_size: 8\n", encoding="utf-8")
            config = load_config(path, {"data.image_size": 128})
        self.assertEqual(config.seed, 7)
        self.assertEqual(config.data.batch_size, 8)
        self.assertEqual(config.data.image_size, 128)

    def test_rejects_unknown_keys(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text("not_a_setting: true\n", encoding="utf-8")
            with self.assertRaises(KeyError):
                load_config(path)

    def test_validation_rejects_invalid_experiment_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text("data:\n  validation_fraction: 1.0\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_config(path)


if __name__ == "__main__":
    unittest.main()

