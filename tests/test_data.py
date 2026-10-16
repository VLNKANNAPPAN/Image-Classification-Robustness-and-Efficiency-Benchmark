from pathlib import Path
import tempfile
import unittest

from PIL import Image

from robustness.config import ExperimentConfig
from robustness.data import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_transforms,
    build_dataloaders,
    discover_dataset_paths,
    stratified_split_indices,
)


class DataTests(unittest.TestCase):
    @staticmethod
    def _write_image(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (12, 12), color=(20, 40, 60)).save(path)

    def test_discovers_nested_intel_layout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split in ("seg_train/seg_train", "seg_test/seg_test"):
                for class_name in ("forest", "sea"):
                    (root / split / class_name).mkdir(parents=True)
            paths = discover_dataset_paths(root)
            self.assertEqual(paths.train, (root / "seg_train/seg_train").resolve())
            self.assertEqual(paths.test, (root / "seg_test/seg_test").resolve())

    def test_missing_root_has_actionable_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "not-mounted"
            with self.assertRaisesRegex(FileNotFoundError, "Attach the Kaggle dataset"):
                discover_dataset_paths(missing)

    def test_rejects_inconsistent_train_test_class_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(2):
                self._write_image(root / "train/forest" / f"{index}.jpg")
                self._write_image(root / "train/sea" / f"{index}.jpg")
            self._write_image(root / "test/forest/0.jpg")
            self._write_image(root / "test/street/0.jpg")
            config = ExperimentConfig()
            config.data.dataset_root = str(root)
            config.data.num_workers = 0
            config.model.num_classes = 2
            with self.assertRaisesRegex(ValueError, "class mappings differ"):
                build_dataloaders(config)

    def test_split_is_reproducible_disjoint_and_stratified(self) -> None:
        targets = [0] * 10 + [1] * 10 + [2] * 10
        train_a, validation_a = stratified_split_indices(targets, 0.2, seed=42)
        train_b, validation_b = stratified_split_indices(targets, 0.2, seed=42)
        self.assertEqual((train_a, validation_a), (train_b, validation_b))
        self.assertTrue(set(train_a).isdisjoint(validation_a))
        self.assertEqual(len(train_a) + len(validation_a), len(targets))
        self.assertEqual([targets[index] for index in validation_a].count(0), 2)
        self.assertEqual([targets[index] for index in validation_a].count(1), 2)
        self.assertEqual([targets[index] for index in validation_a].count(2), 2)

    def test_evaluation_transform_shape_and_normalization_contract(self) -> None:
        _, evaluation_transform = build_transforms(64)
        image = Image.new("RGB", (90, 70), color=(128, 128, 128))
        tensor = evaluation_transform(image)
        self.assertEqual(tuple(tensor.shape), (3, 64, 64))
        expected = tuple((128 / 255 - mean) / std for mean, std in zip(IMAGENET_MEAN, IMAGENET_STD))
        for channel, value in enumerate(expected):
            self.assertAlmostEqual(tensor[channel, 0, 0].item(), value, places=5)


if __name__ == "__main__":
    unittest.main()

