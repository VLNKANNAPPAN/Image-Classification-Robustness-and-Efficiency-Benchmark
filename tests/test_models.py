import unittest

import torch

from robustness.models import SmallCNN, build_mobilenet_v2


class ModelTests(unittest.TestCase):
    def test_baseline_output_contains_one_logit_per_class(self) -> None:
        model = SmallCNN(num_classes=6, channels=[8, 16, 32], dropout=0.0)
        output = model(torch.zeros(2, 3, 64, 64))
        self.assertEqual(tuple(output.shape), (2, 6))

    def test_mobilenet_head_and_freezing_contract(self) -> None:
        model = build_mobilenet_v2(
            num_classes=6, dropout=0.2, pretrained=False, freeze_features=True
        )
        self.assertTrue(all(not parameter.requires_grad for parameter in model.features.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in model.classifier.parameters()))
        output = model(torch.zeros(1, 3, 224, 224))
        self.assertEqual(tuple(output.shape), (1, 6))


if __name__ == "__main__":
    unittest.main()

