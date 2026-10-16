import unittest

import numpy as np
from PIL import Image

from robustness.degradations import adjust_brightness, gaussian_blur, jpeg_compress


class DegradationTests(unittest.TestCase):
    def setUp(self) -> None:
        generator = np.random.default_rng(123)
        self.array = generator.integers(0, 256, size=(31, 47, 3), dtype=np.uint8)
        self.image = Image.fromarray(self.array, mode="RGB")

    def assert_valid_unchanged_source(self, output: Image.Image) -> None:
        result = np.asarray(output)
        self.assertEqual(result.shape, self.array.shape)
        self.assertEqual(result.dtype, np.uint8)
        self.assertGreaterEqual(int(result.min()), 0)
        self.assertLessEqual(int(result.max()), 255)
        np.testing.assert_array_equal(np.asarray(self.image), self.array)

    def test_blur_is_deterministic_and_preserves_source(self) -> None:
        first = gaussian_blur(self.image, sigma=2.0)
        second = gaussian_blur(self.image, sigma=2.0)
        self.assert_valid_unchanged_source(first)
        np.testing.assert_array_equal(np.asarray(first), np.asarray(second))

    def test_brightness_is_deterministic_and_preserves_source(self) -> None:
        first = adjust_brightness(self.image, factor=0.6)
        second = adjust_brightness(self.image, factor=0.6)
        self.assert_valid_unchanged_source(first)
        np.testing.assert_array_equal(np.asarray(first), np.asarray(second))

    def test_jpeg_is_deterministic_and_preserves_source(self) -> None:
        first = jpeg_compress(self.image, quality=40)
        second = jpeg_compress(self.image, quality=40)
        self.assert_valid_unchanged_source(first)
        np.testing.assert_array_equal(np.asarray(first), np.asarray(second))

    def test_invalid_parameters_fail_loudly(self) -> None:
        with self.assertRaises(ValueError):
            gaussian_blur(self.image, sigma=0)
        with self.assertRaises(ValueError):
            adjust_brightness(self.image, factor=0)
        with self.assertRaises(ValueError):
            jpeg_compress(self.image, quality=0)


if __name__ == "__main__":
    unittest.main()

