"""Controlled RGB image corruptions applied before model preprocessing."""

from __future__ import annotations

from io import BytesIO
from typing import Callable

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from .config import DegradationConfig

ImageDegradation = Callable[[Image.Image | np.ndarray], Image.Image]


def _as_rgb_image(image: Image.Image | np.ndarray) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"Expected an HxWx3 RGB array, received shape {array.shape}")
    if array.dtype != np.uint8:
        raise TypeError(f"Expected uint8 pixels in [0, 255], received {array.dtype}")
    return Image.fromarray(array.copy(), mode="RGB")


def gaussian_blur(image: Image.Image | np.ndarray, sigma: float) -> Image.Image:
    """Return an RGB Gaussian-blurred copy; PIL's radius corresponds to sigma."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    return _as_rgb_image(image).filter(ImageFilter.GaussianBlur(radius=float(sigma)))


def adjust_brightness(image: Image.Image | np.ndarray, factor: float) -> Image.Image:
    """Return a brightness-scaled RGB copy with valid 8-bit pixels."""
    if factor <= 0:
        raise ValueError("brightness factor must be positive")
    return ImageEnhance.Brightness(_as_rgb_image(image)).enhance(float(factor))


def jpeg_compress(image: Image.Image | np.ndarray, quality: int) -> Image.Image:
    """Round-trip an RGB copy through in-memory JPEG encoding."""
    if not 1 <= quality <= 100:
        raise ValueError("JPEG quality must be between 1 and 100")
    buffer = BytesIO()
    _as_rgb_image(image).save(buffer, format="JPEG", quality=int(quality))
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return decoded.convert("RGB").copy()


def build_degradations(config: DegradationConfig) -> dict[str, ImageDegradation | None]:
    """Build named, deterministic conditions including the untouched clean view."""
    conditions: dict[str, ImageDegradation | None] = {"clean": None}
    for sigma in config.gaussian_blur_sigmas:
        conditions[f"gaussian_blur_sigma_{sigma:g}"] = (
            lambda image, value=sigma: gaussian_blur(image, value)
        )
    for factor in config.brightness_factors:
        conditions[f"brightness_factor_{factor:g}"] = (
            lambda image, value=factor: adjust_brightness(image, value)
        )
    for quality in config.jpeg_qualities:
        conditions[f"jpeg_quality_{quality}"] = (
            lambda image, value=quality: jpeg_compress(image, value)
        )
    return conditions

