"""Heatmap overlays as PNG bytes (used by the CLI and the API)."""
from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
from matplotlib import colormaps  # noqa: E402
from PIL import Image  # noqa: E402


def heat_rgb(heat: np.ndarray) -> np.ndarray:
    return (colormaps["inferno"](np.clip(heat, 0, 1))[..., :3] * 255).astype(np.uint8)


def overlay(img: np.ndarray, heat: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    h = heat_rgb(heat).astype(np.float32)
    a = (alpha * np.clip(heat, 0, 1))[..., None]
    return np.clip(img.astype(np.float32) * (1 - a) + h * a, 0, 255).astype(np.uint8)


def overlay_png(img: np.ndarray, heat: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(overlay(img, heat)).save(buf, format="PNG")
    return buf.getvalue()
