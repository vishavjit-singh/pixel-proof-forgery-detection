"""Authentic image material: scikit-image sample photos without people + procedural scenes.

Only photos of objects, textures, animals, astronomy and microscopy are used. Images of people
(``astronaut``, ``camera``) are deliberately excluded.
"""
from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np
import skimage.data as skd

PHOTOS = ("brick", "chelsea", "cell", "grass", "hubble_deep_field", "immunohistochemistry", "retina",
          "clock", "coins", "coffee", "gravel", "moon", "rocket")
EXCLUDED = ("astronaut", "camera")      # contain people


@lru_cache(maxsize=None)
def photo(name: str) -> np.ndarray:
    if name in EXCLUDED or name not in PHOTOS:
        raise ValueError(f"{name!r} is not an allowed source photo")
    img = getattr(skd, name)()
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    return np.ascontiguousarray(img[..., :3]).astype(np.uint8)


def random_crop(img: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    h, w = img.shape[:2]
    side = int(min(h, w) * rng.uniform(0.45, 1.0))
    side = max(side, min(h, w, size // 2))
    y, x = rng.integers(0, h - side + 1), rng.integers(0, w - side + 1)
    crop = img[y:y + side, x:x + side]
    out = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA if side > size else cv2.INTER_CUBIC)
    if rng.random() < 0.5:
        out = out[:, ::-1]
    return np.ascontiguousarray(out)


def _smooth_noise(rng: np.random.Generator, size: int, scale: float) -> np.ndarray:
    small = max(2, int(size / scale))
    n = rng.normal(size=(small, small)).astype(np.float32)
    return cv2.resize(n, (size, size), interpolation=cv2.INTER_CUBIC)


def procedural_scene(rng: np.random.Generator, size: int) -> np.ndarray:
    """A plausible 'photo': lit gradient background, textured shapes, soft shadows, mild blur."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / size
    c1, c2 = rng.uniform(30, 225, 3), rng.uniform(30, 225, 3)
    angle = rng.uniform(0, np.pi)
    t = (np.cos(angle) * xx + np.sin(angle) * yy)
    t = (t - t.min()) / (np.ptp(t) + 1e-6)
    img = (c1[None, None] * (1 - t[..., None]) + c2[None, None] * t[..., None])
    img += 18 * _smooth_noise(rng, size, rng.uniform(8, 40))[..., None]
    for _ in range(rng.integers(3, 9)):
        color = rng.uniform(0, 255, 3)
        mask = np.zeros((size, size), np.uint8)
        cx, cy = rng.integers(0, size, 2)
        if rng.random() < 0.5:
            axes = (int(rng.integers(10, size // 3)), int(rng.integers(10, size // 3)))
            cv2.ellipse(mask, (int(cx), int(cy)), axes, float(rng.uniform(0, 180)), 0, 360, 255, -1)
        else:
            pts = (np.array([cx, cy]) + rng.normal(0, size / 7, (int(rng.integers(3, 7)), 2))).astype(np.int32)
            cv2.fillPoly(mask, [pts], 255)
        m = cv2.GaussianBlur(mask, (0, 0), rng.uniform(0.6, 2.0)).astype(np.float32)[..., None] / 255
        texture = color[None, None] + 25 * _smooth_noise(rng, size, rng.uniform(2, 12))[..., None]
        shade = 1 - 0.35 * (yy - cy / size)[..., None]
        img = img * (1 - m) + texture * shade * m
    img = cv2.GaussianBlur(img.astype(np.float32), (0, 0), rng.uniform(0.4, 1.0))
    return np.clip(img, 0, 255).astype(np.uint8)


def base_image(rng: np.random.Generator, size: int, photo_names: list[str],
               photo_share: float) -> tuple[np.ndarray, str]:
    if photo_names and rng.random() < photo_share:
        name = str(rng.choice(photo_names))
        return random_crop(photo(name), size, rng), name
    return procedural_scene(rng, size), "procedural"


def add_sensor_noise(img: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    noisy = img.astype(np.float32) + rng.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(np.round(noisy), 0, 255).astype(np.uint8)
