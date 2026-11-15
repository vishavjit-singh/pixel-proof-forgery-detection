"""The four manipulations. Each returns (tampered image, ground-truth mask as bool HxW)."""
from __future__ import annotations

import cv2
import numpy as np


def region_mask(rng: np.random.Generator, size: int, area: tuple[float, float],
                center: tuple[int, int] | None = None) -> np.ndarray:
    """A random blob (ellipse or polygon) covering roughly ``area`` of the image."""
    target = rng.uniform(*area) * size * size
    mask = np.zeros((size, size), np.uint8)
    r = np.sqrt(target / np.pi)
    cx, cy = center if center else (int(rng.integers(int(r) + 4, size - int(r) - 4)),
                                    int(rng.integers(int(r) + 4, size - int(r) - 4)))
    if rng.random() < 0.5:
        ratio = rng.uniform(0.6, 1.6)
        axes = (max(4, int(r * ratio)), max(4, int(r / ratio)))
        cv2.ellipse(mask, (cx, cy), axes, float(rng.uniform(0, 180)), 0, 360, 1, -1)
    else:
        n = int(rng.integers(5, 9))
        ang = np.linspace(0, 2 * np.pi, n, endpoint=False) + rng.uniform(-0.3, 0.3, n) * (2 * np.pi / n)
        rad = r * rng.uniform(0.75, 1.3, n)
        pts = np.stack([cx + rad * np.cos(ang), cy + rad * np.sin(ang)], 1).astype(np.int32)
        cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


def _blend(base: np.ndarray, patch: np.ndarray, mask: np.ndarray, feather: float) -> tuple[np.ndarray, np.ndarray]:
    alpha = mask.astype(np.float32)
    if feather > 0:
        alpha = cv2.GaussianBlur(alpha, (0, 0), feather)
    out = base.astype(np.float32) * (1 - alpha[..., None]) + patch.astype(np.float32) * alpha[..., None]
    return np.clip(np.round(out), 0, 255).astype(np.uint8), alpha > 0.5


def splice(img: np.ndarray, donor: np.ndarray, rng: np.random.Generator, area) -> tuple[np.ndarray, np.ndarray]:
    """Paste a region from another image (different content, noise and compression history)."""
    size = img.shape[0]
    mask = region_mask(rng, size, area)
    ys, xs = np.nonzero(mask)
    dy = int(rng.integers(-ys.min(), size - ys.max()))
    dx = int(rng.integers(-xs.min(), size - xs.max()))
    shifted = np.roll(np.roll(donor, dy, axis=0), dx, axis=1)       # donor content lands in the mask
    return _blend(img, shifted, mask, rng.uniform(0.5, 1.5))


def copy_move(img: np.ndarray, rng: np.random.Generator, area) -> tuple[np.ndarray, np.ndarray]:
    """Clone a region of the same image to another place (same noise and compression: the hard case)."""
    size = img.shape[0]
    for _ in range(50):
        src = region_mask(rng, size, area)
        ys, xs = np.nonzero(src)
        dy = int(rng.integers(-ys.min(), size - ys.max()))
        dx = int(rng.integers(-xs.min(), size - xs.max()))
        dst = np.roll(np.roll(src, dy, 0), dx, 1)
        if not (dst & src).any() and abs(dy) + abs(dx) > 24:
            break
    moved = np.roll(np.roll(img, dy, 0), dx, 1)
    return _blend(img, moved, dst, rng.uniform(0.5, 1.2))


def retouch(img: np.ndarray, rng: np.random.Generator, area) -> tuple[np.ndarray, np.ndarray]:
    """Remove or smooth an object: inpainting or heavy smoothing wipes out the local noise."""
    size = img.shape[0]
    mask = region_mask(rng, size, area)
    if rng.random() < 0.5:
        filled = cv2.inpaint(img, mask.astype(np.uint8), 5, cv2.INPAINT_TELEA)
    else:
        filled = cv2.bilateralFilter(cv2.GaussianBlur(img, (0, 0), rng.uniform(2.0, 4.0)), 9, 40, 9)
    return _blend(img, filled, mask, rng.uniform(0.5, 1.2))


def resample(img: np.ndarray, rng: np.random.Generator, area) -> tuple[np.ndarray, np.ndarray]:
    """Enlarge or shrink a region in place (interpolation changes its noise spectrum)."""
    size = img.shape[0]
    mask = region_mask(rng, size, area)
    ys, xs = np.nonzero(mask)
    cy, cx = float(ys.mean()), float(xs.mean())
    scale = float(rng.choice([rng.uniform(0.7, 0.88), rng.uniform(1.15, 1.5)]))
    angle = float(rng.uniform(-8, 8))
    m = cv2.getRotationMatrix2D((cx, cy), angle, scale)
    warped = cv2.warpAffine(img, m, (size, size), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    return _blend(img, warped, mask, rng.uniform(0.5, 1.2))
