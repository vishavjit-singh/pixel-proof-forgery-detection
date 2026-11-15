"""Forensic signals: Error Level Analysis, SRM-style noise residuals and JPEG blockiness.

* **ELA**: re-save at a fixed quality and take the absolute difference. Regions with a different
  compression history (pasted, re-rendered, smoothed) re-compress differently from their surroundings.
* **Noise residuals**: high-pass filters from the steganalysis rich model (SRM) suppress content and
  keep the camera noise. Pasted, inpainted or resampled regions carry a different noise "fingerprint".
* **Blockiness**: the 8×8 JPEG grid is strong where the image was compressed in alignment and weak
  where content was moved or resampled.

Maps are normalised per image so the models look at local inconsistency, not at global quality.
"""
from __future__ import annotations

import cv2
import numpy as np

from .jpeg import recompress

SRM_KERNELS = [
    np.array([[0, 0, 0], [1, -2, 1], [0, 0, 0]], np.float32) / 2,                       # 2nd order, horizontal
    np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]], np.float32) / 4,                   # KB (3×3 square)
    np.array([[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2], [2, -6, 8, -6, 2],
              [-1, 2, -2, 2, -1]], np.float32) / 12,                                    # KV (5×5 square)
]
CHANNELS = ("ela", "srm_h2", "srm_kb", "srm_kv", "noise_contrast", "ela_contrast", "grid_contrast")
CONTRAST_WINDOW = 9


def gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)


def ela(img: np.ndarray, quality: int = 90) -> np.ndarray:
    diff = np.abs(img.astype(np.float32) - recompress(img, quality).astype(np.float32))
    return diff.mean(axis=2)


def residuals(img: np.ndarray) -> list[np.ndarray]:
    g = gray(img)
    return [np.clip(cv2.filter2D(g, -1, k, borderType=cv2.BORDER_REFLECT), -8, 8) for k in SRM_KERNELS]


def blockiness(img: np.ndarray) -> np.ndarray:
    """Per-pixel-column/row indicator map: |gradient| on 8×8 block boundaries minus inside."""
    g = gray(img)
    dx = np.abs(np.diff(g, axis=1, append=g[:, -1:]))
    dy = np.abs(np.diff(g, axis=0, append=g[-1:, :]))
    bx = (np.arange(g.shape[1]) % 8 == 7).astype(np.float32)[None, :]
    by = (np.arange(g.shape[0]) % 8 == 7).astype(np.float32)[:, None]
    return dx * (bx * 8 - 1) / 7 + dy * (by * 8 - 1) / 7    # ≈ 0 on average without a grid


def _contrast(local: np.ndarray, eps: float) -> np.ndarray:
    """log(local level / image median level), clipped to ±3 and scaled to ±1."""
    return np.clip(np.log((local + eps) / (np.median(local) + eps)), -3, 3) / 3


def _box(a: np.ndarray, k: int = CONTRAST_WINDOW) -> np.ndarray:
    return cv2.boxFilter(a, -1, (k, k), borderType=cv2.BORDER_REFLECT)


def tensor(img: np.ndarray, ela_quality: int = 90) -> np.ndarray:
    """(7, H, W) float32 input for the patch CNN, normalised per image.

    Channels: ELA, three SRM residuals, and three *contrast* maps that compare the local level of
    noise, ELA and JPEG-grid energy with the image's own median, which is what makes a pasted or
    smoothed region stand out regardless of the image's overall quality.
    """
    e = ela(img, ela_quality)
    rs = residuals(img)
    chans = [np.log1p(e / (np.median(e) + 1.0))]
    for r in rs:
        chans.append(np.clip(r / (r.std() + 1e-3), -4, 4) / 4)
    chans.append(_contrast(np.sqrt(_box(rs[1] ** 2)), 0.05))
    chans.append(_contrast(_box(e), 0.1))
    chans.append(np.clip(_box(blockiness(img)) / (np.abs(_box(blockiness(img))).mean() + 0.1), -3, 3) / 3)
    return np.stack(chans).astype(np.float32)


def _robust(values: np.ndarray) -> list[float]:
    med = np.median(values)
    mad = 1.4826 * np.median(np.abs(values - med)) + 1e-6
    z = (values - med) / mad
    top = np.sort(z)[-3:]
    return [float(z.max()), float(z.min()), float((np.percentile(values, 90) - np.percentile(values, 10)) /
                                                    (abs(med) + 1e-3)), float(top.mean())]


def block_stats(img: np.ndarray, block: int = 32, ela_quality: int = 90) -> np.ndarray:
    """Per-block metrics (rows = blocks): ELA mean, 3 residual stds, blockiness."""
    e = ela(img, ela_quality)
    rs = residuals(img)
    b = blockiness(img)
    h, w = e.shape
    rows = []
    for y in range(0, h - block + 1, block):
        for x in range(0, w - block + 1, block):
            sl = (slice(y, y + block), slice(x, x + block))
            rows.append([e[sl].mean()] + [r[sl].std() for r in rs] + [b[sl].mean()])
    return np.array(rows, np.float32)


HANDCRAFTED = [f"{m}_{s}" for m in ("ela", "srm_h2", "srm_kb", "srm_kv", "blockiness")
               for s in ("zmax", "zmin", "spread", "top3")] + ["ela_median", "noise_median"]


def handcrafted(img: np.ndarray, block: int = 32, ela_quality: int = 90) -> np.ndarray:
    stats = block_stats(img, block, ela_quality)
    feats = []
    for j in range(stats.shape[1]):
        feats += _robust(stats[:, j])
    feats += [float(np.median(stats[:, 0])), float(np.median(stats[:, 2]))]
    return np.array(feats, np.float32)
