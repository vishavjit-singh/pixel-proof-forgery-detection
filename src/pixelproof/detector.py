"""Patch extraction, heatmaps and the full detector (patch CNN + image-level classifiers)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np

from . import features
from .cnn import NumpyCNN, load_patch_model
from .config import Settings

HEAT_FEATURES = ["heat_max", "heat_p98", "heat_mean", "heat_std", "heat_area50", "heat_area80"]


# ------------------------------------------------------------------------------------------- patches
def positions(length: int, size: int, stride: int) -> list[int]:
    if length <= size:
        return [0]
    pos = list(range(0, length - size + 1, stride))
    if pos[-1] != length - size:
        pos.append(length - size)                 # cover the right/bottom edge
    return pos


def _pad(t: np.ndarray, size: int) -> np.ndarray:
    _, h, w = t.shape
    if h >= size and w >= size:
        return t
    return np.pad(t, ((0, 0), (0, max(0, size - h)), (0, max(0, size - w))), mode="reflect")


def grid_patches(t: np.ndarray, size: int, stride: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    t = _pad(t, size)
    _, h, w = t.shape
    coords = [(y, x) for y in positions(h, size, stride) for x in positions(w, size, stride)]
    return np.stack([t[:, y:y + size, x:x + size] for y, x in coords]), coords


def training_patches(t: np.ndarray, mask: np.ndarray, label: int, s: Settings, rng: np.random.Generator,
                     max_pos: int = 24) -> tuple[np.ndarray, np.ndarray]:
    p = s.patch
    patches, coords = grid_patches(t, p.size, p.stride)
    overlap = np.array([mask[y:y + p.size, x:x + p.size].mean() for y, x in coords])
    pos = np.nonzero(overlap >= p.positive_min_overlap)[0] if label else np.array([], int)
    neg = np.nonzero(overlap <= p.negative_max_overlap)[0]
    if len(pos) > max_pos:
        pos = rng.choice(pos, max_pos, replace=False)
    n_neg = p.negatives_per_tampered if label else p.negatives_per_authentic
    neg = rng.choice(neg, min(n_neg, len(neg)), replace=False) if len(neg) else neg
    idx = np.concatenate([pos, neg]).astype(int)
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))]).astype(np.float32)
    return patches[idx].astype(np.float16), y


def heatmap_from_scores(scores: np.ndarray, coords: list[tuple[int, int]], shape: tuple[int, int],
                        size: int) -> np.ndarray:
    h, w = max(shape[0], size), max(shape[1], size)
    acc = np.zeros((h, w), np.float32)
    cnt = np.zeros((h, w), np.float32)
    for sc, (y, x) in zip(scores, coords):
        acc[y:y + size, x:x + size] += sc
        cnt[y:y + size, x:x + size] += 1
    return (acc / np.maximum(cnt, 1))[:shape[0], :shape[1]]


def heat_stats(heat: np.ndarray) -> np.ndarray:
    return np.array([heat.max(), np.percentile(heat, 98), heat.mean(), heat.std(), (heat > 0.5).mean(),
                     (heat > 0.8).mean()], np.float32)


# -------------------------------------------------------------------------------------------- metrics
def pixel_scores(pred: np.ndarray, truth: np.ndarray) -> dict:
    tp = float((pred & truth).sum())
    fp = float((pred & ~truth).sum())
    fn = float((~pred & truth).sum())
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 1.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 1.0
    return {"f1": f1, "iou": iou}


# ------------------------------------------------------------------------------------------- detector
class Detector:
    def __init__(self, settings: Settings, patch_model, baseline=None, fusion=None, thresholds: dict | None = None):
        self.s = settings
        self.patch_model = patch_model
        self.baseline = baseline          # sklearn classifier on the handcrafted ELA/noise statistics
        self.fusion = fusion              # logistic regression on [baseline score, heatmap statistics]
        self.thresholds = thresholds or {"image": 0.5, "pixel": 0.5, "baseline": 0.5}

    def heatmap(self, img: np.ndarray, t: np.ndarray | None = None) -> np.ndarray:
        t = features.tensor(img, self.s.ela_quality) if t is None else t
        patches, coords = grid_patches(t, self.s.patch.size, self.s.patch.stride)
        scores = self.patch_model.predict_proba(patches.astype(np.float32))
        return heatmap_from_scores(scores, coords, img.shape[:2], self.s.patch.size)

    def scores(self, img: np.ndarray) -> dict:
        heat = self.heatmap(img)
        hand = features.handcrafted(img, self.s.block, self.s.ela_quality)
        base = float(self.baseline.predict_proba(hand[None])[0, 1]) if self.baseline is not None else float("nan")
        fused = float(self.fusion.predict_proba(np.r_[base, heat_stats(heat)][None])[0, 1]) \
            if self.fusion is not None else float(heat.max())
        return {"heat": heat, "baseline": base, "cnn_max": float(heat.max()), "fusion": fused}

    def analyze(self, img: np.ndarray) -> dict:
        t0 = time.perf_counter()
        sc = self.scores(img)
        mask = sc["heat"] >= self.thresholds["pixel"]
        tampered = sc["fusion"] >= self.thresholds["image"]
        return {"verdict": "tampered" if tampered else "authentic", "score": round(sc["fusion"], 4),
                "threshold": self.thresholds["image"], "baseline_score": round(sc["baseline"], 4),
                "max_patch_score": round(sc["cnn_max"], 4),
                "suspicious_area": round(float(mask.mean()) if tampered else 0.0, 4),
                "heatmap": sc["heat"], "mask": mask & tampered,
                "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1)}

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if isinstance(self.patch_model, NumpyCNN):
            self.patch_model.save(path / "patch_cnn.npz")
        joblib.dump({"baseline": self.baseline, "fusion": self.fusion}, path / "classifiers.joblib")
        (path / "thresholds.json").write_text(json.dumps(self.thresholds, indent=2))

    @classmethod
    def load(cls, settings: Settings, path: str | Path, patch_model: str | None = None) -> Detector:
        path = Path(path)
        clfs = joblib.load(path / "classifiers.joblib")
        model = load_patch_model(patch_model or path / "patch_cnn.npz")
        thresholds = json.loads((path / "thresholds.json").read_text())
        return cls(settings, model, clfs["baseline"], clfs["fusion"], thresholds)
