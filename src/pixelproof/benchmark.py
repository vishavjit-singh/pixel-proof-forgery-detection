"""Reproducible synthetic tampering benchmark.

For every base image the generator simulates a camera (sensor noise + first JPEG compression), then
writes two samples to the same split:

* **authentic**: the camera image re-saved once more (as a messaging app or CMS would)
* **tampered**: the camera image edited with one of the manipulations, then re-saved

Both are double-compressed, so a detector cannot just look for "was this JPEG saved twice" and has
to find *local* inconsistencies. Splits never share a source photo.
"""
from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import manipulate
from .config import Settings
from .jpeg import decode, encode
from .sources import add_sensor_noise, base_image

SPLITS = ("train", "val", "test")
FIELDS = ["id", "split", "label", "manipulation", "source", "first_quality", "final_quality", "donor_quality",
          "noise_sigma", "mask_area", "sha256"]


@dataclass
class Sample:
    id: str
    split: str
    label: int
    manipulation: str
    source: str
    first_quality: int
    final_quality: int
    donor_quality: int
    noise_sigma: float
    mask_area: float
    sha256: str
    root: Path

    @property
    def image_path(self) -> Path:
        return self.root / self.split / "images" / f"{self.id}.jpg"

    @property
    def mask_path(self) -> Path:
        return self.root / self.split / "masks" / f"{self.id}.png"

    def image(self) -> np.ndarray:
        return decode(self.image_path.read_bytes())

    def mask(self) -> np.ndarray:
        if not self.label:
            img = self.image()
            return np.zeros(img.shape[:2], bool)
        return cv2.imread(str(self.mask_path), cv2.IMREAD_GRAYSCALE) > 127


def _camera(rng, s: Settings, photos: list[str], quality_range) -> tuple[np.ndarray, str, int, float]:
    img, source = base_image(rng, s.image_size, photos, s.photo_share)
    sigma = float(rng.uniform(*s.sensor_noise_sigma))
    q = int(rng.integers(quality_range[0], quality_range[1] + 1))
    return decode(encode(add_sensor_noise(img, sigma, rng), q)), source, q, sigma


def generate(s: Settings, root: Path) -> list[Sample]:
    root = Path(root)
    samples: list[Sample] = []
    for si, split in enumerate(SPLITS):
        (root / split / "images").mkdir(parents=True, exist_ok=True)
        (root / split / "masks").mkdir(parents=True, exist_ok=True)
        for i in range(s.base_images[split]):
            rng = np.random.default_rng([s.seed, si, i])
            camera, source, q1, sigma = _camera(rng, s, s.sources[split], s.first_quality)
            kind = s.manipulations[i % len(s.manipulations)]
            donor_q = 0
            if kind == "splice":
                donor, _, donor_q, _ = _camera(rng, s, s.sources[split], s.donor_quality)
                tampered, mask = manipulate.splice(camera, donor, rng, s.region_area)
            else:
                tampered, mask = getattr(manipulate, kind)(camera, rng, s.region_area)
            for label, img, m in ((0, camera, None), (1, tampered, mask)):
                q2 = int(rng.integers(s.final_quality[0], s.final_quality[1] + 1))
                data = encode(img, q2)
                sid = f"{split}-{i:04d}-{'t' if label else 'a'}"
                (root / split / "images" / f"{sid}.jpg").write_bytes(data)
                if label:
                    cv2.imwrite(str(root / split / "masks" / f"{sid}.png"), (m * 255).astype(np.uint8))
                samples.append(Sample(sid, split, label, kind if label else "authentic", source, q1, q2,
                                      donor_q if label else 0, round(sigma, 3),
                                      round(float(m.mean()), 4) if label else 0.0,
                                      hashlib.sha256(data).hexdigest(), root))
    with (root / "manifest.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for smp in samples:
            w.writerow({k: getattr(smp, k) for k in FIELDS})
    return samples


def load(root: Path, split: str | None = None) -> list[Sample]:
    root = Path(root)
    out = []
    with (root / "manifest.csv").open() as f:
        for r in csv.DictReader(f):
            if split and r["split"] != split:
                continue
            out.append(Sample(r["id"], r["split"], int(r["label"]), r["manipulation"], r["source"],
                              int(r["first_quality"]), int(r["final_quality"]), int(r["donor_quality"]),
                              float(r["noise_sigma"]), float(r["mask_area"]), r["sha256"], root))
    return out
