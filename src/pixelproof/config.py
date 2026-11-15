"""Typed view of config/benchmark.yaml."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config" / "benchmark.yaml"
MANIPULATIONS = ("splice", "copy_move", "retouch", "resample")


@dataclass(frozen=True)
class PatchSettings:
    size: int = 32
    stride: int = 16
    positive_min_overlap: float = 0.5
    negative_max_overlap: float = 0.0
    negatives_per_authentic: int = 24
    negatives_per_tampered: int = 8


@dataclass(frozen=True)
class CNNSettings:
    epochs: int = 8
    batch_size: int = 64
    learning_rate: float = 0.002
    seed: int = 10


@dataclass(frozen=True)
class Settings:
    seed: int
    image_size: int
    base_images: dict[str, int]
    sources: dict[str, list[str]]
    photo_share: float
    manipulations: tuple[str, ...]
    first_quality: tuple[int, int]
    final_quality: tuple[int, int]
    donor_quality: tuple[int, int]
    sensor_noise_sigma: tuple[float, float]
    region_area: tuple[float, float]
    ela_quality: int
    block: int
    patch: PatchSettings = field(default_factory=PatchSettings)
    cnn: CNNSettings = field(default_factory=CNNSettings)


def load_settings(path: str | Path | None = None, **overrides) -> Settings:
    raw = yaml.safe_load(Path(path or DEFAULT_CONFIG).read_text())
    raw.update(overrides)
    bad = set(raw["manipulations"]) - set(MANIPULATIONS)
    if bad:
        raise ValueError(f"unknown manipulations: {sorted(bad)}")
    src = raw["sources"]
    splits = {k: set(v) for k, v in src.items() if k in ("train", "val", "test")}
    if splits["train"] & splits["test"] or splits["train"] & splits["val"] or splits["val"] & splits["test"]:
        raise ValueError("a source photo may belong to one split only")
    j = raw["jpeg"]
    return Settings(
        seed=int(raw["seed"]), image_size=int(raw["image_size"]),
        base_images={k: int(v) for k, v in raw["base_images"].items()},
        sources={k: list(src[k]) for k in ("train", "val", "test")}, photo_share=float(src["photo_share"]),
        manipulations=tuple(raw["manipulations"]),
        first_quality=tuple(j["first_quality"]), final_quality=tuple(j["final_quality"]),
        donor_quality=tuple(j["donor_quality"]),
        sensor_noise_sigma=tuple(float(x) for x in raw["sensor_noise_sigma"]),
        region_area=tuple(float(x) for x in raw["region_area"]),
        ela_quality=int(raw["features"]["ela_quality"]), block=int(raw["features"]["block"]),
        patch=PatchSettings(**raw.get("patch", {})), cnn=CNNSettings(**raw.get("cnn", {})),
    )
