from pathlib import Path

import pytest

from pixelproof.config import load_settings


@pytest.fixture(scope="session")
def settings():
    return load_settings()


@pytest.fixture(scope="session")
def tiny(tmp_path_factory):
    """A small benchmark + trained detector shared by the end-to-end tests (about 30 s)."""
    from pixelproof.benchmark import generate
    from pixelproof.pipeline import evaluate, train

    root = tmp_path_factory.mktemp("tiny")
    s = load_settings(base_images={"train": 48, "val": 32, "test": 24},
                      cnn={"epochs": 3, "batch_size": 64, "learning_rate": 0.003, "seed": 3})
    generate(s, root / "bench")
    train(s, root / "bench", root / "models")
    metrics = evaluate(s, root / "bench", root / "models", root / "eval")
    return s, Path(root), metrics
