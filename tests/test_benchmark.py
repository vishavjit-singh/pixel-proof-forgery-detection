import numpy as np
import pytest

from pixelproof import manipulate
from pixelproof.benchmark import generate, load
from pixelproof.config import load_settings
from pixelproof.sources import EXCLUDED, PHOTOS, photo, procedural_scene


def small(**kw):
    return load_settings(base_images={"train": 8, "val": 4, "test": 4}, **kw)


def test_generation_is_reproducible(tmp_path):
    a = generate(small(), tmp_path / "a")
    b = generate(small(), tmp_path / "b")
    assert [x.sha256 for x in a] == [x.sha256 for x in b]
    c = generate(small(seed=99), tmp_path / "c")
    assert [x.sha256 for x in a] != [x.sha256 for x in c]


def test_manifest_round_trip_and_balance(tmp_path):
    generate(small(), tmp_path)
    samples = load(tmp_path)
    assert len(samples) == 2 * (8 + 4 + 4)
    assert sum(s.label for s in samples) == len(samples) // 2
    t = [s for s in samples if s.label]
    assert {s.manipulation for s in t} == {"splice", "copy_move", "retouch", "resample"}
    for s in t:
        m = s.mask()
        assert m.dtype == bool and 0.01 < m.mean() < 0.3
        assert abs(m.mean() - s.mask_area) < 0.01
    assert all(not s.mask().any() for s in samples if not s.label)


def test_splits_never_share_a_source_photo(tmp_path, settings):
    generate(small(), tmp_path)
    by_split = {}
    for s in load(tmp_path):
        if s.source != "procedural":
            by_split.setdefault(s.split, set()).add(s.source)
    for split, used in by_split.items():
        assert used <= set(settings.sources[split])


def test_no_people_in_the_source_list(settings):
    used = {n for v in settings.sources.values() for n in v}
    assert not used & set(EXCLUDED) and used <= set(PHOTOS)
    with pytest.raises(ValueError):
        photo("astronaut")


@pytest.mark.parametrize("kind", ["copy_move", "retouch", "resample"])
def test_edits_stay_inside_the_mask(kind):
    rng = np.random.default_rng(1)
    img = procedural_scene(np.random.default_rng(5), 128)
    out, mask = getattr(manipulate, kind)(img, rng, (0.05, 0.1))
    changed = np.abs(out.astype(int) - img.astype(int)).max(axis=2) > 2
    halo = np.zeros_like(mask)
    import cv2
    halo = cv2.dilate(mask.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    assert changed[~halo].mean() < 0.001          # only feathering spills, a few pixels wide
    assert changed[mask].mean() > 0.2


def test_splice_brings_foreign_content():
    rng = np.random.default_rng(2)
    img = np.full((96, 96, 3), 40, np.uint8)
    donor = np.full((96, 96, 3), 220, np.uint8)
    out, mask = manipulate.splice(img, donor, rng, (0.08, 0.1))
    assert out[mask].mean() > 150 and out[~mask].mean() < 60
