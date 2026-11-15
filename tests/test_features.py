import numpy as np

from pixelproof import features
from pixelproof.jpeg import encode, recompress
from pixelproof.sources import add_sensor_noise, procedural_scene


def test_ela_flags_a_region_with_a_different_compression_history():
    rng = np.random.default_rng(0)
    base = recompress(add_sensor_noise(procedural_scene(rng, 128), 2.0, rng), 90)
    donor = recompress(add_sensor_noise(procedural_scene(np.random.default_rng(1), 128), 2.0, rng), 60)
    img = base.copy()
    img[40:88, 40:88] = donor[40:88, 40:88]          # never re-saved at 90
    e = features.ela(img, 90)
    assert e[44:84, 44:84].mean() > 1.3 * e[:32, :32].mean()


def test_residuals_see_smoothing():
    rng = np.random.default_rng(0)
    img = add_sensor_noise(procedural_scene(rng, 128), 3.0, rng)
    import cv2
    img[32:96, 32:96] = cv2.GaussianBlur(img, (0, 0), 3)[32:96, 32:96]
    for r in features.residuals(img):
        assert np.abs(r[40:88, 40:88]).mean() < 0.6 * np.abs(r[:24, :24]).mean()


def test_tensor_shape_and_normalisation():
    img = procedural_scene(np.random.default_rng(3), 64)
    t = features.tensor(img)
    assert t.shape == (len(features.CHANNELS), 64, 64) and t.dtype == np.float32
    assert np.abs(t[1:]).max() <= 1.0


def test_handcrafted_vector():
    img = procedural_scene(np.random.default_rng(3), 128)
    v = features.handcrafted(img, 32)
    assert v.shape == (len(features.HANDCRAFTED),) and np.isfinite(v).all()


def test_jpeg_round_trip_is_deterministic():
    img = procedural_scene(np.random.default_rng(3), 64)
    assert encode(img, 80) == encode(img, 80)
