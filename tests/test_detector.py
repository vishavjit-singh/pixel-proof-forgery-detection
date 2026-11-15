import numpy as np

from pixelproof.detector import grid_patches, heatmap_from_scores, pixel_scores, positions


def test_positions_cover_the_edges():
    assert positions(256, 32, 16)[-1] == 224
    assert positions(100, 32, 16) == [0, 16, 32, 48, 64, 68]
    assert positions(20, 32, 16) == [0]


def test_heatmap_aggregation_averages_overlaps():
    t = np.zeros((7, 64, 64), np.float32)
    patches, coords = grid_patches(t, 32, 16)
    assert patches.shape == (9, 7, 32, 32)
    scores = np.array([1.0 if (y, x) == (0, 0) else 0.0 for y, x in coords])
    heat = heatmap_from_scores(scores, coords, (64, 64), 32)
    assert heat[0, 0] == 1.0 and heat[20, 20] == 0.25 and heat[60, 60] == 0.0


def test_small_images_are_padded():
    patches, coords = grid_patches(np.ones((4, 20, 50), np.float32), 32, 16)
    assert patches.shape[2:] == (32, 32)
    heat = heatmap_from_scores(np.ones(len(coords)), coords, (20, 50), 32)
    assert heat.shape == (20, 50)


def test_pixel_scores():
    t = np.zeros((10, 10), bool)
    t[:5] = True
    p = np.zeros((10, 10), bool)
    p[:5, :5] = True
    s = pixel_scores(p, t)
    assert s["iou"] == 0.5 and round(s["f1"], 4) == round(2 * 25 / (2 * 25 + 25), 4)
    assert pixel_scores(np.zeros_like(t), np.zeros_like(t)) == {"f1": 1.0, "iou": 1.0}
