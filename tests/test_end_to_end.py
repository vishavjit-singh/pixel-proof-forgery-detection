"""Train on a small benchmark, evaluate on held-out sources, use the API. Loose bounds: it is tiny."""
import io

import numpy as np
from PIL import Image
from starlette.testclient import TestClient

from pixelproof.api import create_app
from pixelproof.benchmark import load
from pixelproof.detector import Detector


def test_metrics_are_complete_and_better_than_chance(tiny):
    _, _, m = tiny
    assert m["test_images"] == 48 and m["tampered"] == 24
    assert m["image_auc"]["fusion"] > 0.6
    assert set(m["by_manipulation"]) == {"splice", "copy_move", "retouch", "resample"}
    assert 0 <= m["pixel_f1"] <= 1 and 0 <= m["authentic_false_positive_area"] <= 1
    assert 0 < m["thresholds"]["pixel"] < 1


def test_detector_round_trip_and_analyze(tiny):
    s, root, _ = tiny
    det = Detector.load(s, root / "models")
    smp = load(root / "bench", "test")[0]
    res = det.analyze(smp.image())
    assert res["verdict"] in ("tampered", "authentic")
    assert res["heatmap"].shape == (256, 256) and res["mask"].dtype == bool


def test_api(tiny):
    s, root, _ = tiny
    client = TestClient(create_app(Detector.load(s, root / "models")))
    assert client.get("/health").json() == {"status": "ok"}
    smp = load(root / "bench", "test")[1]
    r = client.post("/v1/analyze", content=smp.image_path.read_bytes(), headers={"Content-Type": "image/jpeg"})
    body = r.json()
    assert r.status_code == 200 and body["verdict"] in ("tampered", "authentic") and body["width"] == 256
    png = Image.open(io.BytesIO(__import__("base64").b64decode(body["heatmap_png_base64"])))
    assert png.size == (256, 256)
    r = client.post("/v1/analyze", files={"file": ("x.jpg", smp.image_path.read_bytes(), "image/jpeg")},
                    params={"heatmap": "false"})
    assert r.status_code == 200 and "heatmap_png_base64" not in r.json()
    assert client.post("/v1/analyze", content=b"hello", headers={"Content-Type": "text/plain"}).status_code == 415
    assert client.post("/v1/analyze", content=b"not an image", headers={"Content-Type": "image/png"}).status_code == 422
    big = io.BytesIO()
    Image.fromarray(np.zeros((10, 5000, 3), np.uint8)).save(big, format="PNG")
    assert client.post("/v1/analyze", content=big.getvalue(), headers={"Content-Type": "image/png"}).status_code == 413
