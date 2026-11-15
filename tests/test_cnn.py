import numpy as np
import pytest

from pixelproof.cnn import NumpyCNN, OnnxPatchModel, load_patch_model, sigmoid

SMALL = {"in_channels": 2, "conv": [3, 4, 4], "dense": 5, "patch": 8}


def test_gradients_match_finite_differences():
    m = NumpyCNN(seed=1, arch=SMALL)
    m.params = {k: v.astype(np.float64) for k, v in m.params.items()}
    rng = np.random.default_rng(0)
    x = rng.normal(size=(3, 2, 8, 8))
    y = np.array([1.0, 0.0, 1.0])

    def loss():
        p = sigmoid(m.forward(x).astype(np.float64))
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    logit, cache = m.forward(x, keep=True)
    grads = m.backward((sigmoid(logit) - y) / 3, cache)
    for k, w in m.params.items():
        for _ in range(3):
            idx = tuple(rng.integers(0, s) for s in w.shape)
            w[idx] += 1e-5
            up = loss()
            w[idx] -= 2e-5
            down = loss()
            w[idx] += 1e-5
            assert grads[k][idx] == pytest.approx((up - down) / 2e-5, rel=1e-3, abs=1e-6), k


def test_learns_a_toy_texture_task():
    rng = np.random.default_rng(0)
    n = 400
    y = (rng.random(n) < 0.5).astype(np.float32)
    x = rng.normal(0, 1, (n, 4, 16, 16)).astype(np.float32)
    x[y == 1] *= 0.3                                    # "smoothed" patches have weaker residuals
    m = NumpyCNN(seed=0, arch={"in_channels": 4, "conv": [8, 8, 8], "dense": 8, "patch": 16})
    hist = m.fit(x[:300], y[:300], x[300:], y[300:], epochs=4, batch_size=32, lr=0.01, log=lambda *_: None)
    assert hist[-1]["val_acc"] > 0.9


def test_save_load_round_trip(tmp_path):
    m = NumpyCNN(seed=4)
    x = np.random.default_rng(1).normal(size=(5, 7, 32, 32)).astype(np.float32)
    m.save(tmp_path / "m.npz")
    m2 = load_patch_model(tmp_path / "m.npz")
    np.testing.assert_allclose(m.predict_proba(x), m2.predict_proba(x), rtol=1e-6)


def test_onnx_wrapper_uses_the_same_interface():
    class FakeSession:
        def get_inputs(self):
            return [type("I", (), {"name": "patches"})()]

        def run(self, _, feeds):
            x = feeds["patches"]
            assert x.dtype == np.float32 and x.shape[1:] == (7, 32, 32)
            return [x.mean(axis=(1, 2, 3))[:, None]]

    m = OnnxPatchModel("unused.onnx", session=FakeSession())
    x = np.zeros((300, 7, 32, 32), np.float32)
    p = m.predict_proba(x, batch=128)
    assert p.shape == (300,) and np.allclose(p, 0.5)
