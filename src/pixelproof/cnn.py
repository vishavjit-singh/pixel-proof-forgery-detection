"""A small patch CNN in NumPy (forward, backward, Adam), plus an ONNX Runtime wrapper with the same interface.

Architecture (input N×7×32×32: ELA, three noise residuals, three local-contrast maps):

    conv3×3(7→16) → ReLU → maxpool2 → conv3×3(16→32) → ReLU → maxpool2 → conv3×3(32→32) → ReLU
    → global average pool → dense(32→16) → ReLU → dense(16→1) → logit

About 16k parameters. PyTorch is not needed to train or run it; ``tools/train_torch.py`` builds the
same network in PyTorch and exports ONNX, which :class:`OnnxPatchModel` runs through ONNX Runtime.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ARCH = {"in_channels": 7, "conv": [16, 32, 32], "dense": 16, "patch": 32}


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


# ----------------------------------------------------------------------------------------- primitives
def conv_forward(x: np.ndarray, w: np.ndarray, b: np.ndarray):
    """x: N,H,W,C · w: 3,3,C,F → N,H,W,F (stride 1, 'same' padding)."""
    n, h, wd, c = x.shape
    xp = np.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)))
    cols = np.stack([xp[:, i:i + h, j:j + wd, :] for i in range(3) for j in range(3)], axis=3)  # N,H,W,9,C
    out = cols.reshape(n * h * wd, 9 * c) @ w.reshape(9 * c, -1) + b
    return out.reshape(n, h, wd, -1), cols


def conv_backward(dout: np.ndarray, cols: np.ndarray, w: np.ndarray):
    n, h, wd, f = dout.shape
    c = w.shape[2]
    d2 = dout.reshape(-1, f)
    dw = (cols.reshape(-1, 9 * c).T @ d2).reshape(w.shape)
    db = d2.sum(axis=0)
    dcols = (d2 @ w.reshape(9 * c, f).T).reshape(n, h, wd, 9, c)
    dxp = np.zeros((n, h + 2, wd + 2, c), dout.dtype)
    k = 0
    for i in range(3):
        for j in range(3):
            dxp[:, i:i + h, j:j + wd, :] += dcols[:, :, :, k, :]
            k += 1
    return dxp[:, 1:-1, 1:-1, :], dw, db


def pool_forward(x: np.ndarray):
    n, h, w, c = x.shape
    r = x.reshape(n, h // 2, 2, w // 2, 2, c)
    out = r.max(axis=(2, 4))
    mask = r == out[:, :, None, :, None, :]
    return out, mask


def pool_backward(dout: np.ndarray, mask: np.ndarray):
    n, h2, w2, c = dout.shape
    m = mask / np.maximum(mask.sum(axis=(2, 4), keepdims=True), 1)
    return (m * dout[:, :, None, :, None, :]).reshape(n, h2 * 2, w2 * 2, c)


# --------------------------------------------------------------------------------------------- model
class NumpyCNN:
    def __init__(self, seed: int = 0, arch: dict | None = None):
        self.arch = dict(arch or ARCH)
        rng = np.random.default_rng(seed)
        chans = [self.arch["in_channels"]] + self.arch["conv"]
        self.params: dict[str, np.ndarray] = {}
        for i in range(3):
            fan_in = 9 * chans[i]
            self.params[f"w{i}"] = (rng.normal(0, np.sqrt(2 / fan_in), (3, 3, chans[i], chans[i + 1]))
                                    .astype(np.float32))
            self.params[f"b{i}"] = np.zeros(chans[i + 1], np.float32)
        d = self.arch["dense"]
        self.params["w3"] = rng.normal(0, np.sqrt(2 / chans[-1]), (chans[-1], d)).astype(np.float32)
        self.params["b3"] = np.zeros(d, np.float32)
        self.params["w4"] = rng.normal(0, np.sqrt(1 / d), (d, 1)).astype(np.float32)
        self.params["b4"] = np.zeros(1, np.float32)

    # x is N,C,H,W (the ONNX/PyTorch layout); internally NHWC
    def forward(self, x: np.ndarray, keep: bool = False):
        p = self.params
        a = np.ascontiguousarray(x.transpose(0, 2, 3, 1), dtype=np.float32)
        cache = {}
        for i in range(3):
            z, cols = conv_forward(a, p[f"w{i}"], p[f"b{i}"])
            r = np.maximum(z, 0)
            cache[f"conv{i}"] = (cols, z)
            if i < 2:
                a, m = pool_forward(r)
                cache[f"pool{i}"] = m
            else:
                a = r
        g = a.mean(axis=(1, 2))
        h = g @ p["w3"] + p["b3"]
        hr = np.maximum(h, 0)
        logit = (hr @ p["w4"] + p["b4"])[:, 0]
        if keep:
            cache.update(last=a, g=g, h=h, hr=hr)
            return logit, cache
        return logit

    def backward(self, dlogit: np.ndarray, cache: dict) -> dict[str, np.ndarray]:
        p, grads = self.params, {}
        d = dlogit[:, None].astype(np.float32)
        grads["w4"] = cache["hr"].T @ d
        grads["b4"] = d.sum(0)
        dh = (d @ p["w4"].T) * (cache["h"] > 0)
        grads["w3"] = cache["g"].T @ dh
        grads["b3"] = dh.sum(0)
        dg = dh @ p["w3"].T
        last = cache["last"]
        da = np.broadcast_to(dg[:, None, None, :] / (last.shape[1] * last.shape[2]), last.shape).copy()
        for i in (2, 1, 0):
            if i < 2:
                da = pool_backward(da, cache[f"pool{i}"])
            cols, z = cache[f"conv{i}"]
            dz = da * (z > 0)
            da, grads[f"w{i}"], grads[f"b{i}"] = conv_backward(dz, cols, p[f"w{i}"])
        return grads

    def predict_proba(self, x: np.ndarray, batch: int = 256) -> np.ndarray:
        return np.concatenate([sigmoid(self.forward(x[i:i + batch])) for i in range(0, len(x), batch)]) \
            if len(x) else np.zeros(0, np.float32)

    # ------------------------------------------------------------------------------------- training
    def fit(self, x: np.ndarray, y: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, epochs: int = 8,
            batch_size: int = 64, lr: float = 0.002, seed: int = 0, log=print) -> list[dict]:
        rng = np.random.default_rng(seed)
        pos_w = float((y == 0).sum() / max((y == 1).sum(), 1))      # balance the classes in the loss
        m = {k: np.zeros_like(v) for k, v in self.params.items()}
        v = {k: np.zeros_like(v) for k, v in self.params.items()}
        b1, b2, step, history, best = 0.9, 0.999, 0, [], (np.inf, None)
        for epoch in range(1, epochs + 1):
            order = rng.permutation(len(x))
            losses, correct = [], 0
            for s in range(0, len(x), batch_size):
                idx = order[s:s + batch_size]
                xb = x[idx]
                if rng.random() < 0.5:                       # augmentation: flips keep the forensic signal
                    xb = xb[:, :, :, ::-1]
                if rng.random() < 0.5:
                    xb = xb[:, :, ::-1, :]
                logit, cache = self.forward(xb, keep=True)
                prob = sigmoid(logit)
                yb = y[idx]
                wts = np.where(yb == 1, pos_w, 1.0)
                loss = -np.mean(wts * (yb * np.log(prob + 1e-7) + (1 - yb) * np.log(1 - prob + 1e-7)))
                grads = self.backward((wts * (prob - yb) / len(idx)).astype(np.float32), cache)
                step += 1
                cur_lr = lr * (0.5 * (1 + np.cos(np.pi * (epoch - 1) / epochs)))
                for k in self.params:
                    m[k] = b1 * m[k] + (1 - b1) * grads[k]
                    v[k] = b2 * v[k] + (1 - b2) * grads[k] ** 2
                    mh, vh = m[k] / (1 - b1 ** step), v[k] / (1 - b2 ** step)
                    self.params[k] -= (cur_lr * mh / (np.sqrt(vh) + 1e-8)).astype(np.float32)
                losses.append(loss)
                correct += int(((prob > 0.5) == (yb == 1)).sum())
            pv = self.predict_proba(x_val)
            val_loss = float(-np.mean(y_val * np.log(pv + 1e-7) + (1 - y_val) * np.log(1 - pv + 1e-7)))
            rec = {"epoch": epoch, "train_loss": float(np.mean(losses)), "train_acc": correct / len(x),
                   "val_loss": val_loss, "val_acc": float(((pv > 0.5) == (y_val == 1)).mean())}
            history.append(rec)
            log(f"epoch {epoch}/{epochs} loss {rec['train_loss']:.4f} acc {rec['train_acc']:.3f} · "
                f"val loss {val_loss:.4f} acc {rec['val_acc']:.3f}")
            if val_loss < best[0]:
                best = (val_loss, {k: p.copy() for k, p in self.params.items()})
        if best[1] is not None:
            self.params = best[1]                            # keep the epoch with the best validation loss
        return history

    # ------------------------------------------------------------------------------------- persistence
    def save(self, path: str | Path) -> None:
        np.savez(path, arch=json.dumps(self.arch), **self.params)

    @classmethod
    def load(cls, path: str | Path) -> NumpyCNN:
        data = np.load(path)
        model = cls(arch=json.loads(str(data["arch"])))
        model.params = {k: data[k].astype(np.float32) for k in data.files if k != "arch"}
        return model


class OnnxPatchModel:
    """Same interface as NumpyCNN.predict_proba for a network exported to ONNX (input N×7×32×32 → logit)."""

    def __init__(self, path: str | Path, session=None):
        if session is None:
            import onnxruntime as ort

            session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.session = session
        self.input_name = session.get_inputs()[0].name

    def predict_proba(self, x: np.ndarray, batch: int = 256) -> np.ndarray:
        outs = []
        for i in range(0, len(x), batch):
            logit = self.session.run(None, {self.input_name: np.ascontiguousarray(x[i:i + batch], np.float32)})[0]
            outs.append(sigmoid(np.asarray(logit).reshape(-1)))
        return np.concatenate(outs) if outs else np.zeros(0, np.float32)


def load_patch_model(spec: str | Path):
    """``models/patch_cnn.npz`` (NumPy) or ``onnx:models/patch_cnn.onnx``."""
    spec = str(spec)
    if spec.startswith("onnx:"):
        return OnnxPatchModel(spec[5:])
    return NumpyCNN.load(spec)
