"""JPEG helpers (Pillow, baseline JPEG, 4:2:0 chroma subsampling)."""
from __future__ import annotations

import io

import numpy as np
from PIL import Image


def encode(img: np.ndarray, quality: int) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=int(quality))
    return buf.getvalue()


def decode(data: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(data)) as im:
        return np.asarray(im.convert("RGB")).copy()


def recompress(img: np.ndarray, quality: int) -> np.ndarray:
    return decode(encode(img, quality))
