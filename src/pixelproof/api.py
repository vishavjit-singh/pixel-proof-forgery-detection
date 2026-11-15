"""HTTP API (Starlette; the same handlers would port to FastAPI one-to-one).

POST /v1/analyze   body: the image bytes (image/jpeg, image/png) or multipart form field "file"
GET  /health
"""
from __future__ import annotations

import base64
import os
from pathlib import Path

import numpy as np
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from .config import ROOT, load_settings
from .jpeg import decode
from .render import overlay_png

MAX_BYTES = 10 * 1024 * 1024
MAX_SIDE = 4096


def _detector(request: Request):
    det = getattr(request.app.state, "detector", None)
    if det is None:
        from .detector import Detector

        det = Detector.load(load_settings(), Path(os.environ.get("PIXELPROOF_MODELS", ROOT / "models")))
        request.app.state.detector = det
    return det


async def health(request: Request):
    return JSONResponse({"status": "ok"})


async def analyze(request: Request):
    ctype = request.headers.get("content-type", "")
    if ctype.startswith("multipart/form-data"):
        form = await request.form()
        upload = form.get("file")
        if upload is None or isinstance(upload, str):
            return JSONResponse({"error": "multipart field 'file' is required"}, status_code=422)
        data = await upload.read()
    elif ctype.startswith("image/"):
        data = await request.body()
    else:
        return JSONResponse({"error": "send image/jpeg, image/png or multipart/form-data"}, status_code=415)
    if not data:
        return JSONResponse({"error": "empty body"}, status_code=422)
    if len(data) > MAX_BYTES:
        return JSONResponse({"error": f"image larger than {MAX_BYTES} bytes"}, status_code=413)
    try:
        img = decode(data)
    except Exception:
        return JSONResponse({"error": "not a readable image"}, status_code=422)
    if max(img.shape[:2]) > MAX_SIDE:
        return JSONResponse({"error": f"image side larger than {MAX_SIDE}px"}, status_code=413)
    res = _detector(request).analyze(img)
    body = {k: v for k, v in res.items() if not isinstance(v, np.ndarray)}
    body["width"], body["height"] = int(img.shape[1]), int(img.shape[0])
    if request.query_params.get("heatmap", "true") != "false":
        body["heatmap_png_base64"] = base64.b64encode(overlay_png(img, res["heatmap"])).decode()
    return JSONResponse(body)


def create_app(detector=None) -> Starlette:
    app = Starlette(routes=[Route("/health", health), Route("/v1/analyze", analyze, methods=["POST"])])
    app.state.detector = detector
    return app


app = create_app()
