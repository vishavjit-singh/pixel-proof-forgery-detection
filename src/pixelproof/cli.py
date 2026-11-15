"""``python -m pixelproof <command>``"""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import click
import numpy as np

from .config import ROOT, load_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
DATA, MODELS, EVAL = ROOT / "data" / "benchmark", ROOT / "models", ROOT / "data" / "eval"


@click.group()
@click.option("--config", "config_path", default=None, type=click.Path(exists=True, dir_okay=False))
@click.pass_context
def cli(ctx, config_path):
    """PixelProof: image tampering detection with ELA, noise residuals and a patch CNN."""
    ctx.obj = load_settings(config_path)


@cli.command()
@click.option("--out", default=str(DATA), type=click.Path(file_okay=False))
@click.pass_obj
def generate(s, out):
    """Build the synthetic tampering benchmark (authentic + tampered JPEGs with ground-truth masks)."""
    from .benchmark import generate as gen

    shutil.rmtree(out, ignore_errors=True)
    samples = gen(s, Path(out))
    click.echo(f"wrote {len(samples)} images to {out}")


@cli.command()
@click.option("--data", default=str(DATA), type=click.Path(exists=True, file_okay=False))
@click.option("--models", default=str(MODELS), type=click.Path(file_okay=False))
@click.option("--patch-model", default=None,
              help="Skip CNN training and use this model (e.g. onnx:models/patch_cnn.onnx).")
@click.option("--export-patches", default=str(ROOT / "data" / "patches.npz"),
              help="Where to write the patch dataset (used by tools/train_torch.py).")
@click.pass_obj
def train(s, data, models, patch_model, export_patches):
    """Train the patch CNN (NumPy), the baseline and the fusion classifier; pick thresholds on val."""
    from .pipeline import train as run

    out = run(s, Path(data), Path(models), patch_model, Path(export_patches) if export_patches else None)
    click.echo(json.dumps(out["thresholds"]))


@cli.command()
@click.option("--data", default=str(DATA), type=click.Path(exists=True, file_okay=False))
@click.option("--models", default=str(MODELS), type=click.Path(exists=True, file_okay=False))
@click.option("--out", default=str(EVAL), type=click.Path(file_okay=False))
@click.option("--patch-model", default=None)
@click.pass_obj
def evaluate(s, data, models, out, patch_model):
    """Evaluate on the held-out test split; writes metrics.json and predictions.csv."""
    from .pipeline import evaluate as run

    m = run(s, Path(data), Path(models), Path(out), patch_model)
    click.echo(json.dumps(m, indent=2))


@cli.command()
@click.argument("image", type=click.Path(exists=True, dir_okay=False))
@click.option("--models", default=str(MODELS), type=click.Path(exists=True, file_okay=False))
@click.option("--heatmap", "heatmap_out", default=None, help="Write the tamper heatmap overlay PNG here.")
@click.pass_obj
def analyze(s, image, models, heatmap_out):
    """Analyse one image: verdict, score and (optionally) a heatmap overlay."""
    from .detector import Detector
    from .jpeg import decode
    from .render import overlay_png

    det = Detector.load(s, Path(models))
    img = decode(Path(image).read_bytes())
    res = det.analyze(img)
    if heatmap_out:
        Path(heatmap_out).write_bytes(overlay_png(img, res["heatmap"]))
    click.echo(json.dumps({k: v for k, v in res.items() if not isinstance(v, np.ndarray)}, indent=2))


@cli.command()
@click.option("--eval-dir", default=str(EVAL), type=click.Path(exists=True, file_okay=False))
@click.option("--models", default=str(MODELS), type=click.Path(exists=True, file_okay=False))
@click.option("--out", default=str(ROOT / "docs" / "images"), type=click.Path(file_okay=False))
def report(eval_dir, models, out):
    """Charts for the README from the evaluation outputs."""
    from .report import render_all

    for p in render_all(Path(eval_dir), Path(models), Path(out)):
        click.echo(f"wrote {p}")


@cli.command()
@click.option("--host", default="127.0.0.1")
@click.option("--port", default=8000, type=int)
def serve(host, port):
    """Run the HTTP API (Starlette + uvicorn)."""
    import uvicorn

    uvicorn.run("pixelproof.api:app", host=host, port=port, log_level="info")


def main():
    cli()



if __name__ == "__main__":
    main()
