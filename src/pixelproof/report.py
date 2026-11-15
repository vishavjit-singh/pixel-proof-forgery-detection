"""README images, all rendered from the evaluation outputs (synthetic benchmark, no people)."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

from .render import overlay  # noqa: E402

INK, MUTED, GRID = "#0e1a2b", "#5b6676", "#e3e7ee"
ACCENT, SIGNAL, BAD, GOOD, PURPLE = "#2a78d6", "#eda100", "#e34948", "#1baf7a", "#6250d6"
KINDS = ("splice", "copy_move", "retouch", "resample")

plt.rcParams.update({
    "font.family": "DejaVu Sans", "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.titleweight": "bold", "axes.titlesize": 13, "axes.titlecolor": INK,
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
})


def _save(fig, out: Path, name: str) -> Path:
    fig.tight_layout()
    p = out / name
    fig.savefig(p, facecolor="white")
    plt.close(fig)
    return p


def architecture(out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(13, 5.4))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5.8)
    ax.axis("off")

    def box(x, y, w, h, title, sub, color=INK):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc="white",
                                    ec=color, lw=2))
        ax.text(x + w / 2, y + h * 0.7, title, ha="center", va="center", fontsize=11, weight="bold", color=color)
        ax.text(x + w / 2, y + h * 0.32, sub, ha="center", va="center", fontsize=8.2, color=MUTED, linespacing=1.4)

    def arrow(x1, y1, x2, y2, color=MUTED):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, lw=1.6, color=color))

    ax.text(0.15, 5.5, "pixel-proof-forgery-detection: forensic signals → patch CNN → heatmap → verdict",
            fontsize=13.5, weight="bold", color=INK)
    box(0.2, 2.3, 2.1, 1.4, "JPEG image", "any size\n(API, CLI, batch)")
    box(2.8, 3.55, 2.5, 1.35, "Error Level Analysis", "re-save at q=90, |diff|\nnormalised per image", ACCENT)
    box(2.8, 1.15, 2.5, 1.35, "Noise residuals", "3 SRM high-pass filters\nnormalised per image", ACCENT)
    box(5.8, 3.55, 2.5, 1.35, "Patch CNN", "32×32 patches, stride 16\nNumPy (or ONNX Runtime)", PURPLE)
    box(5.8, 1.15, 2.5, 1.35, "Block statistics", "ELA · residual energy ·\n8×8 blockiness, robust z", ACCENT)
    box(8.8, 3.55, 2.0, 1.35, "Heatmap", "overlapping patch\nscores → pixels", SIGNAL)
    box(8.8, 1.15, 2.0, 1.35, "Baseline", "gradient boosting\non block statistics", INK)
    box(11.2, 2.3, 1.6, 1.4, "Verdict", "fusion LR\n+ mask", GOOD)
    arrow(2.3, 3.2, 2.8, 4.0)
    arrow(2.3, 2.8, 2.8, 2.0)
    arrow(5.3, 4.22, 5.8, 4.22)
    arrow(5.3, 1.82, 5.8, 1.82)
    arrow(5.3, 2.2, 5.8, 3.7)
    arrow(5.3, 3.8, 5.8, 2.3)
    arrow(8.3, 4.22, 8.8, 4.22)
    arrow(8.3, 1.82, 8.8, 1.82)
    arrow(10.8, 4.0, 11.2, 3.4)
    arrow(10.8, 2.0, 11.2, 2.6)
    ax.text(0.2, 0.6, "Trained and measured on a synthetic benchmark: splice · copy-move · retouch · resample, "
            "double JPEG compression, source photos split by train/val/test. Fusion and thresholds are fitted "
            "on val only.", fontsize=8.5, color=MUTED)
    return _save(fig, out, "architecture.png")


def _norm(a: np.ndarray) -> np.ndarray:
    hi = np.percentile(a, 99.5) or 1
    return np.clip(a / hi, 0, 1)


def examples(eval_dir: Path, thresholds: dict, out: Path) -> Path:
    data = np.load(eval_dir / "examples.npz")
    meta = json.loads(str(data["meta"]))
    kinds = [k for k in KINDS + ("authentic",) if k in meta]
    fig, axes = plt.subplots(len(kinds), 5, figsize=(13, 2.75 * len(kinds)))
    titles = ["image (as tested)", "ELA (q=90)", "patch-CNN heatmap", "predicted mask", "ground truth"]
    for r, k in enumerate(kinds):
        img, ela, heat, mask = (data[f"{k}__{f}"] for f in ("image", "ela", "heat", "mask"))
        pred = heat >= thresholds["pixel"]
        panels = [img, _norm(ela), overlay(img, heat), pred, mask]
        for c, (ax, p) in enumerate(zip(axes[r], panels)):
            ax.imshow(p, cmap="gray" if c in (1, 3, 4) else None, vmin=0, vmax=1 if c in (1, 3, 4) else None)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(titles[c], fontsize=10.5, loc="center")
        m = meta[k]
        verdict = "tampered" if m["pred"] else "authentic"
        color = GOOD if (m["pred"] == (k != "authentic")) else BAD
        axes[r][0].set_ylabel(f"{k}\nscore {m['score']:.2f} → {verdict}", fontsize=9.5, color=color,
                              rotation=0, ha="right", va="center", labelpad=12)
    fig.suptitle("Test-split examples (synthetic benchmark; images held out from training)", x=0.01, ha="left",
                 fontsize=13, weight="bold", color=INK)
    return _save(fig, out, "examples.png")


def _preds(eval_dir: Path) -> list[dict]:
    with (eval_dir / "predictions.csv").open() as f:
        return list(csv.DictReader(f))


def roc(eval_dir: Path, out: Path) -> Path:
    preds = _preds(eval_dir)
    y = np.array([int(p["label"]) for p in preds])
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    for key, label, color in (("baseline", "ELA/noise block statistics (GBM)", ACCENT),
                              ("cnn_max", "patch CNN, max heatmap", PURPLE), ("fusion", "fusion", GOOD)):
        s = np.array([float(p[key]) for p in preds])
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, color=color, lw=2.2 if key == "fusion" else 1.6,
                label=f"{label}  AUC {roc_auc_score(y, s):.3f}")
    ax.plot([0, 1], [0, 1], color=GRID, lw=1, ls="--")
    ax.set_xlabel("false positive rate (authentic flagged)")
    ax.set_ylabel("true positive rate (tampered caught)")
    ax.grid(color=GRID)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax.set_title(f"Image-level ROC, test split (n={len(preds)})", fontsize=12)
    return _save(fig, out, "roc.png")


def breakdown(metrics: dict, out: Path) -> Path:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1.4, 1]})
    bm = metrics["by_manipulation"]
    kinds = [k for k in KINDS if k in bm]
    x = np.arange(len(kinds))
    w = 0.26
    series = (("auc", "image AUC (fusion)", GOOD), ("detected", "detected at threshold", ACCENT),
              ("pixel_f1", "pixel F1", PURPLE))
    for i, (key, label, color) in enumerate(series):
        vals = [bm[k]["auc"]["fusion"] if key == "auc" else bm[k][key] for k in kinds]
        a1.bar(x + (i - 1) * w, vals, w, color=color, label=label)
        for xi, v in zip(x + (i - 1) * w, vals):
            a1.text(xi, v + 0.015, f"{v:.2f}", ha="center", fontsize=8, color=INK)
    a1.set_xticks(x, [f"{k}\n(n={bm[k]['n']})" for k in kinds])
    a1.set_ylim(0, 1.12)
    a1.grid(axis="y", color=GRID)
    a1.legend(frameon=False, fontsize=8.5, ncol=3, loc="upper right")
    a1.set_title("By manipulation (test split)")
    bq = metrics["by_final_quality"]
    bins = [b for b in bq if bq[b]["n"]]
    xq = np.arange(len(bins))
    a2.bar(xq - 0.18, [bq[b]["auc_fusion"] for b in bins], 0.36, color=GOOD, label="image AUC")
    a2.bar(xq + 0.18, [bq[b]["pixel_f1"] for b in bins], 0.36, color=PURPLE, label="pixel F1")
    for xi, b in zip(xq, bins):
        a2.text(xi - 0.18, bq[b]["auc_fusion"] + 0.015, f"{bq[b]['auc_fusion']:.2f}", ha="center", fontsize=8)
        a2.text(xi + 0.18, bq[b]["pixel_f1"] + 0.015, f"{bq[b]['pixel_f1']:.2f}", ha="center", fontsize=8)
    a2.set_xticks(xq, [f"q {b}\n(n={bq[b]['n']})" for b in bins])
    a2.set_ylim(0, 1.12)
    a2.grid(axis="y", color=GRID)
    a2.legend(frameon=False, fontsize=8.5, loc="upper right")
    a2.set_title("By final JPEG quality")
    return _save(fig, out, "breakdown.png")


def training(models: Path, out: Path) -> Path | None:
    hist = json.loads((models / "training_history.json").read_text()) if (models / "training_history.json").exists() \
        else []
    if not hist:
        return None
    ep = [h["epoch"] for h in hist]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.4))
    a1.plot(ep, [h["train_loss"] for h in hist], color=ACCENT, marker="o", label="train (class-weighted)")
    a1.plot(ep, [h["val_loss"] for h in hist], color=SIGNAL, marker="o", label="val")
    a1.set_title("Patch CNN loss")
    a2.plot(ep, [h["train_acc"] for h in hist], color=ACCENT, marker="o", label="train")
    a2.plot(ep, [h["val_acc"] for h in hist], color=SIGNAL, marker="o", label="val")
    a2.set_title("Patch accuracy (threshold 0.5)")
    best = int(np.argmin([h["val_loss"] for h in hist]))
    for a in (a1, a2):
        a.axvline(ep[best], color=GRID, ls="--")
        a.set_xlabel("epoch")
        a.grid(color=GRID)
        a.legend(frameon=False, fontsize=8.5)
    a1.text(ep[best], a1.get_ylim()[1], " kept", va="top", fontsize=8, color=MUTED)
    return _save(fig, out, "training.png")


def render_all(eval_dir: Path, models: Path, out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((eval_dir / "metrics.json").read_text())
    paths = [architecture(out), examples(eval_dir, metrics["thresholds"], out), roc(eval_dir, out),
             breakdown(metrics, out), training(models, out)]
    return [p for p in paths if p]
