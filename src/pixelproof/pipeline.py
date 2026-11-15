"""Train and evaluate on the synthetic benchmark.

* The patch CNN learns from **train** patches (early stopping on **val** patches).
* The handcrafted baseline classifier is fitted on **train** images.
* The fusion classifier and both thresholds are fitted on **val** images, so they never see heatmaps
  the CNN was trained on.
* Everything reported is measured on **test**, whose source photos the models never saw.
"""
from __future__ import annotations

import csv
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import benchmark, features
from .cnn import NumpyCNN, load_patch_model
from .config import Settings
from .detector import Detector, heat_stats, pixel_scores, training_patches

log = logging.getLogger("pixelproof")
QUALITY_BINS = [(75, 79), (80, 84), (85, 89), (90, 95)]


def patch_dataset(s: Settings, samples, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for smp in samples:
        img = smp.image()
        x, y = training_patches(features.tensor(img, s.ela_quality), smp.mask(), smp.label, s, rng)
        xs.append(x)
        ys.append(y)
    return np.concatenate(xs), np.concatenate(ys)


def _best_threshold(y: np.ndarray, score: np.ndarray) -> float:
    """Threshold that maximises balanced accuracy (Youden's J) on validation data."""
    fpr, tpr, thr = roc_curve(y, score)
    j = np.argmax(tpr - fpr)
    return float(min(max(thr[j], 1e-6), 1.0))


def train(s: Settings, data: Path, models: Path, patch_model: str | None = None,
          export_patches: Path | None = None) -> dict:
    tr, va = benchmark.load(data, "train"), benchmark.load(data, "val")
    history = []
    if patch_model:
        model = load_patch_model(patch_model)
        log.info("using pre-trained patch model %s", patch_model)
    else:
        x_tr, y_tr = patch_dataset(s, tr, s.cnn.seed)
        x_va, y_va = patch_dataset(s, va, s.cnn.seed + 1)
        log.info("patches: train %d (%d tampered) · val %d (%d tampered)", len(x_tr), int(y_tr.sum()), len(x_va),
                 int(y_va.sum()))
        if export_patches:
            export_patches.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(export_patches, x_train=x_tr, y_train=y_tr, x_val=x_va, y_val=y_va)
        model = NumpyCNN(seed=s.cnn.seed)
        history = model.fit(x_tr, y_tr, x_va.astype(np.float32), y_va, epochs=s.cnn.epochs,
                            batch_size=s.cnn.batch_size, lr=s.cnn.learning_rate, seed=s.cnn.seed, log=log.info)
    det = Detector(s, model)
    log.info("image-level classifiers: handcrafted baseline on train, fusion + thresholds on val")
    hand_tr = np.stack([features.handcrafted(smp.image(), s.block, s.ela_quality) for smp in tr])
    y_img_tr = np.array([smp.label for smp in tr])
    baseline = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.08, subsample=0.8,
                                          random_state=s.seed)
    baseline.fit(hand_tr, y_img_tr)

    rows, heats, masks, y_va_img = [], [], [], []
    for smp in va:
        img = smp.image()
        heat = det.heatmap(img)
        base = baseline.predict_proba(features.handcrafted(img, s.block, s.ela_quality)[None])[0, 1]
        rows.append(np.r_[base, heat_stats(heat)])
        heats.append(heat)
        masks.append(smp.mask())
        y_va_img.append(smp.label)
    xf, y_va_img = np.stack(rows), np.array(y_va_img)
    fusion = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
    fusion.fit(xf, y_va_img)
    # Thresholds from out-of-fold fusion scores, so validation fitting doesn't flatter them
    from sklearn.model_selection import cross_val_predict

    oof = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000)), xf, y_va_img,
                            cv=5, method="predict_proba")[:, 1]
    pix_candidates = np.round(np.arange(0.30, 0.96, 0.05), 2)
    pix_f1 = [np.mean([pixel_scores(h >= t, m)["f1"] for h, m, yy in zip(heats, masks, y_va_img) if yy])
              for t in pix_candidates]
    thresholds = {"image": _best_threshold(y_va_img, oof), "baseline": _best_threshold(y_va_img, xf[:, 0]),
                  "pixel": float(pix_candidates[int(np.argmax(pix_f1))])}
    det = Detector(s, model, baseline, fusion, thresholds)
    det.save(models)
    (models / "training_history.json").write_text(json.dumps(history, indent=1))
    log.info("thresholds %s", thresholds)
    return {"history": history, "thresholds": thresholds}


def _auc(y, score) -> float | None:
    return float(roc_auc_score(y, score)) if len(set(y)) == 2 else None


def evaluate(s: Settings, data: Path, models: Path, out: Path, patch_model: str | None = None) -> dict:
    det = Detector.load(s, models, patch_model)
    test = benchmark.load(data, "test")
    out.mkdir(parents=True, exist_ok=True)
    preds, examples = [], {}
    for smp in test:
        img = smp.image()
        sc = det.scores(img)
        m = smp.mask()
        row = {"id": smp.id, "label": smp.label, "manipulation": smp.manipulation, "source": smp.source,
               "first_quality": smp.first_quality, "final_quality": smp.final_quality,
               "baseline": sc["baseline"], "cnn_max": sc["cnn_max"], "fusion": sc["fusion"],
               "pred": int(sc["fusion"] >= det.thresholds["image"])}
        pm = sc["heat"] >= det.thresholds["pixel"]
        if smp.label:
            row.update(pixel_scores(pm, m))
        else:
            row["fp_area"] = float(pm.mean())
        preds.append(row)
        have = examples.get(smp.manipulation)
        if have is None or (have["procedural"] and smp.source != "procedural"):
            examples[smp.manipulation] = {"id": smp.id, "image": img, "mask": m, "heat": sc["heat"],
                                          "ela": features.ela(img, s.ela_quality), "score": sc["fusion"],
                                          "pred": row["pred"], "procedural": smp.source == "procedural"}
    with (out / "predictions.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in preds for k in r}))
        w.writeheader()
        w.writerows(preds)
    np.savez_compressed(out / "examples.npz", **{f"{k}__{f}": v[f] for k, v in examples.items()
                                                  for f in ("image", "mask", "heat", "ela")},
                        meta=json.dumps({k: {f: v[f] for f in ("id", "score", "pred")} for k, v in examples.items()}))
    metrics = summarise(preds, det.thresholds)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def summarise(preds: list[dict], thresholds: dict) -> dict:
    y = np.array([p["label"] for p in preds])
    auth = [p for p in preds if not p["label"]]
    pred = np.array([p["pred"] for p in preds])
    tampered = [p for p in preds if p["label"]]
    metrics = {
        "test_images": len(preds), "authentic": len(auth), "tampered": len(tampered),
        "thresholds": thresholds,
        "image_auc": {k: _auc(y, [p[k] for p in preds]) for k in ("baseline", "cnn_max", "fusion")},
        "image_accuracy": float((pred == y).mean()),
        "image_tpr": float(pred[y == 1].mean()), "image_fpr": float(pred[y == 0].mean()),
        "pixel_f1": float(np.mean([p["f1"] for p in tampered])),
        "pixel_iou": float(np.mean([p["iou"] for p in tampered])),
        "authentic_false_positive_area": float(np.mean([p["fp_area"] for p in auth])),
        "by_manipulation": {}, "by_final_quality": {},
    }
    groups = defaultdict(list)
    for p in tampered:
        groups[p["manipulation"]].append(p)
    for kind, ps in sorted(groups.items()):
        yy = [0] * len(auth) + [1] * len(ps)
        metrics["by_manipulation"][kind] = {
            "n": len(ps),
            "auc": {k: _auc(yy, [p[k] for p in auth] + [p[k] for p in ps]) for k in ("baseline", "cnn_max", "fusion")},
            "detected": float(np.mean([p["pred"] for p in ps])),
            "pixel_f1": float(np.mean([p["f1"] for p in ps])), "pixel_iou": float(np.mean([p["iou"] for p in ps]))}
    for lo, hi in QUALITY_BINS:
        ps = [p for p in preds if lo <= p["final_quality"] <= hi]
        yy = [p["label"] for p in ps]
        metrics["by_final_quality"][f"{lo}-{hi}"] = {
            "n": len(ps), "auc_fusion": _auc(yy, [p["fusion"] for p in ps]),
            "accuracy": float(np.mean([p["pred"] == p["label"] for p in ps])) if ps else None,
            "pixel_f1": float(np.mean([p["f1"] for p in ps if p["label"]])) if any(yy) else None}
    return metrics
