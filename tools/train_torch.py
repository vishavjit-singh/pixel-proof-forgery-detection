"""Optional: train the same patch CNN in PyTorch and export it to ONNX.

Run by .github/workflows/torch.yml (PyTorch is not needed anywhere else). The exported model plugs into
the detector through ``--patch-model onnx:models/patch_cnn.onnx`` and is executed with ONNX Runtime.

    python -m pixelproof train                         # writes data/patches.npz
    python tools/train_torch.py --patches data/patches.npz --out models/patch_cnn.onnx
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from torch import nn


class PatchCNN(nn.Module):
    """Mirror of pixelproof.cnn.NumpyCNN (input N×7×32×32 → one logit per patch)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(7, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patches", default="data/patches.npz")
    ap.add_argument("--out", default="models/patch_cnn.onnx")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--seed", type=int, default=10)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    d = np.load(args.patches)
    xtr, ytr = torch.tensor(d["x_train"], dtype=torch.float32), torch.tensor(d["y_train"], dtype=torch.float32)
    xva, yva = torch.tensor(d["x_val"], dtype=torch.float32), torch.tensor(d["y_val"], dtype=torch.float32)
    model = PatchCNN()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((ytr == 0).sum() / max((ytr == 1).sum(), 1)))
    best, best_state = float("inf"), None
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(xtr))
        for i in range(0, len(xtr), 64):
            idx = perm[i:i + 64]
            xb = xtr[idx]
            if torch.rand(1) < 0.5:
                xb = torch.flip(xb, dims=[3])
            if torch.rand(1) < 0.5:
                xb = torch.flip(xb, dims=[2])
            opt.zero_grad()
            loss = loss_fn(model(xb).squeeze(1), ytr[idx])
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            logits = model(xva).squeeze(1)
            val_loss = nn.functional.binary_cross_entropy_with_logits(logits, yva).item()
            acc = ((logits > 0).float() == yva).float().mean().item()
        print(f"epoch {epoch}: val loss {val_loss:.4f} acc {acc:.3f}", flush=True)
        if val_loss < best:
            best, best_state = val_loss, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    torch.onnx.export(model, torch.zeros(1, 7, 32, 32), args.out, input_names=["patches"], output_names=["logit"],
                      dynamic_axes={"patches": {0: "n"}, "logit": {0: "n"}}, opset_version=17)
    print(f"exported {args.out}")


if __name__ == "__main__":
    main()
