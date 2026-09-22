"""Small CNN on 128x128 micro-Doppler spectrograms (CPU-friendly).

    python -m src.train_cnn --epochs 30

Model selection uses validation macro-F1; the best epoch's weights and
confusion matrix are saved under results/.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score

from .glasgow_io import ACTIVITY_NAMES
from .split import assign


class SmallCNN(nn.Module):
    def __init__(self, n_classes: int = 6):
        super().__init__()
        ch = (1, 16, 32, 64, 64)
        self.blocks = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(ch[i], ch[i + 1], 3, padding=1), nn.BatchNorm2d(ch[i + 1]), nn.ReLU(), nn.MaxPool2d(2))
             for i in range(4)]
        )
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(64, n_classes))

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.head(x)


def augment(x: torch.Tensor) -> torch.Tensor:
    # random circular time shift and small Doppler shift; both are physically
    # plausible (activity can start anywhere in the 5 s window; radial angle changes)
    b = x.shape[0]
    shifts_t = torch.randint(-16, 17, (b,))
    shifts_f = torch.randint(-4, 5, (b,))
    out = torch.empty_like(x)
    for i in range(b):
        out[i] = torch.roll(x[i], shifts=(int(shifts_f[i]), int(shifts_t[i])), dims=(1, 2))
    # random gain
    out = out * (0.9 + 0.2 * torch.rand(b, 1, 1, 1))
    return out


def evaluate(model, X, y, labels):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), 128):
            preds.append(model(X[i : i + 128]).argmax(1))
    pred = torch.cat(preds).numpy()
    f1 = f1_score(y, pred, average="macro")
    cm = confusion_matrix(y, pred, labels=list(range(len(labels))))
    return f1, (pred == y).mean(), cm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/spectrograms.npz")
    ap.add_argument("--split", default="data/split.json")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--out", default="results/cnn.json")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    d = np.load(a.data, allow_pickle=True)
    split = json.loads(Path(a.split).read_text())
    sets = assign(d["person"], split)
    labels = sorted(ACTIVITY_NAMES)
    names = [ACTIVITY_NAMES[k] for k in labels]
    y_all = np.array([labels.index(v) for v in d["activity"]])
    X_all = torch.from_numpy(d["spec"]).float().unsqueeze(1)

    tr, va = sets == "train", sets == "val"
    Xtr, ytr = X_all[tr], torch.from_numpy(y_all[tr])
    Xva, yva = X_all[va], y_all[va]

    counts = np.bincount(ytr.numpy(), minlength=len(labels))
    weights = torch.tensor(counts.sum() / (len(labels) * np.maximum(counts, 1)), dtype=torch.float32)

    model = SmallCNN(len(labels))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * ((len(Xtr) + 31) // 32))

    best = dict(val_macro_f1=-1)
    history = []
    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        t0, tot = time.time(), 0.0
        for i in range(0, len(Xtr), 32):
            idx = perm[i : i + 32]
            xb, yb = augment(Xtr[idx]), ytr[idx]
            loss = F.cross_entropy(model(xb), yb, weight=weights, label_smoothing=0.05)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(idx)
        f1, acc, cm = evaluate(model, Xva, yva, names)
        history.append(dict(epoch=ep + 1, train_loss=tot / len(Xtr), val_macro_f1=float(f1), val_accuracy=float(acc)))
        print(f"epoch {ep + 1:02d} loss {tot / len(Xtr):.3f} val F1 {f1:.3f} acc {acc:.3f} ({time.time() - t0:.0f}s)", flush=True)
        if f1 > best["val_macro_f1"]:
            best = dict(epoch=ep + 1, val_macro_f1=float(f1), val_accuracy=float(acc), confusion=cm.tolist(), labels=names)
            Path("results").mkdir(exist_ok=True)
            torch.save(model.state_dict(), "results/cnn_best.pt")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(dict(best=best, history=history, n_params=sum(p.numel() for p in model.parameters())), indent=2))
    print("best", best["epoch"], best["val_macro_f1"])


if __name__ == "__main__":
    main()
