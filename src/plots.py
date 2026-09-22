"""Figures for the proposal: sample spectrograms per class, class frequency,
participants per class, recording duration histogram, confusion matrix.

    python -m src.plots
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .glasgow_io import ACTIVITY_NAMES
from .split import assign

SHORT = {1: "walk", 2: "sit", 3: "stand", 4: "pickup", 5: "drink", 6: "fall"}
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
GREY = "#52514e"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "figure.dpi": 150,
    "pdf.fonttype": 42,
})


def fig_samples(d, out: Path, seed=3):
    rng = np.random.default_rng(seed)
    labels = sorted(ACTIVITY_NAMES)
    fig, axes = plt.subplots(2, 3, figsize=(6.5, 3.6), constrained_layout=True)
    for ax, k in zip(axes.ravel(), labels):
        idx = rng.choice(np.where(d["activity"] == k)[0])
        s = d["spec"][idx].astype(float) / 255.0
        ax.imshow(s, aspect="auto", origin="lower", cmap="magma", extent=[0, 5, -250, 250], vmin=0, vmax=1)
        ax.set_title(f"{k}: {ACTIVITY_NAMES[k]}", fontsize=9)
        ax.set_xticks([0, 2.5, 5])
        ax.set_yticks([-200, 0, 200])
    for ax in axes[1]:
        ax.set_xlabel("time (s)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Doppler (Hz)")
    fig.savefig(out / "samples.pdf")
    fig.savefig(out / "samples.png")
    plt.close(fig)


def fig_distribution(d, split, out: Path):
    labels = sorted(ACTIVITY_NAMES)
    names = [ACTIVITY_NAMES[k] for k in labels]
    sets = assign(d["person"], split)
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.3), constrained_layout=True)

    # (a) samples per class, stacked by split
    ax = axes[0]
    bottom = np.zeros(len(labels))
    for name, col in (("train", BLUE), ("val", ORANGE), ("test", AQUA)):
        cnt = np.array([np.sum((d["activity"] == k) & (sets == name)) for k in labels])
        ax.bar(range(len(labels)), cnt, bottom=bottom, color=col, width=0.7, label=name, edgecolor="white", linewidth=0.8)
        bottom += cnt
    for i, tot in enumerate(bottom):
        ax.text(i, tot + 5, str(int(tot)), ha="center", va="bottom", fontsize=7, color=GREY)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([SHORT[k] for k in labels], fontsize=7, rotation=30, ha="right", rotation_mode="anchor")
    ax.set_ylabel("5 s samples")
    ax.set_title("(a) samples per class", fontsize=9)
    ax.legend(frameon=False, fontsize=7)

    # (b) participants per class
    ax = axes[1]
    per = [len(np.unique(d["person"][d["activity"] == k])) for k in labels]
    ax.bar(range(len(labels)), per, color=BLUE, width=0.7)
    for i, v in enumerate(per):
        ax.text(i, v + 0.5, str(v), ha="center", va="bottom", fontsize=7, color=GREY)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([SHORT[k] for k in labels], fontsize=7, rotation=30, ha="right", rotation_mode="anchor")
    ax.set_ylabel("participants")
    ax.set_title("(b) participants per class", fontsize=9)

    # (c) recording durations
    ax = axes[2]
    dur = d["duration_s"]
    keys = np.array([f"{c}/{f}" for c, f in zip(d["collection"], d["filename"])])
    uniq_files, first = np.unique(keys, return_index=True)
    dur = dur[first]
    vals, cnt = np.unique(dur, return_counts=True)
    ax.bar([f"{v:g} s" for v in vals], cnt, color=BLUE, width=0.6)
    for i, v in enumerate(cnt):
        ax.text(i, v + 10, str(v), ha="center", va="bottom", fontsize=7, color=GREY)
    ax.set_xlabel("recording length")
    ax.set_ylabel("recordings")
    ax.set_title("(c) recording duration", fontsize=9)

    fig.savefig(out / "distribution.pdf")
    fig.savefig(out / "distribution.png")
    plt.close(fig)


def fig_confusion(res_path: Path, key: str, out: Path, title: str):
    r = json.loads(res_path.read_text())
    r = r[key] if key in r else r["best"]
    cm = np.array(r["confusion"])
    names = r["labels"]
    cmn = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(3.4, 3.0), constrained_layout=True)
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7, color="white" if cmn[i, j] > 0.5 else "black")
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels([SHORT[i + 1] for i in range(len(names))], rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels([SHORT[i + 1] for i in range(len(names))], fontsize=7)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, fontsize=9)
    ax.spines[:].set_visible(False)
    fig.savefig(out / f"confusion_{key}.pdf")
    fig.savefig(out / f"confusion_{key}.png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/spectrograms.npz")
    ap.add_argument("--split", default="data/split.json")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    d = np.load(a.data, allow_pickle=True)
    split = json.loads(Path(a.split).read_text())
    fig_samples(d, out)
    fig_distribution(d, split, out)
    if Path("results/cnn.json").exists():
        fig_confusion(Path("results/cnn.json"), "best", out, "CNN, validation participants")
    if Path("results/baseline.json").exists():
        fig_confusion(Path("results/baseline.json"), "svm_rbf", out, "SVM, validation participants")
    print("figures written to", out)


if __name__ == "__main__":
    main()
