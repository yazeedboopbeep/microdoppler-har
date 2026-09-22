"""Participant-level train / validation / test split, fixed once and written to
data/split.json before any model is trained.

A recording of one person never appears in more than one set. This is the
split that matters for the use case (a radar that has never seen the person),
and it is also the harder one: random sample-level splits on this dataset
leak the same person's repetitions into the test set and inflate accuracy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SEED = 2026
FRACTIONS = (0.70, 0.15, 0.15)


def make_split(persons: np.ndarray, activity: np.ndarray, seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed)
    uniq = np.unique(persons)
    # keep the participants who performed falls spread across the three sets
    has_fall = np.array([np.any(activity[persons == p] == 6) for p in uniq])
    train, val, test = [], [], []
    for mask in (has_fall, ~has_fall):
        group = uniq[mask].copy()
        rng.shuffle(group)
        n = len(group)
        n_tr = int(round(FRACTIONS[0] * n))
        n_va = int(round(FRACTIONS[1] * n))
        train += list(group[:n_tr])
        val += list(group[n_tr : n_tr + n_va])
        test += list(group[n_tr + n_va :])
    return dict(seed=seed, fractions=FRACTIONS, train=sorted(train), val=sorted(val), test=sorted(test))


def assign(persons: np.ndarray, split: dict) -> np.ndarray:
    out = np.empty(len(persons), dtype=object)
    for name in ("train", "val", "test"):
        out[np.isin(persons, split[name])] = name
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/spectrograms.npz")
    ap.add_argument("--out", default="data/split.json")
    a = ap.parse_args()
    d = np.load(a.data, allow_pickle=True)
    split = make_split(d["person"], d["activity"])
    sets = assign(d["person"], split)
    split["n_samples"] = {k: int((sets == k).sum()) for k in ("train", "val", "test")}
    split["n_persons"] = {k: len(split[k]) for k in ("train", "val", "test")}
    Path(a.out).write_text(json.dumps(split, indent=2))
    print(json.dumps({k: split[k] for k in ("n_persons", "n_samples")}, indent=2))


if __name__ == "__main__":
    main()
