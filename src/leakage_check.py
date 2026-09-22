"""How much does a naive random split (by sample) inflate the score compared
with the participant-level split? Same features, same SVM.

    python -m src.leakage_check
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .train_baseline import load


def svm():
    return make_pipeline(StandardScaler(), SVC(C=10, gamma="scale", class_weight="balanced"))


def main():
    X, y, sets = load("data/spectrograms.npz", "data/split.json")
    tr, va = sets == "train", sets == "val"
    m = svm().fit(X[tr], y[tr])
    print(f"participant split      macro-F1 {f1_score(y[va], m.predict(X[va]), average='macro'):.3f}")

    # random split over the same pool (train + val participants), same val size
    pool = tr | va
    Xp, yp = X[pool], y[pool]
    scores = []
    for i, (a, b) in enumerate(StratifiedShuffleSplit(n_splits=5, test_size=va.sum() / pool.sum(), random_state=0).split(Xp, yp)):
        scores.append(f1_score(yp[b], svm().fit(Xp[a], yp[a]).predict(Xp[b]), average="macro"))
    print(f"random sample split    macro-F1 {np.mean(scores):.3f} +- {np.std(scores):.3f} over 5 draws")


if __name__ == "__main__":
    main()
