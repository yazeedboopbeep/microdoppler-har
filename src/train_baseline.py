"""Classical baselines on hand-crafted features: RBF-SVM and random forest.
Trained on the train participants, model selection on the validation
participants. The test participants are not touched here.

    python -m src.train_baseline
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .features import spectrogram_features
from .glasgow_io import ACTIVITY_NAMES
from .split import assign


def load(data_path: str, split_path: str):
    d = np.load(data_path, allow_pickle=True)
    split = json.loads(Path(split_path).read_text())
    sets = assign(d["person"], split)
    X = np.stack([spectrogram_features(s) for s in d["spec"]])
    y = d["activity"].astype(int)
    return X, y, sets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/spectrograms.npz")
    ap.add_argument("--split", default="data/split.json")
    ap.add_argument("--out", default="results/baseline.json")
    a = ap.parse_args()

    X, y, sets = load(a.data, a.split)
    tr, va = sets == "train", sets == "val"
    labels = sorted(ACTIVITY_NAMES)
    names = [ACTIVITY_NAMES[k] for k in labels]
    results = {}

    models = {
        "svm_rbf": make_pipeline(StandardScaler(), SVC(C=10, gamma="scale", class_weight="balanced")),
        "random_forest": RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=0, n_jobs=-1),
    }
    for name, model in models.items():
        model.fit(X[tr], y[tr])
        pred = model.predict(X[va])
        f1 = f1_score(y[va], pred, average="macro")
        acc = (pred == y[va]).mean()
        cm = confusion_matrix(y[va], pred, labels=labels)
        print(f"\n== {name}: val macro-F1 {f1:.3f}, accuracy {acc:.3f}")
        print(classification_report(y[va], pred, labels=labels, target_names=names, digits=3))
        results[name] = dict(val_macro_f1=float(f1), val_accuracy=float(acc), confusion=cm.tolist(), labels=names)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
