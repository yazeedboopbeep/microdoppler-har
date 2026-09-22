"""Walk the raw .dat files, compute one spectrogram per 5-second segment and
cache everything as a .npz so the training scripts never touch the raw text
files again (16 GB of text for a 2 GB archive).

    python -m src.build_dataset --raw "data/raw/1 December 2017 Dataset" --out data/parts/c1.npz
    python -m src.build_dataset --merge data/parts/*.npz --out data/spectrograms.npz

Each 10 s walking recording becomes two 5 s samples; everything else is one
sample. Spectrograms are resized to 128 Doppler bins (-250..250 Hz) x 128 time
frames and stored as uint8 to keep the cache small.
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.ndimage import zoom

from .glasgow_io import list_recordings, read_dat
from .spectrogram import recording_to_spectrogram

SEG_S = 5.0
OUT_SHAPE = (128, 128)
F_KEEP_HZ = 250.0
KEYS = ("activity", "person", "session", "repetition", "collection", "segment", "duration_s", "n_sweeps", "bandwidth", "filename")


def _resize(s: np.ndarray, shape=OUT_SHAPE) -> np.ndarray:
    fy = shape[0] / s.shape[0]
    fx = shape[1] / s.shape[1]
    return zoom(s, (fy, fx), order=1)


def process_one(path: str) -> dict:
    rec = read_dat(Path(path))
    prf = 1.0 / rec.sweep_time
    n_sweeps = rec.data.shape[1]
    duration = n_sweeps / prf
    s, f, t = recording_to_spectrogram(rec)
    keep = np.abs(f) <= F_KEEP_HZ
    s = s[keep, :]
    n_seg = max(1, int(round(duration / SEG_S)))
    frames_per_seg = s.shape[1] // n_seg
    items = []
    for k in range(n_seg):
        seg = s[:, k * frames_per_seg : (k + 1) * frames_per_seg]
        items.append(
            dict(
                spec=np.clip(_resize(seg) * 255, 0, 255).astype(np.uint8),
                activity=rec.activity, person=rec.person, session=rec.session,
                repetition=rec.repetition, collection=rec.collection, segment=k,
                duration_s=duration, n_sweeps=n_sweeps, bandwidth=rec.bandwidth, filename=Path(path).name,
            )
        )
    return dict(items=items, fc=rec.fc, bw=rec.bandwidth, nts=rec.n_samples, prf=prf)


def build(raw: Path, out: Path, workers: int, limit: int = 0):
    files = [str(p) for p in list_recordings(raw)]
    if limit:
        files = files[:limit]
    print(f"{len(files)} recordings under {raw}")
    cols = {k: [] for k in KEYS}
    specs, radar = [], {}
    with Pool(workers) as pool:
        for i, res in enumerate(pool.imap_unordered(process_one, files, chunksize=2)):
            radar = dict(fc=res["fc"], bw=res["bw"], nts=res["nts"], prf=res["prf"])
            for it in res["items"]:
                specs.append(it["spec"])
                for k in KEYS:
                    cols[k].append(it[k])
            if i % 50 == 0:
                print(f"  {i}/{len(files)}", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, spec=np.stack(specs), **{k: np.array(v) for k, v in cols.items()})
    out.with_suffix(".json").write_text(json.dumps(dict(radar=radar, n_recordings=len(files), n_samples=len(specs)), indent=2))
    print("saved", out, len(specs), "samples")


def merge(parts: list[Path], out: Path):
    arrs = [np.load(p, allow_pickle=True) for p in parts]
    merged = {k: np.concatenate([a[k] for a in arrs]) for k in ("spec",) + KEYS}
    np.savez_compressed(out, **merged)
    print("merged", len(parts), "parts ->", out, merged["spec"].shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/spectrograms.npz")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--merge", nargs="*", help="npz parts to merge instead of building")
    a = ap.parse_args()
    if a.merge:
        merge([Path(p) for p in a.merge], Path(a.out))
    else:
        build(Path(a.raw), Path(a.out), a.workers, a.limit)


if __name__ == "__main__":
    main()
