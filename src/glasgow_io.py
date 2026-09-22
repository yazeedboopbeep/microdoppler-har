"""Readers for the University of Glasgow 'Radar signatures of human activities'
dataset (doi:10.5525/gla.researchdata.848).

Each recording is a text .dat file. The first four lines are the radar
parameters (carrier frequency in Hz, sweep time in ms, samples per sweep,
bandwidth in Hz). Every following line is one complex baseband sample written
as MATLAB-style text, e.g. "-0.0123+0.0456i".
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ACTIVITY_NAMES = {
    1: "walking",
    2: "sitting down",
    3: "standing up",
    4: "picking up object",
    5: "drinking water",
    6: "fall",
}

# e.g. 1P01A01R1.dat -> activity 1, subject 01, A01 (activity code again), repetition 1
_NAME_RE = re.compile(r"^(?P<act>\d)P(?P<person>\d+)A(?P<a>\d+)R(?P<rep>\d+)", re.IGNORECASE)
# insert a space between the real part and the sign of the imaginary part
_SPLIT_RE = re.compile(r"(?<=[0-9.])([+-])")


@dataclass
class Recording:
    path: Path
    activity: int
    person: str
    session: str
    repetition: int
    collection: str
    fc: float
    sweep_time: float  # seconds
    n_samples: int  # samples per sweep
    bandwidth: float
    data: np.ndarray  # shape (n_samples, n_sweeps), complex64


def collection_id(path: Path) -> str:
    """Folder names start with the collection number, e.g. '5 February 2019 UoG Dataset'."""
    for part in Path(path).parts[::-1]:
        m = re.match(r"^(\d)\s", part)
        if m:
            return m.group(1)
    return "0"


def parse_name(path: Path) -> dict:
    m = _NAME_RE.match(Path(path).stem)
    if not m:
        raise ValueError(f"unexpected file name: {Path(path).name}")
    coll = collection_id(path)
    # subject IDs restart in every collection (P03 exists in March 2017 and in
    # Feb 2019), so the participant key has to carry the collection number
    return {
        "activity": int(m["act"]),
        "person": f"C{coll}_P{m['person']}",
        "session": m["a"],
        "repetition": int(m["rep"]),
        "collection": coll,
    }


def _to_complex(txt: str) -> np.ndarray:
    # "a+bi" per line -> "a +b" -> float pairs. np.fromstring with sep is C-fast,
    # which matters: a 10 s recording is 1.28 M complex values of text.
    txt = txt.replace("i", "").replace("j", "")
    txt = _SPLIT_RE.sub(r" \1", txt)
    flat = np.fromstring(txt, dtype=np.float32, sep=" ")
    if flat.size % 2:
        flat = flat[:-1]
    pairs = flat.reshape(-1, 2)
    return (pairs[:, 0] + 1j * pairs[:, 1]).astype(np.complex64)


def read_dat(path: Path) -> Recording:
    path = Path(path)
    with open(path, "r") as f:
        head = [f.readline() for _ in range(4)]
        body = f.read()
    fc = float(head[0])
    sweep_ms = float(head[1])
    nts = int(float(head[2]))
    bw = float(head[3])
    raw = _to_complex(body)
    n_sweeps = raw.size // nts
    data = raw[: n_sweeps * nts].reshape(n_sweeps, nts).T  # (fast-time, slow-time)
    meta = parse_name(path)
    return Recording(
        path=path,
        fc=fc,
        sweep_time=sweep_ms * 1e-3,
        n_samples=nts,
        bandwidth=bw,
        data=data,
        **meta,
    )


def list_recordings(root: Path) -> list[Path]:
    root = Path(root)
    return sorted(p for p in root.rglob("*.dat") if _NAME_RE.match(p.stem))
