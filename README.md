# microdoppler-har

Code for my semester project proposal: classifying six indoor activities
(walking, sitting down, standing up, picking up an object, drinking, falling)
from 5 s micro-Doppler spectrograms of a 5.8 GHz FMCW radar.

Dataset: *Radar signatures of human activities*, University of Glasgow, 2019,
CC BY 4.0, https://doi.org/10.5525/gla.researchdata.848 (2 GB archive,
1754 recordings, 7 collection campaigns, 106 participant entries).

The proposal PDF is in `proposal/`.

## Layout

```
src/glasgow_io.py      read the .dat text files, parse file names
src/spectrogram.py     range FFT -> MTI high-pass -> STFT micro-Doppler
src/build_dataset.py   raw files -> data/spectrograms.npz (uint8, 128x128 per 5 s)
src/split.py           participant-level 70/15/15 split -> data/split.json
src/features.py        86 hand-crafted features per spectrogram
src/train_baseline.py  RBF-SVM and random forest on the features
src/train_cnn.py       small CNN on the spectrograms (CPU is enough)
src/leakage_check.py   participant split vs. random split, same SVM
src/plots.py           figures used in the proposal
data/split.json        frozen split (participant ids per set)
results/               metrics and confusion matrices from the runs in the proposal
figures/               rendered figures
```

`data/spectrograms.npz` (3.8 MB) is the preprocessed cache the training
scripts use. It is committed so the results can be reproduced without
downloading the 2 GB archive. The raw `.dat` files are not in the repo.

## Reproduce

```
pip install -r requirements.txt

# optional: rebuild the cache from the raw archive (16 GB extracted)
# 7z x Dataset_848.7z -odata/raw
# for d in data/raw/*/; do python -m src.build_dataset --raw "$d" --out "data/parts/$(basename "$d" | cut -c1).npz" --workers 4; done
# python -m src.build_dataset --merge data/parts/*.npz --out data/spectrograms.npz

python -m src.split
python -m src.train_baseline
python -m src.train_cnn --epochs 30
python -m src.leakage_check
python -m src.plots
```

The CNN takes about 6 minutes for 30 epochs on two CPU cores.

## Notes on the data

* Subject IDs restart in every collection campaign (P03 exists in March 2017
  and in February 2019), so participants are keyed as `C<campaign>_P<id>`.
* 60 recordings in the West Cumbria campaign (subjects P51 to P55) have a
  100 MHz bandwidth in the header instead of 400 MHz, and the target really
  is in range bins 1 to 3 in those files. `spectrogram.py` picks range bins by
  distance (1.5 to 10 m) rather than by index for that reason.
* Walking recordings are 10 s (eight are 20 s) and are cut into 5 s segments;
  every other activity is a single 5 s recording.
* Falls were only recorded in the five university campaigns (plus one subject
  at NG Homes), so the fall class has 198 samples from 67 participants, all
  aged 44 or under.

## Proposal build

```
cd proposal && latexmk -pdf proposal.tex
```
