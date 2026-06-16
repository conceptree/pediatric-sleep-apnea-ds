#!/usr/bin/env python3
import os
import random
from pathlib import Path

BASE = Path("/Users/nunorodrigues/dev/tese/datasets/nch-sleep")
OUT  = Path("/Users/nunorodrigues/dev/tese/datasets/splits")

def split_files(label, seed=42):
    random.seed(seed)
    src = BASE / label
    edfs = sorted(src.glob("*.edf"))
    stems = [f.stem for f in edfs]

    random.shuffle(stems)

    n = len(stems)
    n_train = int(n * 0.8)
    n_val   = int(n * 0.1)

    train = stems[:n_train]
    val   = stems[n_train:n_train+n_val]
    test  = stems[n_train+n_val:]

    print(f"[{label}] total={n} → train={len(train)} val={len(val)} test={len(test)}")

    for subset, subset_list in (("train",train),("val",val),("test",test)):
        target = OUT / subset / label
        target.mkdir(parents=True, exist_ok=True)
        for stem in subset_list:
            for ext in (".edf",".tsv"):   # .atr opcional
                f = src / f"{stem}{ext}"
                if f.exists():
                    dest = target / f.name
                    if dest.exists():
                        dest.unlink()  # remove se já existir
                    os.symlink(f.resolve(), dest)

for lbl in ("positives","negatives"):
    split_files(lbl)
