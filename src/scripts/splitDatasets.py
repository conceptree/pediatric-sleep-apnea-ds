# split_dataset.py
import random
import shutil
from pathlib import Path

BASE = Path("/Users/nunorodrigues/dev/tese/datasets/nch-sleep")
OUT  = Path("/Users/nunorodrigues/dev/tese/datasets/splits")
OUT.mkdir(exist_ok=True)

def collect(label, n=None):
    """Collects stems (without extension) based on existing .edf/.tsv files"""
    stems = [f.stem for f in (BASE/label).glob("*.edf")]
    stems = [s for s in stems if (BASE/label/f"{s}.tsv").exists()]
    stems.sort()
    if n: stems = stems[:n]
    return stems

def split(stems, train=0.7, val=0.15, seed=42):
    random.Random(seed).shuffle(stems)
    n = len(stems)
    n_train = int(n*train)
    n_val   = int(n*val)
    return stems[:n_train], stems[n_train:n_train+n_val], stems[n_train+n_val:]

def copy_files(stems, label, subset):
    dest = OUT/subset/label
    dest.mkdir(parents=True, exist_ok=True)
    for s in stems:
        for ext in (".edf", ".tsv"):
            src = BASE/label/f"{s}{ext}"
            if src.exists():
                shutil.copy2(src, dest/src.name)

# --- MAIN ---
pos = collect("positives", n=56)
neg = collect("negatives", n=56)

print(f"Positives: {len(pos)} | Negatives: {len(neg)}")

for label, stems in [("positives", pos), ("negatives", neg)]:
    tr, va, te = split(stems)
    copy_files(tr, label, "train")
    copy_files(va, label, "val")
    copy_files(te, label, "test")
    print(f"[{label}] → train={len(tr)}, val={len(va)}, test={len(te)}")

print("✅ Split completed in:", OUT)
