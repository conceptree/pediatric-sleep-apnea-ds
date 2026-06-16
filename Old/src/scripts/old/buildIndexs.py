#!/usr/bin/env python3
from pathlib import Path
import csv

SPLIT = Path("/Users/nunorodrigues/dev/tese/datasets/splits")

rows = []
for subset in ("train","val","test"):
    for label in ("positives","negatives"):
        folder = SPLIT/subset/label
        for edf in sorted(folder.glob("*.edf")):
            stem = edf.stem
            tsv = edf.with_suffix(".tsv")
            if not tsv.exists():
                continue
            rows.append({
                "subset": subset,
                "label": 1 if label=="positives" else 0,
                "stem": stem,
                "edf": str(edf.resolve()),
                "tsv": str(tsv.resolve()),
            })

out = SPLIT/"index.csv"
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["subset","label","stem","edf","tsv"])
    w.writeheader(); w.writerows(rows)
print(f"✅ index saved at: {out} ({len(rows)} rows)")
