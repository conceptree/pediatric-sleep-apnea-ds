#!/usr/bin/env python3
from pathlib import Path
import random, json
import mne, re

SPLIT = Path("/Users/nunorodrigues/dev/tese/datasets/splits")

# wide regex for oximeter
SPO2_PATTERNS = [
    r"\bspo?2\b", r"\bsao?2\b", r"oxim", r"oxi", r"sat", r"oxygen", r"o2",
    r"spo_?2", r"sao_?2"
]
rx_spo2 = re.compile("|".join(SPO2_PATTERNS), re.I)

def sample_files(folder, k=10):
    edfs = sorted(Path(folder).glob("*.edf"))
    random.Random(42).shuffle(edfs)
    return edfs[:k]

report = {}
for subset in ("train","val","test"):
    for label in ("positives","negatives"):
        folder = SPLIT/subset/label
        files = sample_files(folder, k=10)
        seen = []
        matched = []
        for f in files:
            raw = mne.io.read_raw_edf(f, preload=False, verbose=False)
            names = raw.ch_names
            seen += names
            hits = [n for n in names if rx_spo2.search(n)]
            matched += hits or []
        key = f"{subset}/{label}"
        report[key] = {
            "n_files": len(files),
            "unique_channels": sorted(set([n.lower() for n in seen])),
            "matched_spo2": sorted(set([n.lower() for n in matched])),
        }

print(json.dumps(report, indent=2)[:10000])
