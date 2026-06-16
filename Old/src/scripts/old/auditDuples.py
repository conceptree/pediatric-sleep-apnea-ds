#!/usr/bin/env python3
from pathlib import Path

BASE = Path("/Users/nunorodrigues/dev/tese/datasets/nch-sleep")
pos = {p.stem for p in (BASE/"positives").glob("*.edf")}
neg = {p.stem for p in (BASE/"negatives").glob("*.edf")}

dupes = sorted(pos & neg)
print(f"positives={len(pos)} negatives={len(neg)} dupes={len(dupes)}")
for s in dupes[:50]:
    print("  DUP:", s)
if len(dupes) > 50:
    print("  …", len(dupes)-50, "mais")
