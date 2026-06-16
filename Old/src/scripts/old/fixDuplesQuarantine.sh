#!/usr/bin/env python3
from pathlib import Path
from collections import defaultdict

BASE = Path("/Users/nunorodrigues/dev/tese/datasets/splits")
stems_by_split = defaultdict(set)

for subset in ("train","val","test"):
  for label in ("positives","negatives"):
    for edf in (BASE/subset/label).glob("*.edf"):
      stems_by_split[subset].add(edf.stem)

# overlaps entre splits
splits = list(stems_by_split.keys())
for i in range(len(splits)):
  for j in range(i+1, len(splits)):
    a, b = splits[i], splits[j]
    inter = stems_by_split[a] & stems_by_split[b]
    if inter:
      print(f"⚠️  Leakage {a}∩{b} = {len(inter)}")
      for s in sorted(list(inter))[:30]:
        print("   ", s)
