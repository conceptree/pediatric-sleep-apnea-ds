#!/usr/bin/env python3
from pathlib import Path
import sys

BASE = Path("/Users/nunorodrigues/dev/tese/datasets/nch-sleep")
if len(sys.argv) < 2:
    print("Uso: python3 find_stem.py <STEM> [<STEM2> ...]")
    sys.exit(1)

for stem in sys.argv[1:]:
    hits = list(BASE.rglob(f"{stem}.edf")) + list(BASE.rglob(f"{stem}.tsv"))
    if not hits:
        print(f"{stem}: não encontrado")
    else:
        print(f"{stem}:")
        for h in hits:
            print("  ", h)
