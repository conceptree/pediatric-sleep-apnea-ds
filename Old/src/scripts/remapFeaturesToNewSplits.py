#!/usr/bin/env python3
# file: remapFeaturesToNewSplits.py
import argparse
import pandas as pd
from pathlib import Path

def main():
    ap = argparse.ArgumentParser(description="Atualiza 'subset' no features.parquet conforme CSVs de splits.")
    ap.add_argument("--features-in", required=True)
    ap.add_argument("--splits-dir", required=True, help="pasta com train.csv, val.csv, test.csv (colunas: stem,label)")
    ap.add_argument("--features-out", required=True)
    args = ap.parse_args()

    df = pd.read_parquet(args.features_in)
    m = {}
    for sub in ["train","val","test"]:
        p = Path(args.splits_dir) / f"{sub}.csv"
        part = pd.read_csv(p)
        for s in part["stem"].tolist():
            m[s] = sub

    before = df["subset"].value_counts(dropna=False).to_dict() if "subset" in df.columns else {}
    df["subset"] = df["stem"].map(m).fillna("drop")

    # filtra fora o que ficou "drop" (stems que não entraram em nenhum split novo)
    keep = df[df["subset"]!="drop"].copy()
    keep.to_parquet(args.features_out, index=False)

    after = keep["subset"].value_counts().to_dict()
    print("Antes (subset counts):", before)
    print("Depois (subset counts):", after)
    print(f"✅ features atualizado salvo em: {args.features_out}  | rows={len(keep)}")

if __name__ == "__main__":
    main()