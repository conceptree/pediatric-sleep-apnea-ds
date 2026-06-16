#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import mne
import numpy as np
from pathlib import Path
import argparse


def inspect_tsv(tsv_path: Path):
    print(f"\n=== TSV: {tsv_path.name} ===")
    try:
        df = pd.read_csv(tsv_path, sep="\t")
    except Exception as e:
        print("Erro a ler TSV:", e)
        return
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("\nPrimeiras linhas:")
    print(df.head(10))
    if "event" in df.columns:
        print("\nEventos mais frequentes:")
        print(df["event"].value_counts().head(10))
    if "duration" in df.columns:
        print("\nDuração média (s):", df["duration"].mean())

def inspect_edf(edf_path: Path):
    print(f"\n=== EDF: {edf_path.name} ===")
    try:
        raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose=False)
    except Exception as e:
        print("Erro a ler EDF:", e)
        return
    print("N canais:", len(raw.ch_names))
    print("Primeiros canais:", raw.ch_names[:20])

    # procurar spo2 / osat
    spo2_chs = [ch for ch in raw.ch_names if "spo2" in ch.lower() or "osat" in ch.lower()]
    if spo2_chs:
        ch = spo2_chs[0]
        x, _ = raw[ch, :]
        arr = np.asarray(x).ravel()
        print(f"\nCanal SpO₂ detectado: {ch}")
        print(" Stats -> min:", np.nanmin(arr), "max:", np.nanmax(arr), "mean:", np.nanmean(arr))
        print(" NaN %:", np.mean(np.isnan(arr))*100)
    else:
        print("⚠️ Nenhum canal SpO₂ encontrado.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", nargs="*", help="Um ou mais ficheiros .tsv")
    ap.add_argument("--edf", nargs="*", help="Um ou mais ficheiros .edf")
    args = ap.parse_args()

    if args.tsv:
        for f in args.tsv:
            inspect_tsv(Path(f))
    if args.edf:
        for f in args.edf:
            inspect_edf(Path(f))
