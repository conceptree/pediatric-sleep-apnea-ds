#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Verificação de integridade do dataset NCH-Sleep
Modo rápido (--fast)   = apenas checa existência e tamanho mínimo
Modo completo (--full) = leitura de EDF/TSV para checagens extra
"""

import argparse
import pathlib
import pandas as pd
import mne

def check_fast(base_dir, min_edf=50_000_000, min_tsv=10_000):
    base = pathlib.Path(base_dir)
    edfs = list(base.rglob("*.edf"))
    tsps = list(base.rglob("*.tsv"))

    stems_edf = {f.stem for f in edfs if f.stat().st_size >= min_edf}
    stems_tsv = {f.stem for f in tsps if f.stat().st_size >= min_tsv}

    ok = stems_edf & stems_tsv
    missing_edf = stems_tsv - stems_edf
    missing_tsv = stems_edf - stems_tsv

    print(f"📊 Fast check: {len(ok)} pares válidos, {len(missing_edf)} sem EDF, {len(missing_tsv)} sem TSV")
    return ok, missing_edf, missing_tsv


def check_full(base_dir):
    base = pathlib.Path(base_dir)
    edfs = list(base.rglob("*.edf"))
    results = []

    for edf in edfs:
        tsv = edf.with_suffix(".tsv")
        stem = edf.stem
        has_tsv = tsv.exists()

        # tenta abrir EDF
        try:
            raw = mne.io.read_raw_edf(str(edf), preload=False, verbose=False)
            n_channels = len(raw.ch_names)
            dur = raw.n_times / raw.info["sfreq"]
        except Exception as e:
            n_channels, dur = None, None
            print(f"❌ EDF corrompido: {edf} ({e})")

        # tenta abrir TSV
        if has_tsv:
            try:
                df = pd.read_csv(tsv, sep="\t")
                n_rows = len(df)
            except Exception as e:
                n_rows = None
                print(f"❌ TSV corrompido: {tsv} ({e})")
        else:
            n_rows = None

        results.append(dict(stem=stem, has_tsv=has_tsv, n_channels=n_channels, duration_s=dur, n_tsv_rows=n_rows))

    df = pd.DataFrame(results)
    print(f"📊 Full check: {len(df)} EDFs verificados")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", required=True, help="Diretório base com os .edf/.tsv")
    ap.add_argument("--fast", action="store_true", help="Rodar verificação rápida (quickVerify)")
    ap.add_argument("--full", action="store_true", help="Rodar verificação completa (verify)")
    ap.add_argument("--min-edf", type=int, default=50_000_000, help="Tamanho mínimo do EDF (default=50MB)")
    ap.add_argument("--min-tsv", type=int, default=10_000, help="Tamanho mínimo do TSV (default=10KB)")
    ap.add_argument("--out-csv", help="Salvar resultados em CSV (apenas --full)")
    args = ap.parse_args()

    if args.fast:
        check_fast(args.base_dir, args.min_edf, args.min_tsv)

    if args.full:
        df = check_full(args.base_dir)
        if args.out_csv:
            df.to_csv(args.out_csv, index=False)
            print(f"💾 CSV salvo em {args.out_csv}")


if __name__ == "__main__":
    main()
