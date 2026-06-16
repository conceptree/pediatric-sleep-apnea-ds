#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_clean_manifest.py
Filtra estudos "válidos" a partir do verify/report.csv e resolve caminhos EDF/TSV.

Regras por omissão (ajustáveis por flags):
- tem TSV (has_tsv==True)
- duração >= 6h
- n_canais >= 6
- n_linhas TSV >= 200

Saída: manifest CSV com colunas:
stem, duration_h, n_channels, n_tsv_rows, edf_path, tsv_path
"""

import argparse
import pathlib
import pandas as pd

def find_first(base: pathlib.Path, pattern: str):
    matches = list(base.rglob(pattern))
    return str(matches[0]) if matches else ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", required=True, help="Pasta base onde estão os .edf/.tsv (ex.: /Volumes/CORSAIR/tese/datasets/nch-sleep/all_raw)")
    ap.add_argument("--report-csv", required=True, help="CSV gerado por verify.py (--out-csv .../verify/report.csv)")
    ap.add_argument("--out-manifest", required=True, help="CSV de saída com estudos válidos")
    ap.add_argument("--min-hours", type=float, default=6.0, help="Duração mínima em horas (default=6)")
    ap.add_argument("--min-channels", type=int, default=6, help="Mínimo de canais (default=6)")
    ap.add_argument("--min-tsv-rows", type=int, default=200, help="Mínimo de linhas no TSV (default=200)")
    args = ap.parse_args()

    base = pathlib.Path(args.base_dir)
    df = pd.read_csv(args.report_csv)

    # Normaliza nomes de colunas esperadas
    required = {"stem","has_tsv","n_channels","duration_s","n_tsv_rows"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Report CSV não tem colunas esperadas: {missing}")

    df["duration_h"] = df["duration_s"] / 3600.0

    ok = (
        (df["has_tsv"] == True) &
        (df["duration_h"] >= args.min_hours) &
        (df["n_channels"].fillna(0) >= args.min_channels) &
        (df["n_tsv_rows"].fillna(0) >= args.min_tsv_rows)
    )
    good = df.loc[ok].copy()

    # Resolve caminhos reais
    edf_paths = []
    tsv_paths = []
    for stem in good["stem"]:
        edf = find_first(base, f"{stem}.edf")
        tsv = find_first(base, f"{stem}.tsv")
        edf_paths.append(edf)
        tsv_paths.append(tsv)

    good["edf_path"] = edf_paths
    good["tsv_path"] = tsv_paths

    # filtra apenas quem tem caminhos resolvidos
    good = good[(good["edf_path"]!="") & (good["tsv_path"]!="")]

    out_cols = ["stem","duration_h","n_channels","n_tsv_rows","edf_path","tsv_path"]
    good[out_cols].to_csv(args.out_manifest, index=False)
    print(f"✅ Manifest salvo: {args.out_manifest} | linhas={len(good)}")
    print("   Regras: "
          f"has_tsv=True, duration_h>={args.min_hours}, n_channels>={args.min_channels}, n_tsv_rows>={args.min_tsv_rows}")

if __name__ == "__main__":
    main()
