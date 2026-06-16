#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import pathlib

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("report_dir", help="Pasta onde está o report.csv gerado pelo verify.py")
    args = ap.parse_args()

    report_path = pathlib.Path(args.report_dir) / "report.csv"
    if not report_path.exists():
        print(f"❌ Report not found: {report_path}")
        return

    df = pd.read_csv(report_path)
    print("\n=== 📊 VERIFY REPORT SUMMARY ===")
    print(f"📁 Report path          : {report_path}")
    print(f"📦 Total studies        : {len(df)}")

    # checagens básicas com colunas disponíveis
    if "has_tsv" in df.columns:
        print(f"✔️  With TSV            : {df['has_tsv'].sum()} ({df['has_tsv'].mean():.1%})")

    if "duration_s" in df.columns:
        df["duration_h"] = df["duration_s"] / 3600
        print(f"📏 >6h duration         : {(df['duration_h'] > 6).sum()} studies")
        print(f"⚠️  <1h duration         : {(df['duration_h'] < 1).sum()} studies")
        print(f"⏱️  Duration (mean)     : {df['duration_h'].mean():.2f} h")

    if "n_channels" in df.columns:
        print(f"🧠 Avg channels         : {df['n_channels'].dropna().mean():.1f}")

    if "n_tsv_rows" in df.columns:
        print(f"📄 Avg TSV rows         : {df['n_tsv_rows'].dropna().mean():.0f}")

if __name__ == "__main__":
    main()
