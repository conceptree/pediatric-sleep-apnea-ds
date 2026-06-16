#!/usr/bin/env python3
# file: check_split_leakage.py
import argparse
import pandas as pd

def main():
    ap = argparse.ArgumentParser(description="Detecta leakage entre splits (stems/pacientes).")
    ap.add_argument("--features", required=True, help="features.parquet")
    args = ap.parse_args()

    df = pd.read_parquet(args.features)[["stem","subset"]]
    df["patient"] = df["stem"].str.split("_").str[0]

    msgs = []

    # stems repetidos em mais de um subset
    stems_multi = df.groupby("stem")["subset"].nunique()
    bad_stems = stems_multi[stems_multi > 1].index.tolist()
    if bad_stems:
        msgs.append(f"❌ stems em múltiplos subsets: {len(bad_stems)} (ex: {bad_stems[:5]})")
    else:
        msgs.append("🟢 sem stems repetidos entre subsets")

    # pacientes distribuídos em mais de um subset
    byp = df.groupby(["patient","subset"]).size().unstack(fill_value=0)
    patients_multi = byp[(byp > 0).sum(axis=1) > 1].index.tolist()
    if patients_multi:
        msgs.append(f"⚠️ pacientes em múltiplos subsets: {len(patients_multi)} (ex: {patients_multi[:5]})")
    else:
        msgs.append("🟢 sem paciente distribuído entre subsets")

    print("\n".join(msgs))

if __name__ == "__main__":
    main()