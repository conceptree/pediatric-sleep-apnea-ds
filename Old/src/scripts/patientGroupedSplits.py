#!/usr/bin/env python3
# file: makePatientGroupedSplits.py
import argparse, json, math, random
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

def patient_label(df):
    # label do paciente = max dos seus exames (se 1 em algum exame, paciente=1)
    return df.groupby("patient")["label"].max()

def split_by_patient(df, train_size=0.7, val_size=0.15, test_size=0.15, tol=0.02, tries=200, seed0=42):
    assert abs(train_size + val_size + test_size - 1.0) < 1e-9
    groups = df["patient"].unique()
    # label por paciente
    y_pat = patient_label(df)
    target_pos = y_pat.mean()

    best = None
    rng = random.Random(seed0)
    for t in range(tries):
        seed = rng.randrange(10**9)
        gss = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=seed)
        train_idx, hold_idx = next(gss.split(np.zeros(len(df)), groups=df["patient"]))
        train_pats = set(df.iloc[train_idx]["patient"].unique())
        hold_df = df.iloc[hold_idx]
        # split val/test a partir do hold
        gss2 = GroupShuffleSplit(n_splits=1, train_size=val_size/(val_size+test_size), random_state=seed+1)
        v_idx, te_idx = next(gss2.split(np.zeros(len(hold_df)), groups=hold_df["patient"]))
        val_pats = set(hold_df.iloc[v_idx]["patient"].unique())
        test_pats = set(hold_df.iloc[te_idx]["patient"].unique())

        # métricas por subset no nível de paciente
        def pos_rate(pats):
            return y_pat.loc[list(pats)].mean() if len(pats) else np.nan
        pr_tr, pr_va, pr_te = pos_rate(train_pats), pos_rate(val_pats), pos_rate(test_pats)

        # desvio total do alvo global
        dev = sum(abs(x - target_pos) for x in [pr_tr, pr_va, pr_te])
        cand = (dev, train_pats, val_pats, test_pats, (pr_tr, pr_va, pr_te), seed)
        if (best is None) or (dev < best[0]): best = cand
        if all(abs(x - target_pos) <= tol for x in [pr_tr, pr_va, pr_te]):
            best = cand
            break
    return best

def main():
    ap = argparse.ArgumentParser(description="Cria splits por PACIENTE (evita leakage).")
    ap.add_argument("--features", required=True, help="features.parquet com colunas [stem,label,subset,...]")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--train", type=float, default=0.7)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--tol", type=float, default=0.02, help="tolerância de diferença na taxa de positivos por subset")
    ap.add_argument("--tries", type=int, default=200)
    args = ap.parse_args()

    df = pd.read_parquet(args.features)[["stem","label"]].copy()
    assert "stem" in df.columns and "label" in df.columns, "features precisa ter stem e label"
    df["patient"] = df["stem"].str.split("_").str[0]
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    dev, train_pats, val_pats, test_pats, pr, seed = split_by_patient(
        df, train_size=args.train, val_size=args.val, test_size=args.test,
        tol=args.tol, tries=args.tries
    )
    pr_tr, pr_va, pr_te = pr
    y_pat = patient_label(df)
    msg = {
        "seed": seed,
        "patient_counts": {
            "train": len(train_pats), "val": len(val_pats), "test": len(test_pats)
        },
        "patient_pos_rate": {
            "global": float(y_pat.mean()),
            "train": float(pr_tr), "val": float(pr_va), "test": float(pr_te)
        },
        "deviation_sum": float(dev)
    }

    # montar CSVs de stems
    df["subset"] = np.where(df["patient"].isin(train_pats), "train",
                     np.where(df["patient"].isin(val_pats), "val", "test"))

    for sub in ["train","val","test"]:
        part = df[df["subset"]==sub][["stem","label"]].sort_values("stem")
        part.to_csv(out / f"{sub}.csv", index=False)
        print(f"✅ {sub}.csv | rows={len(part)} | pos={int(part['label'].sum())} neg={len(part)-int(part['label'].sum())}")

    # relatório JSON
    (out / "split_report.json").write_text(json.dumps(msg, indent=2))
    print(f"📝 split_report.json salvo em {out}")

if __name__ == "__main__":
    main()