#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    precision_recall_fscore_support, confusion_matrix, accuracy_score
)

def metrics_from_df(df):
    y = df["label"].astype(int).values
    p = df["prob_pos"].astype(float).values
    yhat = df["pred"].astype(int).values if "pred" in df.columns else (p >= 0.5).astype(int)

    auc = roc_auc_score(y, p)
    ap  = average_precision_score(y, p)
    brier = brier_score_loss(y, p)
    acc = accuracy_score(y, yhat)
    prec, rec, f1, _ = precision_recall_fscore_support(y, yhat, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, yhat).ravel()

    return dict(AUC=auc, AP=ap, Brier=brier, Accuracy=acc,
                Precision=prec, Recall=rec, F1=f1,
                TP=int(tp), FP=int(fp), TN=int(tn), FN=int(fn),
                Support=int(len(y)))

def main():
    ap = argparse.ArgumentParser(description="Collect and compare metrics from two prediction CSVs.")
    ap.add_argument("--pred-a", required=True, help="CSV do Modelo A (colunas: subset,label,prob_pos[,pred])")
    ap.add_argument("--pred-b", required=True, help="CSV do Modelo B")
    ap.add_argument("--label-a", default="Model A")
    ap.add_argument("--label-b", default="Model B")
    ap.add_argument("--subset", default="test", choices=["train","val","test","all"])
    ap.add_argument("--out-csv", required=True, help="Caminho do CSV de saída com a tabela comparativa")
    args = ap.parse_args()

    dfA = pd.read_csv(args.pred_a)
    dfB = pd.read_csv(args.pred_b)
    if args.subset != "all":
        dfA = dfA[dfA["subset"] == args.subset]
        dfB = dfB[dfB["subset"] == args.subset]

    mA = metrics_from_df(dfA)
    mB = metrics_from_df(dfB)

    out_df = pd.DataFrame([
        dict(model=args.label_a, subset=args.subset, **mA),
        dict(model=args.label_b, subset=args.subset, **mB),
    ])

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False)
    print("=== Metrics comparison ===")
    print(out_df.round(3).to_string(index=False))
    print(f"\n💾 saved: {out}")

if __name__ == "__main__":
    main()