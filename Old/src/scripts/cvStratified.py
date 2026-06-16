#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from xgboost import XGBClassifier

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out-json", required=True)
    args=ap.parse_args()

    df = pd.read_parquet(args.features)
    # usar apenas train+val para CV; deixar test como hold-out
    df = df[df["subset"].isin(["train","val"])].copy()

    X = df.drop(columns=[c for c in ["subset","label","stem","prob_pos"] if c in df.columns], errors="ignore")
    y = df["label"].astype(int).values

    skf = StratifiedKFold(n_splits=args.k, shuffle=True, random_state=42)
    rows=[]
    for i,(tr,va) in enumerate(skf.split(X,y),1):
        Xtr, Xva = X.iloc[tr], X.iloc[va]
        ytr, yva = y[tr], y[va]
        clf = XGBClassifier(
            n_estimators=400, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            n_jobs=4, eval_metric="logloss", random_state=42
        )
        clf.fit(Xtr, ytr)
        p = clf.predict_proba(Xva)[:,1]
        auc  = roc_auc_score(yva, p)
        ap   = average_precision_score(yva, p)
        f1   = f1_score(yva, (p>=0.5).astype(int))
        rows.append(dict(fold=i, AUC=auc, AP=ap, F1=f1, n_val=len(yva), pos=int(yva.sum())))
        print(f"fold {i}: AUC={auc:.3f} AP={ap:.3f} F1={f1:.3f} n={len(yva)} pos={int(yva.sum())}")

    res = pd.DataFrame(rows)
    summary = {
        "k": args.k,
        "AUC_mean": float(res["AUC"].mean()), "AUC_std": float(res["AUC"].std()),
        "AP_mean":  float(res["AP"].mean()),  "AP_std":  float(res["AP"].std()),
        "F1_mean":  float(res["F1"].mean()),  "F1_std":  float(res["F1"].std()),
        "folds": rows
    }
    Path(args.out_json).write_text(json.dumps(summary, indent=2))
    print(f"✅ CV summary salvo em {args.out_json}")

if __name__=="__main__":
    main()
