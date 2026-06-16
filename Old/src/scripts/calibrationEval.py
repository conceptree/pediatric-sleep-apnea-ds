#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import joblib, pandas as pd, numpy as np
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
import matplotlib.pyplot as plt

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--subset", default="test")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--calibrate", choices=["none","isotonic","sigmoid"], default="none")
    args=ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.features)
    drop = ["subset","label","stem","prob_pos"]
    X = df.drop(columns=[c for c in drop if c in df.columns], errors="ignore")
    y = df["label"].astype(int)
    mask = (df["subset"]==args.subset)
    Xs, ys = X[mask], y[mask]

    base = joblib.load(args.model)
    clf = base
    if args.calibrate!="none":
        # calibrar em val e avaliar em test para evitar overfit
        mval = (df["subset"]=="val")
        Xv, yv = X[mval], y[mval]
        calib = CalibratedClassifierCV(base, method=args.calibrate, cv="prefit")
        calib.fit(Xv, yv)
        clf = calib
        joblib.dump(clf, out/f"calibrated_{args.calibrate}.joblib")

    prob = clf.predict_proba(Xs)[:,1]
    brier = brier_score_loss(ys, prob)
    frac_pos, mean_pred = calibration_curve(ys, prob, n_bins=10, strategy="quantile")

    with open(out/"calibration.txt","w") as f:
        f.write(f"Brier({args.subset}) = {brier:.4f}\n")
        f.write("Bins (mean_pred, frac_pos):\n")
        for mp, fp in zip(mean_pred, frac_pos):
            f.write(f"{mp:.3f}\t{fp:.3f}\n")
    plt.figure(figsize=(5,5))
    plt.plot([0,1],[0,1],"--",label="perfect")
    plt.plot(mean_pred, frac_pos, marker="o")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed fraction positive")
    ttl = f"Reliability — {args.subset} (Brier={brier:.3f})"
    if args.calibrate!="none": ttl += f" | {args.calibrate}"
    plt.title(ttl); plt.tight_layout()
    plt.savefig(out/f"reliability_{args.subset}.png", dpi=180); plt.close()
    print(f"✅ calibração ({args.subset}) -> {out}")

if __name__=="__main__":
    main()
