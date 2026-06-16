#!/usr/bin/env python3
import argparse, pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

def expected_calibration_error(y_true, y_prob, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins+1)
    idx = np.digitize(y_prob, bins) - 1
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            conf = y_prob[mask].mean()
            acc = y_true[mask].mean()
            ece += (mask.mean()) * abs(acc - conf)
    return ece

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-csv", required=True, help="CSV com colunas: subset,label,prob_pos")
    ap.add_argument("--subset", default="test", choices=["train","val","test","all"])
    ap.add_argument("--n-bins", type=int, default=10)
    ap.add_argument("--out-png", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.pred_csv)
    if args.subset != "all" and "subset" in df.columns:
        df = df[df["subset"] == args.subset].copy()

    y = df["label"].values.astype(int)
    p = df["prob_pos"].values.astype(float)

    bs = brier_score_loss(y, p)
    ece = expected_calibration_error(y, p, n_bins=args.n_bins)

    frac_pos, mean_pred = calibration_curve(y, p, n_bins=args.n_bins, strategy="uniform")

    plt.figure(figsize=(6,6))
    # reliability
    plt.plot([0,1],[0,1], linestyle="--", label="Perfectly calibrated")
    plt.plot(mean_pred, frac_pos, marker="o", linewidth=1.5, label="Model")
    # histogram
    ax2 = plt.twinx()
    ax2.hist(p, bins=20, alpha=0.2)
    ax2.set_ylabel("Count")

    plt.title(f"Calibration ({args.subset}) | Brier={bs:.3f} | ECE={ece:.3f}")
    plt.xlabel("Predicted probability (positive)")
    plt.ylabel("Observed frequency")
    plt.legend(loc="upper left")
    pathlib.Path(args.out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=150)
    print(f"🖼️  saved: {args.out_png} | Brier={bs:.3f} | ECE={ece:.3f}")

if __name__ == "__main__":
    main()