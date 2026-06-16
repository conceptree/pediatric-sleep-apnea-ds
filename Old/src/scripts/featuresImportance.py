#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import joblib, pandas as pd, numpy as np
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--subset", default="test")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--do-shap", action="store_true")
    args=ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.features)
    drop = ["subset","label","stem","prob_pos"]
    X = df.drop(columns=[c for c in drop if c in df.columns], errors="ignore")
    y = df["label"].astype(int)
    mask = (df["subset"]==args.subset)
    Xs, ys = X[mask], y[mask]

    model = joblib.load(args.model)
    # permutation
    pi = permutation_importance(model, Xs, ys, n_repeats=20, random_state=42, scoring="roc_auc")
    imp = pd.DataFrame({
        "feature": Xs.columns,
        "importance_mean": pi.importances_mean,
        "importance_std": pi.importances_std
    }).sort_values("importance_mean", ascending=False)
    imp.to_csv(out/"permutation_importance.csv", index=False)
    top = imp.head(20)
    plt.figure(figsize=(7,6))
    plt.barh(top["feature"][::-1], top["importance_mean"][::-1])
    plt.title("Permutation importance (AUC drop)")
    plt.tight_layout()
    plt.savefig(out/"permutation_importance.png", dpi=180)
    plt.close()
    print(f"✅ permutation -> {out/'permutation_importance.csv'}")

    # SHAP opcional (melhor no XGBoost)
    if args.do_shap:
        try:
            import shap
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(Xs)
            shap.summary_plot(shap_vals, Xs, show=False)
            plt.tight_layout(); plt.savefig(out/"shap_summary.png", dpi=180); plt.close()
            print(f"✅ shap -> {out/'shap_summary.png'}")
        except Exception as e:
            print(f"⚠️ SHAP falhou/indisponível: {e}")

if __name__=="__main__":
    main()
