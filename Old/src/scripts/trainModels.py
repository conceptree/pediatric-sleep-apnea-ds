#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Treino e avaliação com features tabulares (SpO2, etc.).
- Lê um features.parquet com colunas: ['subset','label', ...features...]
- Usa 'subset' para separar train/val/test (se não existir, faz split holdout).
- Treina: Logistic Regression, RandomForest e XGBoost (se disponível).
- Faz otimização simples do threshold no VAL (youden/f1/recall opcional).
- Reporta métricas no TEST, guarda gráficos ROC/PR e modelos .joblib.

Exemplo:
python3 train_models.py \
  --features /Volumes/CORSAIR/tese/datasets/splits_tsv_thr3/features.parquet \
  --out-dir /Volumes/CORSAIR/tese/datasets/splits_tsv_thr3/reports \
  --opt-threshold youden
"""

import argparse
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report, roc_auc_score, RocCurveDisplay,
    average_precision_score, PrecisionRecallDisplay, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import joblib

# XGBoost opcional
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False
    warnings.warn("XGBoost não encontrado. O modelo XGB será ignorado.")

# -------------------- utils --------------------
def choose_threshold(y_val, p_val, mode="youden", recall_target=0.95):
    """Escolhe threshold com base no conjunto de validação."""
    if mode == "none":
        return 0.5, None

    from sklearn.metrics import roc_curve, precision_recall_curve
    if mode == "youden":
        fpr, tpr, thr = roc_curve(y_val, p_val)
        j = tpr - fpr
        k = np.argmax(j)
        return float(thr[k]), {"youden": float(j[k])}

    if mode == "f1":
        prec, rec, thr = precision_recall_curve(y_val, p_val)
        f1 = (2*prec*rec) / (prec+rec+1e-9)
        k = np.nanargmax(f1)
        # precision_recall_curve devolve thresholds com len-1
        thr_val = float(thr[max(k-1,0)]) if k < len(thr) else 0.5
        return thr_val, {"f1": float(np.nanmax(f1)), "prec": float(prec[k]), "rec": float(rec[k])}

    if mode == "recall":
        prec, rec, thr = precision_recall_curve(y_val, p_val)
        # escolher o maior threshold cujo recall >= alvo
        idx = np.where(rec >= recall_target)[0]
        if len(idx) == 0:
            return 0.5, {"recall_reached": float(np.max(rec))}
        k = idx[-1]
        thr_val = float(thr[max(k-1,0)]) if k < len(thr) else 0.5
        return thr_val, {"recall_reached": float(rec[k]), "prec": float(prec[k])}

    return 0.5, None

def eval_and_save(name, model, X_val, y_val, X_test, y_test, out_dir, thr=0.5):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Probabilidades
    p_val  = model.predict_proba(X_val)[:,1]
    p_test = model.predict_proba(X_test)[:,1]

    # Métricas com threshold
    yhat_val  = (p_val  >= thr).astype(int)
    yhat_test = (p_test >= thr).astype(int)

    # AUC / AP
    auc_val  = roc_auc_score(y_val, p_val)
    auc_test = roc_auc_score(y_test, p_test)
    ap_val   = average_precision_score(y_val, p_val)
    ap_test  = average_precision_score(y_test, p_test)

    rep_val  = classification_report(y_val,  yhat_val,  output_dict=True)
    rep_test = classification_report(y_test, yhat_test, output_dict=True)
    cm_test  = confusion_matrix(y_test, yhat_test).tolist()

    # salvar relatório JSON
    report = dict(
        model=name, threshold=thr,
        auc_val=float(auc_val), auc_test=float(auc_test),
        ap_val=float(ap_val),   ap_test=float(ap_test),
        report_val=rep_val, report_test=rep_test,
        cm_test=cm_test
    )
    with open(out_dir/f"{name}_metrics.json", "w") as f:
        json.dump(report, f, indent=2)

    # Gráficos ROC / PR
    RocCurveDisplay.from_predictions(y_test, p_test, name=f"{name} (AUC={auc_test:.3f})")
    plt.title(f"ROC — {name} (test)")
    plt.tight_layout()
    plt.savefig(out_dir/f"{name}_roc.png", dpi=120)
    plt.close()

    PrecisionRecallDisplay.from_predictions(y_test, p_test, name=f"{name} (AP={ap_test:.3f})")
    plt.title(f"Precision-Recall — {name} (test)")
    plt.tight_layout()
    plt.savefig(out_dir/f"{name}_pr.png", dpi=120)
    plt.close()

    return report

# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="features.parquet gerado pelo extractFeatures.py")
    ap.add_argument("--out-dir", required=True, help="pasta onde salvar modelos e relatórios")
    ap.add_argument("--opt-threshold", choices=["none","youden","f1","recall"], default="youden",
                    help="estratégia para escolher threshold com base no VAL (default=youden)")
    ap.add_argument("--recall-target", type=float, default=0.95, help="alvo quando --opt-threshold recall")
    ap.add_argument("--save-models", action="store_true", help="guardar modelos .joblib")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    (out_dir/"models").mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.features)

    if "label" not in df.columns:
        raise ValueError("Coluna 'label' em falta no parquet.")
    if "subset" not in df.columns:
        # fallback: cria split holdout se não existir
        train, test = train_test_split(df, test_size=0.15, random_state=42, stratify=df["label"])
        train, val  = train_test_split(train, test_size=0.15, random_state=42, stratify=train["label"])
        train["subset"] = "train"; val["subset"] = "val"; test["subset"] = "test"
        df = pd.concat([train, val, test], ignore_index=True)

    # features: remove meta
    drop_cols = [c for c in ["subset","label","stem","error"] if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df["label"].astype(int)
    subset = df["subset"]

    # seleção de conjuntos
    X_train = X[subset=="train"]; y_train = y[subset=="train"]
    X_val   = X[subset=="val"];   y_val   = y[subset=="val"]
    X_test  = X[subset=="test"];  y_test  = y[subset=="test"]

    # imputação simples
    X_train = X_train.fillna(X_train.median(numeric_only=True))
    X_val   = X_val.fillna(X_train.median(numeric_only=True))
    X_test  = X_test.fillna(X_train.median(numeric_only=True))

    print(f"Train={len(y_train)} (pos={int(y_train.sum())}) | "
          f"Val={len(y_val)} (pos={int(y_val.sum())}) | "
          f"Test={len(y_test)} (pos={int(y_test.sum())})")
    print(f"Num features: {X_train.shape[1]}")

    reports = []

    # ---------- Logistic Regression ----------
    pipe_lr = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=None))
    ])
    pipe_lr.fit(X_train, y_train)

    # escolher threshold com base no VAL
    p_val_lr = pipe_lr.predict_proba(X_val)[:,1]
    thr_lr, extra = choose_threshold(y_val.values, p_val_lr, mode=args.opt_threshold, recall_target=args.recall_target)
    if extra:
        print(f"[LogReg] threshold({args.opt_threshold})={thr_lr:.3f} | extras={extra}")

    rep_lr = eval_and_save("logreg", pipe_lr, X_val, y_val, X_test, y_test, out_dir, thr=thr_lr)
    reports.append(rep_lr)
    if args.save_models:
        joblib.dump(pipe_lr, out_dir/"models"/"logreg.joblib")

    # ---------- Random Forest ----------
    rf = RandomForestClassifier(
        n_estimators=400, random_state=42, class_weight="balanced",
        max_depth=None, min_samples_leaf=2, max_features="sqrt", n_jobs=-1
    )
    rf.fit(X_train, y_train)

    p_val_rf = rf.predict_proba(X_val)[:,1]
    thr_rf, extra = choose_threshold(y_val.values, p_val_rf, mode=args.opt_threshold, recall_target=args.recall_target)
    if extra:
        print(f"[RF] threshold({args.opt_threshold})={thr_rf:.3f} | extras={extra}")

    rep_rf = eval_and_save("random_forest", rf, X_val, y_val, X_test, y_test, out_dir, thr=thr_rf)
    reports.append(rep_rf)
    if args.save_models:
        joblib.dump(rf, out_dir/"models"/"random_forest.joblib")

    # ---------- XGBoost (opcional) ----------
    if HAS_XGB:
        xgb = XGBClassifier(
            n_estimators=800,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=42,
            eval_metric="logloss",
            n_jobs=0,
            tree_method="hist"
        )
        xgb.fit(X_train, y_train)
        p_val_xgb = xgb.predict_proba(X_val)[:,1]
        thr_xgb, extra = choose_threshold(y_val.values, p_val_xgb, mode=args.opt_threshold, recall_target=args.recall_target)
        if extra:
            print(f"[XGB] threshold({args.opt_threshold})={thr_xgb:.3f} | extras={extra}")

        rep_xgb = eval_and_save("xgboost", xgb, X_val, y_val, X_test, y_test, out_dir, thr=thr_xgb)
        reports.append(rep_xgb)
        if args.save_models:
            joblib.dump(xgb, out_dir/"models"/"xgboost.joblib")

    # ---------- ranking final ----------
    df_rep = pd.DataFrame([{
        "model": r["model"],
        "thr": r["threshold"],
        "AUC_val": r["auc_val"], "AUC_test": r["auc_test"],
        "AP_val": r["ap_val"],   "AP_test": r["ap_test"],
        "F1_test": r["report_test"]["weighted avg"]["f1-score"],
        "Accuracy_test": r["report_test"]["accuracy"]
    } for r in reports]).sort_values("AUC_test", ascending=False)
    df_rep.to_csv(out_dir/"model_comparison.csv", index=False)
    print("\n=== Model comparison (sorted by AUC_test) ===")
    print(df_rep.to_string(index=False))

if __name__ == "__main__":
    main()