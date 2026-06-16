#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
calibrateAndPredict.py

- Lê features.parquet e um modelo .joblib (XGB, RF, etc.)
- Calibração no conjunto VAL: raw (sem), platt (sigmoid), isotonic
  * Implementada manualmente para funcionar com classificadores OU regresssores
- Escolha de threshold no VAL:
  * recall_at_precision (alvo via --target)
  * youden
- Avalia em val/test/all e guarda:
  * CSV de previsões (prob_pos, pred, subset)
  * JSON meta (AUC/AP/threshold/report)
  * Curvas ROC/PR (PNG) para VAL e subset(s) avaliados
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    auc,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    classification_report,
)
import matplotlib.pyplot as plt


# ----------------------------- utils -----------------------------

def find_split_col(df: pd.DataFrame) -> str:
    if "split" in df.columns:
        return "split"
    if "subset" in df.columns:
        return "subset"
    raise ValueError("Nenhuma coluna de split encontrada (esperava 'split' ou 'subset').")

def select_xy(df: pd.DataFrame, split_col: str, split_value: str):
    part = df[df[split_col] == split_value].copy()
    if part.empty:
        raise ValueError(f"Subset '{split_value}' não encontrado/está vazio.")
    y = part["label"].astype(int).values
    drop_cols = {"label", "stem", split_col}
    X = part.drop(columns=[c for c in drop_cols if c in part.columns], errors="ignore")
    X = X.select_dtypes(include=[np.number])
    return X, y, part

def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def plot_roc_pr(y_true, y_prob, title_prefix: str, out_prefix: Path):
    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    ensure_dir(out_prefix.with_suffix(".png"))
    plt.figure()
    plt.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0,1],[0,1], linestyle="--")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"{title_prefix} - ROC")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_prefix.with_name(out_prefix.name + "_roc.png"))
    plt.close()

    # PR
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    pr_ap = average_precision_score(y_true, y_prob)
    plt.figure()
    plt.plot(rec, prec, lw=2, label=f"AP = {pr_ap:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"{title_prefix} - Precision-Recall")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(out_prefix.with_name(out_prefix.name + "_pr.png"))
    plt.close()

    return roc_auc, pr_ap

def youden_threshold(y_true, y_prob):
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    J = tpr - fpr
    i = np.argmax(J)
    return float(thr[i])

def recall_at_precision_threshold(y_true, y_prob, precision_target: float):
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    thr = np.r_[thr, 1.0]  # alinhar tamanhos
    mask = prec >= precision_target
    if not np.any(mask):
        return 0.5
    idx = np.argmax(np.where(mask, rec, -1))
    return float(thr[idx])

def apply_threshold(y_prob, thr):
    return (y_prob >= thr).astype(int)

def metrics_text(y_true, y_prob, y_pred):
    return classification_report(y_true, y_pred, digits=3, output_dict=False)

def _predict_proba_raw(model, X: pd.DataFrame) -> np.ndarray:
    """Obtém prob_pos 'brutas' do modelo:
       - se tiver predict_proba -> usa [:,1]
       - senão, usa predict() e faz clip para [0,1]
    """
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        if isinstance(p, np.ndarray) and p.ndim == 2 and p.shape[1] >= 2:
            prob = p[:, 1]
        else:
            # alguns modelos retornam 1D
            prob = np.asarray(p).ravel()
    else:
        prob = np.asarray(model.predict(X)).ravel()
    # garantir [0,1]
    prob = np.clip(prob, 0.0, 1.0)
    return prob

class Calibrator:
    """Wrapper simples para calibração raw/platt/isotonic sobre probabilidades brutas."""
    def __init__(self, method="raw"):
        self.method = method
        self.iso = None
        self.lr = None

    def fit(self, p_raw: np.ndarray, y: np.ndarray):
        p_raw = np.asarray(p_raw).ravel()
        y = np.asarray(y).ravel()
        if self.method == "raw":
            return self
        elif self.method == "isotonic":
            self.iso = IsotonicRegression(out_of_bounds="clip")
            self.iso.fit(p_raw, y)
        elif self.method == "platt":
            self.lr = LogisticRegression(max_iter=1000)
            self.lr.fit(p_raw.reshape(-1,1), y)
        else:
            raise ValueError(f"Método de calibração desconhecido: {self.method}")
        return self

    def transform(self, p_raw: np.ndarray) -> np.ndarray:
        p_raw = np.asarray(p_raw).ravel()
        if self.method == "raw":
            return p_raw
        elif self.method == "isotonic":
            return self.iso.transform(p_raw)
        elif self.method == "platt":
            return self.lr.predict_proba(p_raw.reshape(-1,1))[:,1]
        else:
            raise ValueError(f"Método de calibração desconhecido: {self.method}")


# ----------------------------- main -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="features.parquet")
    ap.add_argument("--model", required=True, help="modelo .joblib (XGB, RF, etc.)")
    ap.add_argument("--out-prefix", required=True, help="prefixo de saída (sem extensão)")
    ap.add_argument("--calibration", choices=["raw", "platt", "isotonic"], default="raw")
    ap.add_argument("--objective", choices=["recall_at_precision", "youden"], default="recall_at_precision")
    ap.add_argument("--target", type=float, default=0.95, help="alvo (precision mínima em recall_at_precision)")
    ap.add_argument("--eval-subset", choices=["val", "test", "all"], default="test")
    ap.add_argument("--n-bins", type=int, default=10, help="(reservado)")
    args = ap.parse_args()

    features_path = Path(args.features)
    out_prefix = Path(args.out_prefix)

    # 1) Load data
    df = pd.read_parquet(features_path)
    split_col = find_split_col(df)

    # 2) Split sets
    X_val, y_val, df_val = select_xy(df, split_col, "val")
    have_test = (df[split_col] == "test").any()
    if args.eval_subset in ("test", "all") and not have_test:
        raise ValueError("Não existe subset 'test' nas features.")

    # 3) Load model
    model = joblib.load(args.model)

    # 4) Probabilidades brutas no VAL
    p_val_raw = _predict_proba_raw(model, X_val)

    # 5) Calibração por fora (se pedida)
    calibrator = Calibrator(method=args.calibration).fit(p_val_raw, y_val)
    p_val = calibrator.transform(p_val_raw)

    # 6) Threshold no VAL
    if args.objective == "youden":
        thr = youden_threshold(y_val, p_val)
        thr_info = f"youden -> {thr:.3f}"
    else:
        thr = recall_at_precision_threshold(y_val, p_val, precision_target=args.target)
        thr_info = f"recall_at_precision target={args.target} -> {thr:.3f}"

    # 7) Curvas/metricas VAL
    val_auc, val_ap = plot_roc_pr(
        y_val, p_val, f"VAL ({args.calibration})", out_prefix.with_name(out_prefix.name + "_val")
    )
    y_val_pred = apply_threshold(p_val, thr)
    rep_val = metrics_text(y_val, p_val, y_val_pred)

    # Guardar VAL CSV e meta
    ensure_dir(out_prefix.with_suffix(".json"))
    out_val_csv = out_prefix.with_name(out_prefix.name + "_val_predictions.csv")
    df_out_val = df_val[["stem", "label"]].copy() if "stem" in df_val.columns else pd.DataFrame(index=df_val.index)
    if "label" in df_val.columns and "stem" not in df_out_val.columns:
        df_out_val["label"] = df_val["label"].values
    df_out_val["prob_pos"] = p_val
    df_out_val["pred"] = y_val_pred
    df_out_val["subset"] = "val"
    df_out_val.to_csv(out_val_csv, index=False)

    meta_val = {
        "subset": "val",
        "calibration": args.calibration,
        "objective": args.objective,
        "target": args.target,
        "threshold": float(thr),
        "AUC": float(val_auc),
        "AP": float(val_ap),
        "report": rep_val
    }
    with open(out_prefix.with_name(out_prefix.name + "_val_meta.json"), "w") as f:
        json.dump(meta_val, f, indent=2)

    print(f"✅ VAL: threshold={thr_info} | AUC={val_auc:.3f} AP={val_ap:.3f}")
    print(rep_val)

    # 8) Avaliar subset(s) pedidos
    subsets_to_eval = ["test"] if args.eval_subset == "test" else (["val"] if args.eval_subset == "val" else ["val", "test"])

    for subset in subsets_to_eval:
        if subset == "test":
            X_eval, y_eval, df_eval = select_xy(df, split_col, "test")
            p_eval_raw = _predict_proba_raw(model, X_eval)
            p_eval = calibrator.transform(p_eval_raw)
        else:
            X_eval, y_eval, df_eval = X_val, y_val, df_val
            p_eval = p_val

        y_pred = apply_threshold(p_eval, thr)

        # curvas + métricas
        eval_auc, eval_ap = plot_roc_pr(
            y_eval, p_eval, f"{subset.upper()} ({args.calibration})", out_prefix.with_name(out_prefix.name + f"_{subset}")
        )
        rep = classification_report(y_eval, y_pred, digits=3)

        # CSV
        out_csv = out_prefix.with_name(out_prefix.name + f"_{subset}_predictions.csv")
        df_out = df_eval[["stem", "label"]].copy() if "stem" in df_eval.columns else pd.DataFrame(index=df_eval.index)
        if "label" in df_eval.columns and "stem" not in df_out.columns:
            df_out["label"] = df_eval["label"].values
        df_out["prob_pos"] = p_eval
        df_out["pred"] = y_pred
        df_out["subset"] = subset
        df_out.to_csv(out_csv, index=False)

        # META
        meta = {
            "subset": subset,
            "calibration": args.calibration,
            "objective": args.objective,
            "target": args.target,
            "threshold": float(thr),
            "AUC": float(eval_auc),
            "AP": float(eval_ap),
            "report": rep
        }
        with open(out_prefix.with_name(out_prefix.name + f"_{subset}_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        print(f"✅ {subset.upper()}: saved {out_csv}")


if __name__ == "__main__":
    main()