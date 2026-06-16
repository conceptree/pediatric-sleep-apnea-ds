#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

def load_preds(path: Path, subset: str | None):
    df = pd.read_csv(path)
    needed = {"stem", "label"}
    if not needed.issubset(df.columns):
        raise ValueError(f"{path} precisa ter colunas {needed}. Tem: {df.columns.tolist()}")

    # Se existir prob_pos e não houver 'pred', faremos threshold=0.5
    has_probs = "prob_pos" in df.columns
    has_pred  = "pred" in df.columns

    # Filtrar subset se existir coluna
    if subset is not None and "subset" in df.columns:
        df = df[df["subset"] == subset].copy()

    # Se não houver 'pred', criar com base em prob_pos
    if not has_pred:
        if not has_probs:
            raise ValueError(f"{path} não tem 'pred' nem 'prob_pos' para calcular predições.")
        df["pred"] = (df["prob_pos"] >= 0.5).astype(int)

    # Garantir tipos
    df["label"] = df["label"].astype(int)
    df["pred"]  = df["pred"].astype(int)
    if has_probs:
        df["prob_pos"] = df["prob_pos"].astype(float)
    else:
        # Se não houver prob_pos, criar dummy (para permitir métricas sem AUC/AP/Brier)
        df["prob_pos"] = np.nan

    return df

def compute_metrics(df: pd.DataFrame, model_name: str, subset: str | None):
    y = df["label"].to_numpy()
    yhat = df["pred"].to_numpy()
    p = df["prob_pos"].to_numpy()

    # Métricas com probabilidades (se existirem valores válidos)
    mask_valid = ~np.isnan(p)
    has_valid_probs = mask_valid.any()

    auc = average_precision = brier = np.nan
    if has_valid_probs:
        try:
            auc = roc_auc_score(y[mask_valid], p[mask_valid])
        except Exception:
            auc = np.nan
        try:
            average_precision = average_precision_score(y[mask_valid], p[mask_valid])
        except Exception:
            average_precision = np.nan
        try:
            brier = brier_score_loss(y[mask_valid], p[mask_valid])
        except Exception:
            brier = np.nan

    acc = accuracy_score(y, yhat)
    prec = precision_score(y, yhat, zero_division=0)
    rec = recall_score(y, yhat, zero_division=0)
    f1 = f1_score(y, yhat, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0,1]).ravel()

    row = {
        "model": model_name,
        "subset": subset if subset else (df["subset"].iloc[0] if "subset" in df.columns and not df.empty else "all"),
        "AUC": round(auc, 3) if not np.isnan(auc) else np.nan,
        "AP": round(average_precision, 3) if not np.isnan(average_precision) else np.nan,
        "Brier": round(brier, 3) if not np.isnan(brier) else np.nan,
        "Accuracy": round(acc, 3),
        "Precision": round(prec, 3),
        "Recall": round(rec, 3),
        "F1": round(f1, 3),
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
        "Support": int(len(df)),
    }
    return row, (tn, fp, fn, tp)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", nargs="+", required=True, help="Lista de CSVs de predições")
    ap.add_argument("--subset", choices=["train","val","test"], default="test", help="Subset a avaliar (se existir coluna 'subset')")
    ap.add_argument("--out", required=True, help="CSV de saída com as métricas")
    ap.add_argument("--save-confusions-dir", default=None, help="(Opcional) pasta para salvar matrizes de confusão individuais")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.save_confusions_dir:
        conf_dir = Path(args.save_confusions_dir)
        conf_dir.mkdir(parents=True, exist_ok=True)
    else:
        conf_dir = None

    rows = []
    for pred_file in args.preds:
        pred_path = Path(pred_file)
        model_name = pred_path.stem  # usa o nome do ficheiro como rótulo
        df = load_preds(pred_path, args.subset)
        row, cm = compute_metrics(df, model_name, args.subset)
        rows.append(row)

        # Guardar matriz de confusão por modelo (opcional)
        if conf_dir is not None:
            tn, fp, fn, tp = cm
            pd.DataFrame(
                [[tn, fp],[fn, tp]],
                index=["true_0","true_1"], columns=["pred_0","pred_1"],
            ).to_csv(conf_dir / f"{model_name}_confusion_{args.subset}.csv", index=True)

    table = pd.DataFrame(rows)
    # ordenar por AUC desc, depois por AP desc
    sort_cols = [c for c in ["AUC","AP"] if c in table.columns]
    if sort_cols:
        table = table.sort_values(sort_cols, ascending=False)

    table.to_csv(out_path, index=False)

    # Print amigável no terminal
    print("\n=== Metrics comparison ===")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(table.to_string(index=False))

if __name__ == "__main__":
    main()
