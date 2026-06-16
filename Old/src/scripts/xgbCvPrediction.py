#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, pathlib
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    classification_report, precision_recall_curve, roc_curve
)
import joblib
import matplotlib.pyplot as plt

def _load_threshold_from_file(path: str | pathlib.Path) -> float | None:
    p = pathlib.Path(path)
    if not p.exists():
        print(f"⚠️  threshold-file não existe: {p}")
        return None
    try:
        with open(p, "r") as f:
            data = json.load(f)
        # tenta em ordem: best_threshold, threshold, best_row.thr
        for key in ["best_threshold", "threshold"]:
            if key in data and isinstance(data[key], (int, float)):
                return float(data[key])
        # nested
        if "best_row" in data and isinstance(data["best_row"], dict):
            thr = data["best_row"].get("thr", None)
            if isinstance(thr, (int, float)):
                return float(thr)
    except Exception as e:
        print(f"⚠️  Falha a ler threshold-file: {e}")
    return None

def _choose_threshold_from_val(df_val: pd.DataFrame, mode: str, recall_target: float|None, precision_target: float|None) -> float:
    y = df_val["label"].values
    s = df_val["prob_pos"].values
    prec, rec, thr = precision_recall_curve(y, s)
    thr = np.append(thr, 1.0)  # alinhar tamanhos com prec/rec

    if mode == "youden":
        # Youden J em ROC
        fpr, tpr, thr_roc = roc_curve(y, s)
        youden = tpr - fpr
        return float(thr_roc[np.argmax(youden)])
    elif mode == "f1":
        f1 = 2 * prec * rec / (prec + rec + 1e-12)
        return float(thr[np.nanargmax(f1)])
    elif mode == "recall":
        target = recall_target if recall_target is not None else 0.95
        ok = np.where(rec >= target)[0]
        if len(ok):
            return float(thr[ok[np.argmax(prec[ok])]])  # entre os com recall>=target, escolhe maior precisão
        return 0.5
    elif mode == "recall_at_precision":
        target = precision_target if precision_target is not None else 0.95
        ok = np.where(prec >= target)[0]
        if len(ok):
            return float(thr[ok[np.argmax(rec[ok])]])  # entre os com precisão>=target, maior recall
        return 0.5
    else:
        return 0.5

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval-subset", choices=["train","val","test","all"], default="test")

    # thresholds
    ap.add_argument("--manual-threshold", type=float, default=None,
                    help="Se definido, usa este threshold diretamente.")
    ap.add_argument("--threshold-file", type=str, default=None,
                    help="JSON com best_threshold/threshold/best_row.thr.")
    ap.add_argument("--opt-threshold", choices=["none","youden","f1","recall","recall_at_precision"], default="none")
    ap.add_argument("--recall-target", type=float, default=None)
    ap.add_argument("--precision-target", type=float, default=None)

    ap.add_argument("--save-meta", type=str, default=None)
    ap.add_argument("--save-curves-prefix", type=str, default=None)
    args = ap.parse_args()

    # Carrega features e separa subset
    df = pd.read_parquet(args.features)
    if args.eval_subset != "all":
        df_eval = df[df["subset"] == args.eval_subset].copy()
    else:
        df_eval = df.copy()

    # X/y
    # meta (removidas do X)
    meta_cols = ["subset", "split", "label", "stem"]  # <- adicionámos "split"
    meta = df_eval[[c for c in meta_cols if c in df_eval.columns]].copy()

    # features numéricas apenas
    X = df_eval.drop(columns=meta_cols, errors="ignore")
    # opcional/robusto: força tipos numéricos quando possível
    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.fillna(0.0)
    y = df_eval["label"].values

    # Carrega modelo
    model = joblib.load(args.model)

    # Probabilidades
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X)[:, 1]
    else:
        # XGBoost pode usar predict com output='probability' dependendo de versão
        y_proba = model.predict(X)
        # garante 1D
        y_proba = np.asarray(y_proba).ravel()

    meta["prob_pos"] = y_proba

    # Determina threshold
    thr = 0.5
    source = "default(0.5)"
    if args.manual_threshold is not None:
        thr = float(args.manual_threshold); source = "manual"
    elif args.threshold_file:
        t = _load_threshold_from_file(args.threshold_file)
        if t is not None:
            thr = float(t); source = f"file({args.threshold_file})"
        else:
            print("⚠️  threshold-file sem chave válida; a usar 0.5.")
    elif args.opt_threshold != "none":
        if args.eval_subset == "val":
            thr = _choose_threshold_from_val(meta.assign(label=y), args.opt_threshold, args.recall_target, args.precision_target)
            source = f"val::{args.opt_threshold}"
        else:
            print("ℹ️  Otimização de threshold requer subset=val; a usar 0.5.")

    # Predições binárias
    meta["pred"] = (meta["prob_pos"] >= thr).astype(int)

    # Métricas
    auc = roc_auc_score(y, y_proba)
    ap_score = average_precision_score(y, y_proba)
    print(f"✅ Previsões salvas em: {args.out}  |  linhas={len(meta)}")
    print(f"ℹ️  Threshold usado: {thr:.3f} ({source})")
    print(f"📊 Métricas (subset={args.eval_subset}): AUC={auc:.3f} | AP={ap_score:.3f}")
    print(classification_report(y, meta["pred"], digits=3))

    # Salva CSV
    out_p = pathlib.Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    meta.to_csv(out_p, index=False)

    # Meta JSON
    if args.save_meta:
        info = dict(
            subset=args.eval_subset,
            threshold=thr,
            threshold_source=source,
            auc=auc,
            average_precision=ap_score,
            n=len(meta),
            positives=int(meta["label"].sum()),
            negatives=int(len(meta) - meta["label"].sum())
        )
        with open(args.save_meta, "w") as f:
            json.dump(info, f, indent=2)
        print(f"📝 Meta salva em: {args.save_meta}")

    # Curvas (opcional)
    if args.save_curves_prefix:
        pref = pathlib.Path(args.save_curves_prefix)
        pref.parent.mkdir(parents=True, exist_ok=True)
        # ROC
        fpr, tpr, _ = roc_curve(y, y_proba)
        plt.figure()
        plt.plot(fpr, tpr)
        plt.plot([0,1],[0,1],'--')
        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC ({args.eval_subset}) AUC={auc:.3f}")
        plt.savefig(pref.with_suffix(".roc.png")); plt.close()
        # PR
        prec, rec, _ = precision_recall_curve(y, y_proba)
        plt.figure()
        plt.plot(rec, prec)
        plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR ({args.eval_subset}) AP={ap_score:.3f}")
        plt.savefig(pref.with_suffix(".pr.png")); plt.close()
        print(f"🖼️  Curvas salvas com prefixo: {pref}")

if __name__ == "__main__":
    main()