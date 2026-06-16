#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import (
    classification_report, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    precision_score, recall_score
)

# ---------- utils ----------

def safe_roc_auc(y_true, prob):
    try:
        return float(roc_auc_score(y_true, prob))
    except Exception:
        return float("nan")

def get_estimator_and_featnames(loaded):
    if isinstance(loaded, dict):
        est = loaded.get("model") or loaded.get("estimator") or loaded.get("clf")
        featnames = (loaded.get("features") or loaded.get("feature_names")
                     or loaded.get("featnames"))
        return est, featnames
    return loaded, None

def predict_proba_pos(estimator, X):
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] == 2:
            return proba[:, 1]
    if hasattr(estimator, "decision_function"):
        s = estimator.decision_function(X)
        s = (s - s.min()) / (s.max() - s.min() + 1e-9)
        return s
    return estimator.predict(X).astype(float)

def choose_threshold_by_recall(y_true, prob_pos, target=1.0):
    """Escolhe o MENOR threshold que atinge recall >= target.
       Entre empatados, fica com maior precisão."""
    y_true = np.asarray(y_true, int)
    prob_pos = np.asarray(prob_pos, float)

    # candidatos: todos os valores únicos + 0.0
    thr_candidates = np.unique(np.r_[0.0, prob_pos])
    best = None  # (precision, -threshold, recall, threshold)
    found = False
    for t in thr_candidates:
        pred = (prob_pos >= t).astype(int)
        r = recall_score(y_true, pred, zero_division=0)
        if r + 1e-12 >= target:
            p = precision_score(y_true, pred, zero_division=0)
            # maior precisão, e em empate fica com menor threshold
            cand = (p, -t, r, t)
            if (not found) or cand > best:
                best = cand
                found = True
    if found:
        return float(best[3]), float(best[2])

    # fallback: não foi possível atingir o target -> escolhe o de MAIOR recall
    best = None  # (recall, precision, -threshold, threshold)
    for t in thr_candidates:
        pred = (prob_pos >= t).astype(int)
        r = recall_score(y_true, pred, zero_division=0)
        p = precision_score(y_true, pred, zero_division=0)
        cand = (r, p, -t, t)
        if (best is None) or cand > best:
            best = cand
    return float(best[3]), float(best[0])

def choose_threshold(y_true, prob_pos, mode="f1", recall_target=1.0):
    y_true = np.asarray(y_true, dtype=int)
    prob_pos = np.asarray(prob_pos, dtype=float)

    if mode == "youden":
        fpr, tpr, thr = roc_curve(y_true, prob_pos)
        j = tpr - fpr
        k = int(np.nanargmax(j))
        return float(np.clip(thr[k], 0.0, 1.0)), float(j[k])

    if mode == "recall":
        return choose_threshold_by_recall(y_true, prob_pos, target=recall_target)

    prec, rec, thr = precision_recall_curve(y_true, prob_pos)
    thr_ext = np.r_[thr, 1.0]

    if mode == "f1":
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        k = int(np.nanargmax(f1))
        return float(thr_ext[k]), float(f1[k])

    if mode == "precision":
        k = int(np.nanargmax(prec))
        top = np.where(prec == prec[k])[0]
        best = top[np.argmax(rec[top])]
        return float(thr_ext[best]), float(prec[best])

    return 0.5, float("nan")

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description="Prever com modelo sklearn/XGB + threshold opcional no VAL.")
    ap.add_argument("--features", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--save-meta", default=None)
    ap.add_argument("--eval-subset", default="all", choices=["all","train","val","test"])
    ap.add_argument("--opt-threshold", dest="opt_threshold", default="none",
                    choices=["none","f1","recall","precision","youden"])
    ap.add_argument("--recall-target", type=float, default=1.0,
                    help="Alvo de recall quando --opt-threshold recall (p.ex. 0.95).")
    args = ap.parse_args()

    feats_path = Path(args.features)
    df = pd.read_parquet(feats_path)

    # meta defensivo
    has = {c: (c in df.columns) for c in ["subset","label","stem","error"]}
    subset = df["subset"] if has["subset"] else pd.Series(["all"]*len(df), index=df.index)
    label  = df["label"]  if has["label"]  else pd.Series([-1]*len(df), index=df.index)
    stem   = df["stem"]   if has["stem"]   else pd.Series([f"row_{i}" for i in range(len(df))], index=df.index)

    # features numéricas
    X_all = df.drop(columns=[c for c in ["subset","label","stem","error"] if c in df.columns], errors="ignore")
    X_all = X_all.select_dtypes(include=[np.number])

    # carregar modelo e alinhar features
    loaded = joblib.load(args.model)
    model, featnames = get_estimator_and_featnames(loaded)
    if featnames is not None:
        for m in featnames:
            if m not in X_all.columns:
                X_all[m] = 0.0
        X_all = X_all[featnames]
    else:
        featnames = list(X_all.columns)

    # probabilidades para todas as linhas
    prob_pos_all = predict_proba_pos(model, X_all.values)

    subset = subset.astype(str).fillna("all")
    if args.eval_subset != "all":
        mask_eval = (subset == args.eval_subset)
    else:
        mask_eval = np.ones(len(df), dtype=bool)

    # threshold
    thr = 0.5
    thr_info = None
    if args.opt_threshold != "none":
        mask_val = (subset == "val")
        y_val = label.loc[mask_val]
        prob_val = prob_pos_all[mask_val]
        if mask_val.sum() >= 5 and y_val.nunique() == 2:
            thr, score_val = choose_threshold(
                y_val.values, prob_val, mode=args.opt_threshold,
                recall_target=args.recall_target
            )
            thr_info = {"mode": args.opt_threshold, "thr": float(thr),
                        "score_val": float(score_val), "n_val": int(mask_val.sum()),
                        "recall_target": args.recall_target if args.opt_threshold=="recall" else None}
            print(f"🔧 Threshold escolhido no VAL ({args.opt_threshold}"
                  f"{'@'+str(args.recall_target) if args.opt_threshold=='recall' else ''}): "
                  f"{thr:.3f} | score_val={score_val:.3f} | n_val={mask_val.sum()}")
        else:
            print("⚠️ Sem VAL suficiente (>=5 e 2 classes). Usando thr=0.5.")

    pred_all = (prob_pos_all >= thr).astype(int)

    # guardar CSV
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame({
        "subset": subset.values,
        "label":  label.values,
        "stem":   stem.values,
        "prob_pos": prob_pos_all,
        "pred":     pred_all,
    })
    out_df.to_csv(out, index=False)
    print(f"✅ Previsões salvas em: {out}  |  linhas={len(out_df)}")

    # métricas no subset
    eval_df = out_df.loc[mask_eval].copy()
    eval_df = eval_df[eval_df["label"].isin([0,1])]
    report_txt = None
    auc = float("nan")
    ap_score = float("nan")
    if len(eval_df) >= 1 and eval_df["label"].nunique() == 2:
        report_txt = classification_report(eval_df["label"], eval_df["pred"], digits=3)
        auc = safe_roc_auc(eval_df["label"], eval_df["prob_pos"])
        ap_score = float(average_precision_score(eval_df["label"], eval_df["prob_pos"]))
        print(f"\n📊 Métricas (subset={args.eval_subset}):\n{report_txt}\nROC AUC: {auc:.3f} | AP: {ap_score:.3f}")
    else:
        print("⚠️ Não há labels suficientes no subset escolhido para métricas binárias.")

    # meta JSON
    if args.save_meta:
        meta_out = {
            "features_path": str(feats_path),
            "model_path": str(args.model),
            "n_rows": int(len(df)),
            "eval_subset": args.eval_subset,
            "threshold": float(thr),
            "threshold_info": thr_info,
            "feature_names": featnames,
            "metrics": {
                "roc_auc": auc,
                "average_precision": ap_score,
                "classification_report": report_txt,
                "n_eval": int(len(eval_df)),
                "positives_eval": int(eval_df["label"].sum()) if len(eval_df) else 0,
            },
        }
        meta_path = Path(args.save_meta)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w") as f:
            json.dump(meta_out, f, indent=2)
        print(f"📝 Meta salva em: {meta_path}")

if __name__ == "__main__":
    main()
