#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import pathlib
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import brier_score_loss, roc_auc_score, average_precision_score
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

# ---------------------------
# utilitários
# ---------------------------

def select_feature_matrix(df: pd.DataFrame, feature_names=None):
    """Se o modelo guardar feature_names_in_, usamos; caso contrário, usa colunas numéricas."""
    drop_cols = {"label", "subset", "split", "stem"}
    if feature_names is not None:
        X = df[[c for c in feature_names if c in df.columns]].copy()
    else:
        X = df.select_dtypes(include=[np.number]).copy()
        X = X.drop(columns=[c for c in drop_cols if c in X.columns], errors="ignore")
    return X

def ece_score(y_true, p_pred, n_bins=10):
    """Expected Calibration Error (binning uniforme nos scores)."""
    y_true = np.asarray(y_true)
    p_pred = np.asarray(p_pred)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    inds = np.digitize(p_pred, bins) - 1
    ece = 0.0
    for b in range(n_bins):
        mask = inds == b
        if not np.any(mask):
            continue
        conf = p_pred[mask].mean()
        acc = y_true[mask].mean()
        w = mask.mean()
        ece += np.abs(acc - conf) * w
    return float(ece)

def reliability_points(y_true, p_pred, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centers, confs, accs, counts = [], [], [], []
    inds = np.digitize(p_pred, bins) - 1
    for b in range(n_bins):
        left, right = bins[b], bins[b+1]
        mask = inds == b
        if mask.sum() == 0:
            continue
        centers.append(0.5 * (left + right))
        confs.append(p_pred[mask].mean())
        accs.append(y_true[mask].mean())
        counts.append(int(mask.sum()))
    return np.array(centers), np.array(confs), np.array(accs), np.array(counts)

def smooth_curve(x, y, window=2):
    """Suavização simples (média móvel) para a linha da reliability curve."""
    if len(y) == 0:
        return x, y
    y_s = y.copy().astype(float)
    for i in range(len(y)):
        i0 = max(0, i - window)
        i1 = min(len(y), i + window + 1)
        y_s[i] = y[i0:i1].mean()
    return x, y_s

def plot_reliability(y_true, p_pred, title, out_png, n_bins=10):
    bs = brier_score_loss(y_true, p_pred)
    ece = ece_score(y_true, p_pred, n_bins=n_bins)

    # pontos por bin
    centers, confs, accs, counts = reliability_points(y_true, p_pred, n_bins=n_bins)
    _, accs_s = smooth_curve(centers, accs, window=1)

    fig, ax1 = plt.subplots(figsize=(6.5, 5))
    ax1.plot([0, 1], [0, 1], ls="--")
    ax1.scatter(confs, accs, s=30)
    ax1.plot(confs, accs_s, lw=2)

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel("Predicted probability")
    ax1.set_ylabel("Observed frequency")
    ax1.set_title(f"{title} | Brier={bs:.3f} | ECE={ece:.3f}")

    # histograma de scores
    ax2 = ax1.twinx()
    ax2.hist(p_pred, bins=np.linspace(0, 1, 21), alpha=0.15)
    ax2.set_ylabel("Observed frequency")

    pathlib.Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()
    return bs, ece

def fit_platt(p_val, y_val):
    lr = LogisticRegression(max_iter=1000)
    lr.fit(p_val.reshape(-1, 1), y_val)
    def mapper(p):
        return lr.predict_proba(p.reshape(-1, 1))[:, 1]
    return mapper, lr

def fit_isotonic(p_val, y_val):
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_val, y_val)
    def mapper(p):
        return iso.predict(p)
    return mapper, iso

# ---------------------------
# pipeline
# ---------------------------

def probas_from_model(model, X):
    # compatível com xgboost/sklearn
    try:
        proba = model.predict_proba(X)[:, 1]
    except Exception:
        # alguns estimadores expõem .predict com output de prob positivo
        pred = model.predict(X)
        if pred.ndim == 1:
            proba = pred
        else:
            proba = pred[:, 1]
    return np.asarray(proba, dtype=float)

def process_one_model(name, model_path, df_val, df_test, out_dir, n_bins=10):
    model = joblib.load(model_path)

    # selecionar features consistentes
    feat_names = getattr(model, "feature_names_in_", None)
    X_val = select_feature_matrix(df_val, feat_names)
    X_test = select_feature_matrix(df_test, feat_names)
    y_val = df_val["label"].values.astype(int)
    y_test = df_test["label"].values.astype(int)

    # probs "cruas"
    p_val_raw = probas_from_model(model, X_val)
    p_test_raw = probas_from_model(model, X_test)

    # calibradores (treinam em val)
    platt_map, _ = fit_platt(p_val_raw, y_val)
    iso_map, _ = fit_isotonic(p_val_raw, y_val)

    p_val_platt = platt_map(p_val_raw)
    p_val_iso   = iso_map(p_val_raw)
    p_test_platt = platt_map(p_test_raw)
    p_test_iso   = iso_map(p_test_raw)

    # métricas + plots
    rows = []
    for split, y, p_raw, p_platt, p_iso in [
        ("val", y_val, p_val_raw,  p_val_platt,  p_val_iso),
        ("test", y_test, p_test_raw, p_test_platt, p_test_iso),
    ]:
        # AUC e AP sempre sobre probabilidades (não calibradas influenciam apenas calibração)
        auc = roc_auc_score(y, p_raw)
        ap  = average_precision_score(y, p_raw)

        bs_raw,  ece_raw  = plot_reliability(y, p_raw,
                          f"Calibration {name} ({split}) — raw",
                          str(pathlib.Path(out_dir)/f"calibration_{name}_{split}_raw.png"),
                          n_bins=n_bins)
        bs_pl,   ece_pl   = plot_reliability(y, p_platt,
                          f"Calibration {name} ({split}) — Platt",
                          str(pathlib.Path(out_dir)/f"calibration_{name}_{split}_platt.png"),
                          n_bins=n_bins)
        bs_iso,  ece_iso  = plot_reliability(y, p_iso,
                          f"Calibration {name} ({split}) — Isotonic",
                          str(pathlib.Path(out_dir)/f"calibration_{name}_{split}_iso.png"),
                          n_bins=n_bins)

        rows += [
            dict(model=name, split=split, method="raw",     AUC=auc, AP=ap, Brier=bs_raw,  ECE=ece_raw),
            dict(model=name, split=split, method="platt",   AUC=auc, AP=ap, Brier=bs_pl,   ECE=ece_pl),
            dict(model=name, split=split, method="isotonic",AUC=auc, AP=ap, Brier=bs_iso,  ECE=ece_iso),
        ]
    return rows

def main():
    ap = argparse.ArgumentParser(description="Comparação de calibração (raw vs Platt vs Isotonic) para XGB/RF.")
    ap.add_argument("--features", required=True, help="features.parquet com colunas ['subset','label',...features]")
    ap.add_argument("--xgb-model", required=True, help="Caminho para xgboost.joblib")
    ap.add_argument("--rf-model",  required=True, help="Caminho para random_forest.joblib")
    ap.add_argument("--out-dir",   required=True, help="Pasta onde salvar gráficos e metrics.csv")
    ap.add_argument("--n-bins", type=int, default=10)
    ap.add_argument("--subset-val",  default="val")
    ap.add_argument("--subset-test", default="test")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.features)
    if "subset" not in df.columns:
        raise SystemExit("features.parquet precisa ter a coluna 'subset' (train/val/test).")

    df_val  = df[df["subset"] == args.subset_val].reset_index(drop=True)
    df_test = df[df["subset"] == args.subset_test].reset_index(drop=True)
    if len(df_val)==0 or len(df_test)==0:
        raise SystemExit(f"Nº amostras — val={len(df_val)}, test={len(df_test)}. Verifique 'subset'.")

    all_rows = []
    all_rows += process_one_model("xgboost", args.xgb_model, df_val, df_test, out_dir, n_bins=args.n_bins)
    all_rows += process_one_model("random_forest", args.rf_model, df_val, df_test, out_dir, n_bins=args.n_bins)

    dfm = pd.DataFrame(all_rows)
    dfm_path = out_dir / "calibration_metrics.csv"
    dfm.to_csv(dfm_path, index=False)

    # resumo no terminal
    print("\n=== Calibration metrics (lower Brier/ECE is better) ===")
    print(dfm.sort_values(["split", "model", "method"]).to_string(index=False))
    print(f"\n📄 Metrics saved to: {dfm_path}")
    print(f"🖼️  Plots saved under: {out_dir}")

if __name__ == "__main__":
    main()