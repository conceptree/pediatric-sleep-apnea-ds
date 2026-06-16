#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SHAP Feature Importance (XGBoost / RandomForest) com probabilidade (interventional) e fallback para raw.

Exemplo:
python3 shapImportance.py \
  --features /.../features.parquet \
  --model /.../xgboost.joblib \
  --subset test \
  --out-dir /.../reports/shap/xgb \
  --topk 20 --max-samples 1500 --save-matrix
"""

import argparse
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------
# Helpers
# ---------------------------
def get_pos_class_index(model):
    try:
        classes = getattr(model, "classes_", None)
        if classes is None:
            return 1
        classes = list(classes)
        if 1 in classes:
            return classes.index(1)
        return len(classes) - 1
    except Exception:
        return 1


def coerce_shap_to_array(shap_values, pos_idx=1):
    """Garante (n_samples, n_features). Seleciona classe positiva quando necessário."""
    if isinstance(shap_values, list):
        return np.asarray(shap_values[pos_idx])
    sv = np.asarray(shap_values)
    if sv.ndim == 3 and sv.shape[-1] >= pos_idx + 1:
        return sv[:, :, pos_idx]
    if sv.ndim == 2:
        return sv
    raise ValueError(f"SHAP values devem ser 2-D (amostras x features). Recebido shape={sv.shape}")


def align_columns_to_model(df: pd.DataFrame, model):
    # remove metas
    drop = [c for c in ["subset", "split", "stem", "label"] if c in df.columns]
    X = df.drop(columns=drop, errors="ignore")
    # numéricas
    X = X.apply(pd.to_numeric, errors="coerce")
    if hasattr(model, "feature_names_in_"):
        keep = [c for c in model.feature_names_in_ if c in X.columns]
        X = X[keep]
        return X, keep
    keep = X.select_dtypes(include=[np.number]).columns.tolist()
    X = X[keep]
    return X, keep


def pick_background(df_all: pd.DataFrame, model, max_bg=200):
    """Escolhe background do subset train se existir; senão amostra global."""
    if "subset" in df_all.columns:
        bg_df = df_all[df_all["subset"] == "train"].copy()
        if bg_df.empty:
            bg_df = df_all.copy()
    elif "split" in df_all.columns:
        bg_df = df_all[df_all["split"] == "train"].copy()
        if bg_df.empty:
            bg_df = df_all.copy()
    else:
        bg_df = df_all.copy()

    X_bg, used_bg = align_columns_to_model(bg_df, model)
    X_bg = X_bg.fillna(X_bg.median(numeric_only=True))
    if len(X_bg) > max_bg:
        X_bg = X_bg.sample(max_bg, random_state=42)
    return X_bg


def save_beeswarm(shap_vals, X, out_png, max_display=20):
    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_vals, X, plot_type="dot", show=False, max_display=max_display)
    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()


def save_barplot(shap_vals, X, out_png, max_display=20):
    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_vals, X, plot_type="bar", show=False, max_display=max_display)
    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()


# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="Parquet com features")
    ap.add_argument("--model", required=True, help="Modelo .joblib (XGB / RF)")
    ap.add_argument("--subset", default="test", choices=["train", "val", "test", "all"], help="Subset a usar")
    ap.add_argument("--out-dir", required=True, help="Diretório de saída (CSV/plots)")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--max-samples", type=int, default=1500)
    ap.add_argument("--save-matrix", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Carregar features (inteiras para poder escolher background do train)
    print(f"📦 Carregando features: {args.features} | subset={args.subset}")
    df_all = pd.read_parquet(args.features)
    if "label" not in df_all.columns:
        raise ValueError("O parquet precisa conter a coluna 'label'.")

    # Filtra subset para explicação
    if args.subset != "all":
        subset_col = "subset" if "subset" in df_all.columns else ("split" if "split" in df_all.columns else None)
        if subset_col is None:
            raise ValueError("Não encontrei coluna 'subset' nem 'split' no features.parquet.")
        df = df_all[df_all[subset_col] == args.subset].copy()
    else:
        df = df_all.copy()

    y = df["label"].astype(int).values

    print(f"🧠 Carregando modelo: {args.model}")
    model = joblib.load(args.model)

    # Alinhar X ao modelo
    X, used_cols = align_columns_to_model(df, model)
    X = X.fillna(X.median(numeric_only=True))

    # Amostrar para acelerar SHAP
    if args.max_samples and len(X) > args.max_samples:
        X = X.sample(args.max_samples, random_state=42)
        y = y[X.index.values]

    # Background para modo interventional
    bg = pick_background(df_all, model, max_bg=200)

    # Tentar: probabilidade + interventional (recomendado)
    use_raw_fallback = False
    try:
        explainer = shap.TreeExplainer(
            model,
            data=bg,
            feature_perturbation="interventional",
            model_output="probability",
        )
        print("🔎 Calculando SHAP (probability, interventional)...")
        # Em versões antigas do shap, o check_additivity é argumento do shap_values():
        shap_values_raw = explainer.shap_values(X, check_additivity=False)
    except Exception as e:
        warnings.warn(f"Interventional probability falhou ({e}); a usar fallback RAW (log-odds).")
        use_raw_fallback = True

    if use_raw_fallback:
        explainer = shap.TreeExplainer(
            model,
            feature_perturbation="tree_path_dependent",
            model_output="raw"
        )
        print("🔎 Calculando SHAP (RAW / log-odds, tree_path_dependent)...")
        shap_values_raw = explainer.shap_values(X)

    # Normalizar para (n_samples, n_features), classe positiva
    pos_idx = get_pos_class_index(model)
    shap_values = coerce_shap_to_array(shap_values_raw, pos_idx=pos_idx)

    # Importância global
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    imp = pd.DataFrame({"feature": used_cols, "mean_abs_shap": mean_abs})
    imp = imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    out_csv = out_dir / f"{Path(args.model).stem}_shap_importance_{args.subset}.csv"
    imp.to_csv(out_csv, index=False)
    print(f"💾 Importância global salva: {out_csv} (top{min(5, args.topk)})")
    print(imp.head(min(5, args.topk)).to_string(index=False))

    # Plots
    beeswarm_png = out_dir / f"{Path(args.model).stem}_shap_beeswarm_{args.subset}.png"
    bar_png      = out_dir / f"{Path(args.model).stem}_shap_bar_{args.subset}.png"
    try:
        save_beeswarm(shap_values, X, beeswarm_png, max_display=args.topk)
        print(f"🖼️  Beeswarm salvo: {beeswarm_png}")
    except Exception as e:
        print(f"⚠️  Falha a gerar beeswarm: {e}")

    try:
        save_barplot(shap_values, X, bar_png, max_display=args.topk)
        print(f"🖼️  Bar plot salvo: {bar_png}")
    except Exception as e:
        print(f"⚠️  Falha a gerar bar plot: {e}")

    # Matriz
    if args.save_matrix:
        out_npz = out_dir / f"{Path(args.model).stem}_shap_values_{args.subset}.npz"
        np.savez_compressed(out_npz, shap_values=shap_values, features=np.array(used_cols))
        print(f"💾 SHAP matrix salva: {out_npz} (shape={shap_values.shape})")

    if use_raw_fallback:
        print("ℹ️  Nota: valores SHAP em unidades de log-odds (raw). Para comparar magnitudes entre modelos, prefira o modo 'probability' quando possível.")

    print("✅ Concluído.")


if __name__ == "__main__":
    main()