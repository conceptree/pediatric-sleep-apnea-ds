#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plotReliabilityComparison.py
Comparação de curvas de confiabilidade (reliability / calibration) entre dois modelos.

Uso (exemplo):
python3 plotReliabilityComparison.py \
  --pred-a /path/xgb_val_predictions.csv \
  --pred-b /path/rf_val_predictions.csv \
  --label-a "XGBoost (iso)" \
  --label-b "RandomForest (iso)" \
  --subset val \
  --n-bins 10 \
  --smooth-win 3 \
  --out-png /path/reliability_xgb_vs_rf_val.png \
  --out-json /path/reliability_xgb_vs_rf_val.json

Requisitos de colunas nos CSVs: 'prob_pos', 'label' e (opcional) 'subset'.
Se 'subset' existir, será filtrado pelo valor passado em --subset.
"""

import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def reliability_curve(probs: pd.Series,
                      y_true: pd.Series,
                      n_bins: int = 10,
                      smooth_win: int = 3):
    """
    Devolve (bin_centers, observed_pos_rate_suavizada, counts) de forma robusta:
      - usa pandas.cut para criar bins em [0,1]
      - evita divisões por zero e descarta bins vazios
      - aplica suavização por rolling (min_periods=1) para não gerar NaN
    """
    df = pd.DataFrame({"p": probs.astype(float), "y": y_true.astype(int)}).dropna()
    # garantir domínio [0,1]
    df["p"] = df["p"].clip(0, 1)

    # criar bins e agregar
    df["bin"] = pd.cut(df["p"], bins=n_bins, include_lowest=True)
    g = df.groupby("bin", observed=True)

    centers = g["p"].mean().rename("center")
    obs_rate = g["y"].mean().rename("obs_rate")  # fração de positivos no bin
    counts = g.size().rename("count")

    out = pd.concat([centers, obs_rate, counts], axis=1)
    out = out.dropna(subset=["center", "obs_rate"])
    out = out[out["count"] > 0]  # remove bins vazios

    if out.empty:
        return np.array([]), np.array([]), np.array([])

    # suavização leve nas y (taxa observada)
    win = max(1, int(smooth_win))
    obs_sm = out["obs_rate"].rolling(window=win, min_periods=1, center=True).mean()

    return out["center"].to_numpy(), obs_sm.to_numpy(), out["count"].to_numpy()


def load_predictions(path: str, subset: str | None):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {p}")

    df = pd.read_csv(p)
    required = {"prob_pos", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{p} falta colunas: {missing}")

    # se houver coluna 'subset' e for pedido filtro, aplica
    if subset is not None and "subset" in df.columns:
        df = df[df["subset"] == subset].copy()

    # garantir tipos
    df["prob_pos"] = df["prob_pos"].astype(float)
    df["label"] = df["label"].astype(int)
    return df.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-a", required=True, help="CSV de previsões do Modelo A")
    ap.add_argument("--pred-b", required=True, help="CSV de previsões do Modelo B")
    ap.add_argument("--label-a", required=True, help="Legenda para o Modelo A")
    ap.add_argument("--label-b", required=True, help="Legenda para o Modelo B")
    ap.add_argument("--subset", required=True, help="Filtro do subset (ex.: val, test). É usado se existir coluna 'subset'.")
    ap.add_argument("--n-bins", type=int, default=10, help="Número de bins (default=10)")
    ap.add_argument("--smooth-win", type=int, default=3, help="Janela de suavização (default=3)")
    ap.add_argument("--out-png", required=True, help="Caminho de saída do PNG")
    ap.add_argument("--out-json", required=True, help="Caminho de saída do JSON")
    args = ap.parse_args()

    # carregar
    dfA = load_predictions(args.pred_a, args.subset)
    dfB = load_predictions(args.pred_b, args.subset)

    # curvas
    bins_a, probs_a, freq_a = reliability_curve(dfA["prob_pos"], dfA["label"], args.n_bins, args.smooth_win)
    bins_b, probs_b, freq_b = reliability_curve(dfB["prob_pos"], dfB["label"], args.n_bins, args.smooth_win)

    # plot
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 5))
    ax.plot([0, 1], [0, 1], ls="--", alpha=0.6, label="_perfect_")  # diagonal

    plotted_any = False
    if len(bins_a) > 0:
        ax.plot(bins_a, probs_a, marker="o", label=args.label_a)
        plotted_any = True
    if len(bins_b) > 0:
        ax.plot(bins_b, probs_b, marker="o", label=args.label_b)
        plotted_any = True

    # histograma da distribuição das probabilidades (modelo A)
    ax2 = ax.twinx()
    ax2.hist(dfA["prob_pos"].clip(0, 1), bins=args.n_bins, alpha=0.15)

    ax.set_title(f"Reliability — {args.subset}  (bins={args.n_bins}, smooth={args.smooth_win})")
    ax.set_xlabel("Predicted probability (positive)")
    ax.set_ylabel("Observed frequency (positive)")
    ax2.set_ylabel("Observed frequency (hist A)")
    if plotted_any:
        ax.legend(loc="upper left")

    Path(args.out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=160)
    plt.close(fig)

    # JSON serializável
    payload = {
        "subset": args.subset,
        "n_bins": int(args.n_bins),
        "smooth_win": int(args.smooth_win),
        "model_a": {
            "label": args.label_a,
            "bins": bins_a.tolist(),
            "probs": probs_a.tolist(),
            "freq": freq_a.tolist(),
            "n_rows": int(len(dfA))
        },
        "model_b": {
            "label": args.label_b,
            "bins": bins_b.tolist(),
            "probs": probs_b.tolist(),
            "freq": freq_b.tolist(),
            "n_rows": int(len(dfB))
        }
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"🖼️   saved: {args.out_png}")
    print(f"📄 json: {args.out_json}")


if __name__ == "__main__":
    main()