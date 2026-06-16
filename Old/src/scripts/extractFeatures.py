#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extração de features (baseline) para NCH-Sleep

- Lê CSVs de splits (train/val/test) com colunas: subset, label, stem
- Procura EDF/TSV em --raw-base como {stem}.edf / {stem}.tsv (layout plano)
- Extrai features de SpO2 (estatísticas e desaturações)
- Robusto: --resume, --skip-errors, checkpoint periódico, progress bar

Uso típico:
  python3 extractFeatures.py \
    --splits-dir /.../splits_tsv_thr3/csv \
    --raw-base   /.../nch-sleep/all_raw/Sleep_Data \
    --out-parquet /.../splits_tsv_thr3/features.parquet \
    --resume --skip-errors
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
import mne


# ----------------------------
# Utils
# ----------------------------

SPO2_CANDIDATES = {
    "spo2", "osat", "oximetry", "oxy", "oximetria", "spo2/oximetry",
    "o2", "sao2", "sp02", "sat", "tcco2"  # mantemos nomes próximos; tcco2 será ignorado
}

def normalize_name(ch: str) -> str:
    return " ".join(ch.strip().lower().replace("_", " ").replace("-", " ").split())


def pick_spo2_channel(raw: mne.io.BaseRaw):
    """Escolhe o canal de SpO2 a partir de candidatos conhecidos."""
    choices = []
    for idx, ch in enumerate(raw.ch_names):
        norm = normalize_name(ch)
        if any(tok in norm for tok in SPO2_CANDIDATES) and "tcco2" not in norm and "etco2" not in norm:
            choices.append((idx, ch))
    # heurística simples: preferir nomes que contém "spo2" ou "osat"
    def score(name):
        n = normalize_name(name)
        s = 0
        if "spo2" in n: s += 3
        if "osat" in n: s += 2
        if "sat" in n:  s += 1
        return -s
    choices.sort(key=lambda x: score(x[1]))
    return choices[0] if choices else (None, None)


def to_spo2_percent(arr: np.ndarray) -> np.ndarray:
    """Converte série para %:
       - Se max<=1.2 -> assume 0..1  => *100
       - Se max<=6   -> assume 0..5V => *20
       - Caso contrário assume já em %
       Clipa para 50..100 (range típico noturno) para robustez.
    """
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return arr

    mx = np.nanmax(finite)
    if mx <= 1.2:
        out = arr * 100.0
    elif mx <= 6.0:
        out = arr * 20.0
    else:
        out = arr

    out = np.clip(out, 50.0, 100.0)
    return out


def desat_events(spo2_pct: np.ndarray, sfreq: float, drop=3.0, min_dur_s=10.0):
    """Conta eventos de desaturação >= drop% com duração mínima."""
    if spo2_pct.size == 0 or not np.isfinite(spo2_pct).any():
        return 0

    # baseline rolante simples (mediana de 60s) para detectar quedas relativas
    win = int(max(1, sfreq * 60))
    pad = np.pad(spo2_pct, (win, win), mode="edge")
    # mediana por janelas (versão rápida com stride simples)
    med = pd.Series(pad).rolling(win, center=True, min_periods=1).median().to_numpy()[win:-win]

    delta = med - spo2_pct
    below = delta >= drop
    # agrupar sequências contínuas
    events = 0
    run = 0
    for b in below:
        if b:
            run += 1
        else:
            if run / sfreq >= min_dur_s:
                events += 1
            run = 0
    if run / sfreq >= min_dur_s:
        events += 1
    return events


def extract_spo2_features(edf_path: Path) -> dict:
    """Extrai features de SpO2 de um EDF (sem preload, rápido)."""
    raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose=False)

    idx, chname = pick_spo2_channel(raw)
    feats = {"has_spo2": 0}

    if idx is None:
        return feats  # sem spo2

    x, _ = raw[idx, :]
    arr = np.asarray(x).ravel().astype(float)
    arr = to_spo2_percent(arr)
    sfreq = float(raw.info["sfreq"])
    t_hours = arr.size / sfreq / 3600.0

    # estatísticas básicas
    p = np.nanpercentile(arr, [5, 25, 50, 75, 95])
    feats.update({
        "has_spo2": 1,
        "spo2_mean": float(np.nanmean(arr)),
        "spo2_std": float(np.nanstd(arr)),
        "spo2_p5": float(p[0]),
        "spo2_p25": float(p[1]),
        "spo2_p50": float(p[2]),
        "spo2_p75": float(p[3]),
        "spo2_p95": float(p[4]),
        "spo2_pct_below_90": float(np.mean(arr < 90.0) * 100.0),
        "spo2_pct_below_92": float(np.mean(arr < 92.0) * 100.0),
    })

    # eventos de desaturação por hora
    try:
        ev = desat_events(arr, sfreq, drop=3.0, min_dur_s=10.0)
        feats["spo2_desat_events_3pct_per_h"] = float(ev / max(1e-6, t_hours))
    except Exception:
        feats["spo2_desat_events_3pct_per_h"] = np.nan

    return feats


def load_splits_csvs(splits_dir: Path) -> pd.DataFrame:
    csv_dir = Path(splits_dir)
    parts = []
    for name in ("train.csv", "val.csv", "test.csv"):
        p = csv_dir / name
        if p.exists():
            df = pd.read_csv(p)
            # garantir coluna subset coerente
            if "subset" not in df.columns:
                df["subset"] = name.replace(".csv", "")
            parts.append(df)
    if not parts:
        raise FileNotFoundError(f"Nenhum CSV encontrado em {csv_dir}")
    df = pd.concat(parts, ignore_index=True)
    # normalizar colunas mínimas
    for col in ("stem", "label", "subset"):
        if col not in df.columns:
            raise ValueError(f"Coluna obrigatória ausente nos CSVs: '{col}'")
    return df[["subset", "label", "stem"]].copy()


def merge_resume(existing_path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(existing_path)
    except Exception:
        return pd.DataFrame()


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits-dir", required=True, help="Dir com train.csv / val.csv / test.csv")
    ap.add_argument("--raw-base", required=True, help="Pasta com EDF/TSV (layout plano: {stem}.edf)")
    ap.add_argument("--out-parquet", required=True, help="Ficheiro .parquet de saída")
    ap.add_argument("--checkpoint-every", type=int, default=50, help="Gravar checkpoint a cada N estudos")
    ap.add_argument("--resume", action="store_true", help="Continuar (não recalcular stems já presentes)")
    ap.add_argument("--skip-errors", action="store_true", help="Ignorar estudos com erro")
    ap.add_argument("--output-log", default="", help="Guardar log detalhado (opcional)")

    args = ap.parse_args()
    splits_dir = Path(args.splits_dir)
    raw_base   = Path(args.raw_base)
    out_path   = Path(args.out_parquet)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_path  = out_path.with_suffix(".partial.parquet")

    log_lines = []

    # carregar plano de trabalho
    df_plan = load_splits_csvs(splits_dir)

    # aplicar resume
    done_df = pd.DataFrame()
    done_stems = set()
    if args.resume and out_path.exists():
        done_df = merge_resume(out_path)
        if not done_df.empty and "stem" in done_df.columns:
            done_stems = set(done_df["stem"].astype(str).tolist())

    todo = df_plan[~df_plan["stem"].astype(str).isin(done_stems)].copy()
    if todo.empty:
        print(f"✅ Nada a fazer: {len(done_stems)} stems já extraídos em {out_path}")
        return

    rows = []
    counter = 0
    print(f"🔧 A extrair features de {len(todo)} estudos (resume={args.resume}) …")

    for _, r in tqdm(todo.iterrows(), total=len(todo), unit="study"):
        stem = str(r["stem"])
        edf = raw_base / f"{stem}.edf"

        try:
            if not edf.exists():
                raise FileNotFoundError(f"EDF não encontrado: {edf}")

            feats = extract_spo2_features(edf)
            feats.update({
                "subset": r["subset"],
                "label": int(r["label"]),
                "stem": stem,
            })
            rows.append(feats)
            counter += 1

        except Exception as e:
            msg = f"❌ Erro em {stem}: {e}"
            print(msg)
            log_lines.append(msg)
            if not args.skip_errors:
                raise
            # salva linha mínima para manter rastreio
            rows.append({
                "subset": r["subset"], "label": int(r["label"]),
                "stem": stem, "has_spo2": 0
            })
            counter += 1

        # checkpoint periódico
        if counter % args.checkpoint_every == 0:
            part = pd.DataFrame(rows)
            if not done_df.empty:
                part = pd.concat([done_df, part], ignore_index=True)
            part.to_parquet(ckpt_path, index=False)
            print(f"💾 checkpoint: {ckpt_path}  | rows={len(part)}")

    # consolidar com o que já existia
    df_out = pd.DataFrame(rows)
    if not done_df.empty:
        df_out = pd.concat([done_df, df_out], ignore_index=True)
        # remover duplicados por 'stem' mantendo a última
        df_out = df_out.drop_duplicates(subset=["stem"], keep="last")

    df_out.to_parquet(out_path, index=False)
    print(f"✅ Features salvas em: {out_path}  | shape={df_out.shape}")

    # estatísticas rápidas
    with pd.option_context("display.max_columns", 30):
        num = df_out.select_dtypes(include=[np.number])
        if not num.empty:
            print("\n📊 Estatísticas das features (amostra):")
            print(
                num.describe().T[["mean", "std", "min", "50%", "max"]]
                .round(2).head(20)
            )

    # log opcional
    if args.output_log:
        Path(args.output_log).write_text("\n".join(log_lines))
        print(f"📝 Log salvo em: {args.output_log}")


if __name__ == "__main__":
    main()
