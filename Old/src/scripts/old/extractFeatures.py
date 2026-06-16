#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extrai features simples de SpO2 (e algumas de respiração, se existir)
para todos os estudos em train/val/test (pastas de symlinks).
Salva um Parquet com uma linha por estudo.

Correções:
- Detecta canal "OSAT" em volts (0-5 V) e converte para % (x20).
- Dá prioridade a "spo2" sobre "osat" quando ambos existem.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import mne
import json

# ========= CONFIG =========
SPLITS_BASE = Path("/Users/nunorodrigues/dev/tese/datasets/splits")
OUT_PARQUET = SPLITS_BASE / "features.parquet"

# nomes candidatos para canais
CAND_SPO2 = ["spo2", "osat"]           # prioridade na ordem
CAND_RESP = ["resp airflow", "resp ptaf", "resp flow"]

# ========= UTILS =========
def find_channel(raw, candidates):
    """Devolve (index, nome) do 1.º canal cujo nome (lower) bate certo."""
    names = [ch.strip().lower() for ch in raw.ch_names]
    for wanted in candidates:
        for i, nm in enumerate(names):
            if nm == wanted:
                return i, raw.ch_names[i]
    return None, None

def resample_1hz(sig, sfreq):
    """Reamostra para ~1 Hz por média de blocos."""
    if sfreq <= 1.5:
        return sig
    decim = int(round(sfreq))
    n = len(sig) // decim
    if n == 0:
        return sig
    sig = sig[:n * decim].reshape(n, decim).mean(axis=1)
    return sig

def pct_below(x, thr):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    return 100.0 * np.mean(x < thr)

def quantiles(x):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return [np.nan]*5
    return np.percentile(x, [5,25,50,75,95]).tolist()

def desat_events_3pct_per_h(spo2_1hz, sfreq_1hz=1.0):
    """Contagem muito simples de quedas >=3pp entre amostras consecutivas; normaliza por hora."""
    x = np.asarray(spo2_1hz, float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return np.nan
    drops = np.sum(np.diff(x) <= -3.0)
    horas = x.size / (sfreq_1hz * 3600.0)
    if horas <= 0:
        return np.nan
    return drops / horas

def load_tsv_events(tsv_path):
    """Conta linhas (eventos) num TSV anotado, se existir."""
    try:
        df = pd.read_csv(tsv_path, sep="\t")
        return int(len(df))
    except Exception:
        return 0

# ========= MAIN =========
rows = []

for subset in ["train", "val", "test"]:
    for label in ["positives", "negatives"]:
        d = SPLITS_BASE / subset / label
        if not d.exists():
            continue
        for edf in sorted(d.glob("*.edf")):
            stem = edf.stem
            tsv = edf.with_suffix(".tsv")
            row = dict(subset=subset, label=1 if label == "positives" else 0, stem=stem)

            # defaults
            row.update({
                "has_spo2": 0,
                "spo2_mean": np.nan,
                "spo2_std": np.nan,
                "spo2_p5": np.nan,
                "spo2_p25": np.nan,
                "spo2_p50": np.nan,
                "spo2_p75": np.nan,
                "spo2_p95": np.nan,
                "spo2_pct_below_90": np.nan,
                "spo2_pct_below_92": np.nan,
                "spo2_desat_events_3pct_per_h": np.nan,

                "has_resp": 0,
                "resp_mean": np.nan,
                "resp_std": np.nan,
                "resp_p5": np.nan,
                "resp_p25": np.nan,
                "resp_p50": np.nan,
                "resp_p75": np.nan,
                "resp_p95": np.nan,

                "tsv_apnea_events": 0,
                "tsv_events_per_h": np.nan,
            })

            try:
                raw = mne.io.read_raw_edf(str(edf), preload=False, verbose=False)

                # ---- SPO2 ----
                idx_spo2, name_spo2 = find_channel(raw, CAND_SPO2)
                if idx_spo2 is not None:
                    x, _ = raw[idx_spo2, :]
                    sig = np.asarray(x).ravel().astype(float)
                    sf = float(raw.info["sfreq"])

                    # auto-escala se OSAT em volts (máx <= 10)
                    if name_spo2.strip().lower() == "osat":
                        vmax = np.nanmax(sig)
                        if np.isfinite(vmax) and vmax <= 10.0:
                            sig = sig * 20.0  # 0-5 V -> 0-100 %

                    # limpeza básica
                    sig = np.clip(sig, 50, 100)
                    s1 = resample_1hz(sig, sf)

                    row["has_spo2"] = 1
                    row["spo2_mean"] = float(np.nanmean(s1))
                    row["spo2_std"] = float(np.nanstd(s1))
                    q5, q25, q50, q75, q95 = quantiles(s1)
                    row["spo2_p5"] = q5
                    row["spo2_p25"] = q25
                    row["spo2_p50"] = q50
                    row["spo2_p75"] = q75
                    row["spo2_p95"] = q95
                    row["spo2_pct_below_90"] = pct_below(s1, 90)
                    row["spo2_pct_below_92"] = pct_below(s1, 92)
                    row["spo2_desat_events_3pct_per_h"] = desat_events_3pct_per_h(s1, 1.0)

                # ---- RESP (opcional) ----
                idx_resp, name_resp = find_channel(raw, CAND_RESP)
                if idx_resp is not None:
                    x, _ = raw[idx_resp, :]
                    sig = np.asarray(x).ravel().astype(float)
                    sf = float(raw.info["sfreq"])
                    r1 = resample_1hz(sig, sf)

                    row["has_resp"] = 1
                    row["resp_mean"] = float(np.nanmean(r1))
                    row["resp_std"] = float(np.nanstd(r1))
                    q5, q25, q50, q75, q95 = quantiles(r1)
                    row["resp_p5"] = q5
                    row["resp_p25"] = q25
                    row["resp_p50"] = q50
                    row["resp_p75"] = q75
                    row["resp_p95"] = q95

                # ---- TSV ----
                n_ev = load_tsv_events(tsv)
                row["tsv_apnea_events"] = n_ev
                # duração aproximada (em horas) = (n amostras / sfreq) / 3600
                # se não tivermos s1, tentamos inferir duração total do registo
                try:
                    dur_sec = raw.n_times / float(raw.info["sfreq"])
                    row["tsv_events_per_h"] = n_ev / max(1e-6, (dur_sec / 3600.0))
                except Exception:
                    row["tsv_events_per_h"] = np.nan

            except Exception as exc:
                row["error"] = str(exc)

            rows.append(row)

# salva
df = pd.DataFrame(rows)

# garante ordem das colunas
cols = [
    "subset", "label", "stem",
    "has_spo2", "spo2_mean", "spo2_std", "spo2_p5", "spo2_p25", "spo2_p50", "spo2_p75", "spo2_p95",
    "spo2_pct_below_90", "spo2_pct_below_92", "spo2_desat_events_3pct_per_h",
    "has_resp", "resp_mean", "resp_std", "resp_p5", "resp_p25", "resp_p50", "resp_p75", "resp_p95",
    "tsv_apnea_events", "tsv_events_per_h", "error"
]
df = df.reindex(columns=[c for c in cols if c in df.columns])

# salvar parquet
OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(OUT_PARQUET, index=False)

print(f"✅ features em: {OUT_PARQUET}  | shape={df.shape}")

# estatísticas das colunas numéricas (sem subset/label/stem)
num = df.drop(columns=["subset","label","stem"], errors="ignore").select_dtypes(include="number")

print("\n📊 Estatísticas das features (train+val+test):")
print(num.describe().T[["mean","std","min","50%","max"]].round(2))

