#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plota SpO2 ao longo do tempo para stems escolhidos (procura-os em train/val/test).
Correções:
- Se canal for OSAT em volts (0-5), converte p/ % (x20).
- Limites do eixo Y adaptativos se não está em %.
"""

import argparse
from pathlib import Path
import mne
import numpy as np
import matplotlib.pyplot as plt

CAND_SPO2 = ["spo2", "osat"]  # ordem = prioridade

def find_channel(raw, candidates):
    names = [ch.strip().lower() for ch in raw.ch_names]
    for wanted in candidates:
        for i, nm in enumerate(names):
            if nm == wanted:
                return i, raw.ch_names[i]
    return None, None

def resample_1hz(sig, sfreq):
    if sfreq <= 1.5:
        return sig
    decim = int(round(sfreq))
    n = len(sig) // decim
    if n == 0:
        return sig
    return sig[:n*decim].reshape(n, decim).mean(axis=1)

def build_index(base_splits: Path):
    idx = {}
    for subset in ["train", "val", "test"]:
        for label in ["positives", "negatives"]:
            for edf in (base_splits / subset / label).glob("*.edf"):
                idx[edf.stem] = edf
    return idx

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-splits", required=True, help="Pasta base dos splits (train/val/test)")
    ap.add_argument("--subset", choices=["train","val","test"], default=None,
                    help="Se indicado, só procura neste subset")
    ap.add_argument("--stems", nargs="*", default=None, help="Lista de stems a plotar")
    ap.add_argument("--max-n", type=int, default=0, help="Limite máximo de plots (0 = sem limite)")
    ap.add_argument("--out-dir", required=True, help="Pasta de saída para os PNGs")
    args = ap.parse_args()

    base = Path(args.base_splits)
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # indexa todos os .edf pelos stems
    if args.subset:
        index = {}
        for label in ["positives", "negatives"]:
            for edf in (base / args.subset / label).glob("*.edf"):
                index[edf.stem] = edf
    else:
        index = build_index(base)

    # seleciona stems
    stems = args.stems or sorted(index.keys())
    if args.max_n and len(stems) > args.max_n:
        stems = stems[:args.max_n]

    print(f"Vou plotar {len(stems)} estudo(s).")

    for stem in stems:
        edf = index.get(stem)
        if not edf:
            print(f"⚠️  Stem {stem} não encontrado.")
            continue
        try:
            raw = mne.io.read_raw_edf(str(edf), preload=False, verbose=False)
            idx, name = find_channel(raw, CAND_SPO2)
            if idx is None:
                print(f"⚠️  Sem canal SpO2 para {stem}")
                continue

            x, _ = raw[idx, :]
            y = np.asarray(x).ravel().astype(float)
            sf = float(raw.info["sfreq"])

            # auto-escala OSAT volts -> %
            if name.strip().lower() == "osat" and np.nanmax(y) <= 10.0:
                y = y * 20.0

            # limpeza leve + 1 Hz p/ suavizar
            y = np.clip(y, 50, 100)
            y1 = resample_1hz(y, sf)
            t_min = np.arange(y1.size) / 60.0  # 1 Hz -> segundos -> minutos

            plt.figure(figsize=(14,4))
            plt.plot(t_min, y1, linewidth=1)

            # eixos
            ymax = float(np.nanmax(y1)) if np.isfinite(np.nanmax(y1)) else 100.0
            ymin = float(np.nanmin(y1)) if np.isfinite(np.nanmin(y1)) else 50.0
            if ymax > 20:   # assume já em %
                plt.ylim(70, 100)
                plt.ylabel("SpO₂ (%)")
            else:
                plt.ylim(ymin-1, max(5, ymax)+1)
                plt.ylabel("SpO₂ (un.)")
            plt.xlabel("Tempo (min)")
            plt.title(f"SpO₂ — {stem}  |  canal: {name}")

            out = outdir / f"{stem}_spo2.png"
            plt.tight_layout()
            plt.savefig(out, dpi=130)
            plt.close()
            print(f"🖼️  salvo: {out}")
        except Exception as e:
            print(f"❌  Erro em {stem}: {e}")

if __name__ == "__main__":
    main()
