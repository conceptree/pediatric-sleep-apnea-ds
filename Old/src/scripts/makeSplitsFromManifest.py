#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
makeSplitsFromManifest.py  (versão corrigida)

Funções:
- Lê um manifest "limpo" (ex.: gerado por verify/makeCleanManifest).
- Pode atribuir labels de duas formas:
  (A) labels-mode=folders  -> positivos/negativos vindos de pastas dadas
  (B) labels-mode=tsv      -> calcula AHI a partir de eventos nos .tsv
                              (conta apneas: obstructive|central|mixed e hypopneas)
- Faz split estratificado 80/10/10 (train/val/test).
- Cria estrutura de symlinks de EDF/TSV em out-dir/{train,val,test}/{positives,negatives}.
- (Opcional) Exporta tabela de AHI por estudo (--out-ahi-csv).

Requisitos no manifest:
- Colunas necessárias: ['stem', 'duration_s']
- Opcional: se houver 'duration_h', será ignorada; usamos duration_s.

Uso (modo TSV/Events → AHI):
  python3 makeSplitsFromManifest.py \
    --manifest /.../clean_manifest.csv \
    --labels-mode tsv \
    --tsv-root /.../all_raw \
    --ahi-threshold 3.0 \
    --out-ahi-csv /.../ahi_labels_thr3.csv \
    --out-dir /.../splits_tsv_thr3

Uso (modo pastas):
  python3 makeSplitsFromManifest.py \
    --manifest /.../clean_manifest.csv \
    --labels-mode folders \
    --labels-from-folders /.../positives /.../negatives \
    --out-dir /.../splits_from_folders
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path
import re
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------- Utilidades ----------

def read_manifest(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"stem", "duration_s"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {missing}")
    # Normaliza tipos
    df["stem"] = df["stem"].astype(str)
    df["duration_s"] = pd.to_numeric(df["duration_s"], errors="coerce")
    df = df.dropna(subset=["duration_s"])
    return df

def robust_read_tsv(tsv_path: Path) -> pd.DataFrame:
    # Os TSV do NCH-Sleep são tab-separated; usa um parser robusto
    return pd.read_csv(tsv_path, sep="\t", engine="python")

APNEA_PAT = re.compile(r"\b(apnea)\b", re.IGNORECASE)
HYPO_PAT   = re.compile(r"\b(hypopnea)\b", re.IGNORECASE)

def count_events_in_tsv(tsv_path: Path) -> tuple[int, int]:
    """
    Conta eventos de Apnea (obstructive|central|mixed etc.) e Hypopnea.
    Regras:
      - Hypopnea: qualquer linha cujo description contenha "hypopnea"
      - Apnea   : qualquer linha cujo description contenha "apnea"
                  EXCETO as que já casaram como hypopnea (para não duplicar)
    """
    df = robust_read_tsv(tsv_path)
    if "description" not in df.columns:
        return 0, 0
    desc = df["description"].astype(str).str.lower()

    is_hypo = desc.str.contains(HYPO_PAT)
    is_apn  = desc.str.contains(APNEA_PAT) & (~is_hypo)  # exclui hypopnea

    n_hypo = int(is_hypo.sum())
    n_apn  = int(is_apn.sum())
    return n_apn, n_hypo

def compute_ahi(n_apnea: int, n_hypopnea: int, duration_s: float,
                include_hypopnea: bool = True) -> float:
    hours = max(duration_s, 1e-9) / 3600.0
    denom = hours if hours > 0 else 1.0
    total_events = n_apnea + (n_hypopnea if include_hypopnea else 0)
    return total_events / denom

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def make_symlink(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    try:
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(src, dst)
    except FileExistsError:
        pass

def pick_paths(tsv_root: Path, stem: str) -> tuple[Path, Path]:
    # Estrutura: .../Sleep_Data/<stem>.ext
    base = tsv_root / "Sleep_Data"
    return base / f"{stem}.edf", base / f"{stem}.tsv"

# ---------- Labeling ----------

def labels_from_folders(manifest: pd.DataFrame,
                        pos_dir: Path,
                        neg_dir: Path) -> pd.DataFrame:
    pos_stems = {p.stem for p in Path(pos_dir).glob("*.edf")}
    neg_stems = {p.stem for p in Path(neg_dir).glob("*.edf")}
    both = pos_stems & neg_stems
    if both:
        print(f"⚠️  Stems em ambas as pastas (serão marcados como positivos): {sorted(list(both))[:5]} ... ({len(both)} no total)")

    def label_stem(s: str) -> int | None:
        if s in pos_stems: return 1
        if s in neg_stems: return 0
        return None

    manifest["label"] = manifest["stem"].map(label_stem)
    labeled = manifest.dropna(subset=["label"]).copy()
    labeled["label"] = labeled["label"].astype(int)
    return labeled

def labels_from_tsv(manifest: pd.DataFrame,
                    tsv_root: Path,
                    ahi_threshold: float,
                    include_hypopnea: bool = True,
                    out_ahi_csv: Path | None = None) -> pd.DataFrame:
    rows = []
    for _, r in manifest.iterrows():
        stem = r["stem"]
        duration_s = float(r["duration_s"])
        edf, tsv = pick_paths(tsv_root, stem)
        if not tsv.exists():
            # sem tsv -> sem label
            continue
        try:
            n_apn, n_hyp = count_events_in_tsv(tsv)
            ahi = compute_ahi(n_apn, n_hyp, duration_s, include_hypopnea=include_hypopnea)
            label = int(ahi >= ahi_threshold)
            rows.append({
                "stem": stem,
                "duration_s": duration_s,
                "n_apnea": n_apn,
                "n_hypopnea": n_hyp,
                "ahi": ahi,
                "label": label
            })
        except Exception as e:
            print(f"⚠️  Falha a processar TSV de {stem}: {e}")

    df = pd.DataFrame(rows)
    if out_ahi_csv is not None:
        ensure_dir(Path(out_ahi_csv).parent)
        df.to_csv(out_ahi_csv, index=False)
        print(f"📝 AHI table saved: {out_ahi_csv}")
    return df

# ---------- Splits + Symlinks ----------

def stratified_splits(df_lbl: pd.DataFrame,
                      test_size: float = 0.10,
                      val_size: float = 0.10,
                      seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Primeiro retira test do conjunto total
    df_tmp, df_test = train_test_split(
        df_lbl, test_size=test_size, stratify=df_lbl["label"], random_state=seed
    )
    # Depois separa val do remanescente
    val_ratio = val_size / (1.0 - test_size)
    df_train, df_val = train_test_split(
        df_tmp, test_size=val_ratio, stratify=df_tmp["label"], random_state=seed
    )
    return df_train.copy(), df_val.copy(), df_test.copy()

def materialize_symlinks(df_split: pd.DataFrame,
                         subset_name: str,
                         out_dir: Path,
                         tsv_root: Path) -> None:
    for _, r in df_split.iterrows():
        stem = r["stem"]
        label = "positives" if int(r["label"]) == 1 else "negatives"
        edf_src, tsv_src = pick_paths(tsv_root, stem)
        # destino
        base_dst = out_dir / subset_name / label
        edf_dst = base_dst / f"{stem}.edf"
        tsv_dst = base_dst / f"{stem}.tsv"
        # cria symlinks
        if edf_src.exists():
            make_symlink(edf_src, edf_dst)
        if tsv_src.exists():
            make_symlink(tsv_src, tsv_dst)

# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="CSV com colunas ['stem', 'duration_s']")
    ap.add_argument("--labels-mode", choices=["folders", "tsv"], required=True)
    ap.add_argument("--labels-from-folders", nargs=2, metavar=("POS_DIR", "NEG_DIR"),
                    help="Pastas com .edf para rotular (modo folders)")
    ap.add_argument("--tsv-root", help="Raiz onde estão Sleep_Data/<stem>.tsv/.edf (modo tsv)")
    ap.add_argument("--ahi-threshold", type=float, default=3.0, help="Limiar de AHI para label=1 (default=3.0)")
    ap.add_argument("--no-hypopnea", action="store_true", help="Não incluir hypopnea no AHI (por omissão inclui)")
    ap.add_argument("--out-ahi-csv", help="Guarda tabela de AHI por estudo (modo tsv)")
    ap.add_argument("--out-dir", required=True, help="Diretório base para os splits com symlinks")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    manifest = read_manifest(args.manifest)

    if args.labels_mode == "folders":
        if not args.labels_from_folders or len(args.labels_from_folders) != 2:
            raise SystemExit("--labels-from-folders requer POS_DIR e NEG_DIR")
        pos_dir, neg_dir = map(Path, args.labels_from_folders)
        df_lbl = labels_from_folders(manifest, pos_dir, neg_dir)
        labeling_info = f"labeling=folders | labeled={len(df_lbl)}"

    else:  # 'tsv'
        if not args.tsv_root:
            raise SystemExit("--tsv-root é obrigatório no modo labels-mode=tsv")
        tsv_root = Path(args.tsv_root)
        df_lbl = labels_from_tsv(
            manifest,
            tsv_root=tsv_root,
            ahi_threshold=args.ahi_threshold,
            include_hypopnea=(not args.no_hypopnea),
            out_ahi_csv=Path(args.out_ahi_csv) if args.out_ahi_csv else None
        )
        labeling_info = f"labeling=tsv | labeled={len(df_lbl)}"

    if df_lbl.empty:
        raise SystemExit("Nenhum estudo rotulado. Verifique os caminhos e parâmetros.")

    # Splits 80/10/10
    df_train, df_val, df_test = stratified_splits(df_lbl, test_size=0.10, val_size=0.10, seed=args.seed)

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    # Materializa symlinks (usa tsv_root se for modo tsv; no modo folders
    # também precisa apontar para os ficheiros originais. Usaremos a mesma
    # lógica de tsv_root=parent comum 'Sleep_Data'.)
    if args.labels_mode == "tsv":
        tsv_root = Path(args.tsv_root)
    else:
        # tenta deduzir um root comum com Sleep_Data; se não houver, usa o pai de POS_DIR
        pos_dir, _ = map(Path, args.labels_from_folders)
        tsv_root = pos_dir.parent.parent if (pos_dir.parent.name != "Sleep_Data") else pos_dir.parent.parent

    materialize_symlinks(df_train, "train", out_dir, tsv_root)
    materialize_symlinks(df_val,   "val",   out_dir, tsv_root)
    materialize_symlinks(df_test,  "test",  out_dir, tsv_root)

    # Resumo
    def c(df): 
        return int((df["label"]==1).sum()), int((df["label"]==0).sum())
    p_tr, n_tr = c(df_train); p_va, n_va = c(df_val); p_te, n_te = c(df_test)

    print(f"🧾 Manifest: {len(manifest)} lines | {labeling_info}")
    print(f"✅ Splits created in: {out_dir}")
    print(f"   Train={len(df_train)} | Val={len(df_val)} | Test={len(df_test)}")
    print(f"   Pos(train)={p_tr}  Neg(train)={n_tr}")
    print(f"   Pos(val)  ={p_va}  Neg(val)  ={n_va}")
    print(f"   Pos(test) ={p_te}  Neg(test) ={n_te}")

if __name__ == "__main__":
    main()
