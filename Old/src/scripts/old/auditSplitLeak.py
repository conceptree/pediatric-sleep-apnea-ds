#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, sys
from pathlib import Path
import pandas as pd

SUBSETS = ("train", "val", "test")
LABELS = ("positives", "negatives")
VALID_EXT = {".edf", ".tsv", ".atr"}  # .atr é opcional

def stem_of(p: Path) -> str:
    return p.stem  # "1234_5678" de "1234_5678.edf"

def collect_splits(base: Path):
    """
    Varre base/subset/label e devolve DataFrame com:
    subset, label, stem, ext, path, is_symlink, realpath
    """
    rows = []
    for subset in SUBSETS:
        for label in LABELS:
            d = base / subset / label
            if not d.exists():
                continue
            for f in d.iterdir():
                if f.is_file() and f.suffix.lower() in VALID_EXT:
                    rows.append({
                        "subset": subset,
                        "label": 1 if label == "positives" else 0,
                        "label_name": label,
                        "stem": stem_of(f),
                        "ext": f.suffix.lower(),
                        "path": str(f),
                        "is_symlink": f.is_symlink(),
                        "realpath": str(f.resolve()) if f.exists() else ""
                    })
    if not rows:
        return pd.DataFrame(columns=["subset","label","label_name","stem","ext","path","is_symlink","realpath"])
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser(description="Auditoria de fuga entre splits (train/val/test).")
    ap.add_argument("--base-splits", required=True, help="Pasta base dos splits (contendo train/ val/ test/).")
    ap.add_argument("--out-csv", default=None, help="Opcional: grava CSV com todas as linhas varridas.")
    ap.add_argument("--out-report", default=None, help="Opcional: grava um resumo em TXT.")
    args = ap.parse_args()

    base = Path(args.base_splits).expanduser()
    if not base.exists():
        print(f"❌ Base não existe: {base}", file=sys.stderr)
        sys.exit(1)

    df = collect_splits(base)

    if df.empty:
        msg = f"⚠️ Nenhum ficheiro encontrado em {base}/(train|val|test)/(positives|negatives)"
        print(msg)
        if args.out_report:
            Path(args.out_report).write_text(msg + "\n", encoding="utf-8")
        sys.exit(0)

    if args.out_csv:
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out_csv, index=False)

    # --- Resumo de contagens ---
    by_subset = df.groupby("subset")["stem"].nunique().to_dict()
    by_label = df.groupby("label_name")["stem"].nunique().to_dict()
    total_unique_stems = df["stem"].nunique()

    print("=== RESUMO ===")
    print(f"Base: {base}")
    print(f"Stems únicos (global): {total_unique_stems}")
    print("Stems únicos por subset:", by_subset)
    print("Stems únicos por label :", by_label)

    # --- Fuga entre subsets (mesmo stem em múltiplos subsets) ---
    stems_subsets = (df.groupby("stem")["subset"]
                     .nunique()
                     .reset_index(name="n_subsets"))
    leak_between_subsets = stems_subsets[stems_subsets["n_subsets"] > 1]["stem"].tolist()

    # --- Label flip (mesmo stem aparece em positivo e negativo em QUALQUER subset) ---
    stems_labels = (df.groupby("stem")["label_name"]
                    .nunique()
                    .reset_index(name="n_labels"))
    label_flip = stems_labels[stems_labels["n_labels"] > 1]["stem"].tolist()

    # --- Checagem de pares .edf/.tsv por stem (não é leakage, mas útil) ---
    has_edf = df[df["ext"]==".edf"].groupby("stem").size()
    has_tsv = df[df["ext"]==".tsv"].groupby("stem").size()
    no_tsv = sorted(list(set(has_edf.index) - set(has_tsv.index)))
    no_edf = sorted(list(set(has_tsv.index) - set(has_edf.index)))

    # --- Symlinks: realpath duplicado em diferentes subsets? ---
    # Se o mesmo realpath aparece em >1 subset, é sinal de possível fuga via link/copiar
    realpath_subset = df[df["is_symlink"]==True].groupby(["realpath"])["subset"].nunique().reset_index()
    realpath_multi_subset = realpath_subset[realpath_subset["subset"] > 1]["realpath"].tolist()

    print("\n=== POSSÍVEIS FUGAS ===")
    if leak_between_subsets:
        print(f"🔴 Mesmos stems em múltiplos subsets ({len(leak_between_subsets)}): {leak_between_subsets[:10]}{' ...' if len(leak_between_subsets)>10 else ''}")
    else:
        print("🟢 Sem stems repetidos entre train/val/test.")

    if label_flip:
        print(f"🔴 Label flip (stems com pos e neg) ({len(label_flip)}): {label_flip[:10]}{' ...' if len(label_flip)>10 else ''}")
    else:
        print("🟢 Sem label flip (stems com rótulos inconsistentes).")

    print("\n=== QUALIDADE DOS PARES ===")
    if no_tsv:
        print(f"🟡 Stems com .edf mas sem .tsv ({len(no_tsv)}): {no_tsv[:10]}{' ...' if len(no_tsv)>10 else ''}")
    else:
        print("🟢 Todos os .edf têm .tsv correspondente (por stem).")

    if no_edf:
        print(f"🟡 Stems com .tsv mas sem .edf ({len(no_edf)}): {no_edf[:10]}{' ...' if len(no_edf)>10 else ''}")
    else:
        print("🟢 Todos os .tsv têm .edf correspondente (por stem).")

    print("\n=== SYMLINKS ===")
    n_symlinks = int(df["is_symlink"].sum())
    print(f"Symlinks encontrados: {n_symlinks}")
    if realpath_multi_subset:
        print(f"🔴 realpath(s) presentes em múltiplos subsets ({len(realpath_multi_subset)}):")
        for rp in realpath_multi_subset[:10]:
            print("   -", rp)
        if len(realpath_multi_subset) > 10:
            print("   ...")
    else:
        print("🟢 Nenhum realpath duplicado em múltiplos subsets.")

    # --- Relatório opcional ---
    if args.out_report:
        lines = []
        lines.append(f"Base: {base}")
        lines.append(f"Stems únicos (global): {total_unique_stems}")
        lines.append(f"Stems por subset: {by_subset}")
        lines.append(f"Stems por label : {by_label}")
        lines.append("")
        lines.append(f"Leak entre subsets: {len(leak_between_subsets)}")
        if leak_between_subsets:
            lines.append(", ".join(leak_between_subsets))
        lines.append(f"Label flip: {len(label_flip)}")
        if label_flip:
            lines.append(", ".join(label_flip))
        lines.append("")
        lines.append(f".edf sem .tsv: {len(no_tsv)}")
        if no_tsv:
            lines.append(", ".join(no_tsv))
        lines.append(f".tsv sem .edf: {len(no_edf)}")
        if no_edf:
            lines.append(", ".join(no_edf))
        lines.append("")
        lines.append(f"Symlinks: {n_symlinks}")
        lines.append(f"realpaths multi-subset: {len(realpath_multi_subset)}")
        if realpath_multi_subset:
            lines.extend(realpath_multi_subset)
        Path(args.out_report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_report).write_text("\n".join(lines) + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()
