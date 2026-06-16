#!/usr/bin/env python3
# file: check_feature_leakage.py
import argparse
import re
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

SUSPECT_PATTERNS = [
    r"\bahi\b", r"apnea", r"hypop", r"score", r"label", r"target",
    r"diagn", r"result", r"severity", r"gt_", r"gold", r"truth"
]

def is_probably_binary(s: pd.Series) -> bool:
    vals = np.unique(s.dropna().values)
    return len(vals) <= 2 and set(vals).issubset({0,1})

def main():
    ap = argparse.ArgumentParser(description="Checagem de possíveis leaks via nomes e AUC univariado.")
    ap.add_argument("--features", required=True, help="features.parquet com colunas [label, subset, ...]")
    ap.add_argument("--out-csv", required=False, default=None, help="Opcional: salva ranking de AUC por feature")
    ap.add_argument("--auc-flag", type=float, default=0.95, help="AUC >= este valor será marcado como suspeito (default=0.95)")
    args = ap.parse_args()

    df = pd.read_parquet(args.features)
    assert "label" in df.columns, "Arquivo precisa ter coluna 'label'"
    assert "subset" in df.columns, "Arquivo precisa ter coluna 'subset'"

    # 1) Nomes suspeitos
    sus_regex = re.compile("|".join(SUSPECT_PATTERNS), flags=re.IGNORECASE)
    name_suspects = [c for c in df.columns if sus_regex.search(c)]
    name_suspects = [c for c in name_suspects if c not in ("label","subset","stem","patient")]
    
    # 2) AUC univariado por feature numérica
    y = df["label"].astype(int).values
    feat_cols = [c for c in df.columns if c not in ("label","subset","stem","patient")]
    auc_rows = []
    for c in feat_cols:
        s = df[c]
        # precisa ser numerico e não constante
        if not np.issubdtype(s.dtype, np.number): 
            continue
        if s.nunique(dropna=True) <= 1:
            continue
        # AUC só faz sentido com variação em y e na feature
        try:
            auc = roc_auc_score(y, s.fillna(s.median()))
            auc_rows.append((c, float(auc)))
        except Exception:
            pass

    auc_df = pd.DataFrame(auc_rows, columns=["feature","auc"]).sort_values("auc", ascending=False)

    # 3) Quase-constantes por subset (podem denunciar alguma contagem do split)
    quasi_constant = {}
    for sub, g in df.groupby("subset"):
        subs = []
        for c in feat_cols:
            s = g[c]
            if np.issubdtype(s.dtype, np.number):
                # proporção do valor mais frequente
                freq = s.value_counts(dropna=True, normalize=True)
                if len(freq) > 0 and freq.iloc[0] >= 0.99:
                    subs.append(c)
        quasi_constant[sub] = subs

    # 4) Relatório simples
    print("=== POSSÍVEIS SINAIS DE LEAKAGE POR COLUNA ===")
    if name_suspects:
        print(f"⚠️  Nomes suspeitos ({len(name_suspects)}): {name_suspects[:15]}{' ...' if len(name_suspects)>15 else ''}")
    else:
        print("🟢 Sem nomes de colunas suspeitos por regex.")

    high_auc = auc_df[auc_df["auc"] >= args.auc_flag]
    if not high_auc.empty:
        print(f"⚠️  Features com AUC >= {args.auc_flag} (potencial vazamento):")
        print(high_auc.head(20).to_string(index=False))
    else:
        print(f"🟢 Nenhuma feature com AUC >= {args.auc_flag}.")

    for sub, cols in quasi_constant.items():
        if cols:
            print(f"⚠️  Quase-constantes em '{sub}': {len(cols)} (ex: {cols[:10]})")
        else:
            print(f"🟢 Nenhuma quase-constante em '{sub}'.")

    if args.out_csv:
        auc_df.to_csv(args.out_csv, index=False)
        print(f"📄 Ranking AUC salvo em: {args.out_csv}")

if __name__ == "__main__":
    main()