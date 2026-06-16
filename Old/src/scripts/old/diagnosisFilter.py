import argparse, re, random
from pathlib import Path
import pandas as pd

APNEA_CODE_RX = re.compile(r'^G47\.3', re.I)        # G47.3*
APNEA_TEXT_RX = re.compile(r'(apne[ao]a|OSA\b)', re.I)  # apnea/apnoea/OSA

def resolve_cols(path: Path):
    cols = [c.upper() for c in pd.read_csv(path, nrows=0).columns]
    def pick(cands):  # devolve o 1º nome que existir no CSV (ignora maiúsc./minúsc.)
        for c in cands:
            if c.upper() in cols: return c
        raise KeyError(f"Nenhuma coluna encontrada entre {cands} em {path.name}")
    id_col   = pick(['STUDY_PAT_ID'])
    code_col = pick(['DIAGNOSIS_CODE','DX_CODE','CODE'])
    name_col = pick(['DIAGNOSIS_DESC','DX_NAME','DX_DESC','DESCRIPTION'])
    return id_col, code_col, name_col

def get_apnea_ids(diag_csv: Path) -> list[int]:
    id_col, code_col, name_col = resolve_cols(diag_csv)
    ids = set()
    for ch in pd.read_csv(
        diag_csv,
        usecols=[id_col, code_col, name_col],
        chunksize=200_000,
        dtype={id_col:'Int64', code_col:'string', name_col:'string'}
    ):
        code_hit = ch[code_col].fillna('').str.match(APNEA_CODE_RX)
        name_hit = ch[name_col].fillna('').str.contains(APNEA_TEXT_RX)
        m = code_hit | name_hit
        ids.update(ch.loc[m, id_col].dropna().astype('int64').tolist())
    return sorted(ids)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--docs-dir', required=True, help="Diretório com os CSVs (ex.: .../docs/Health_Data)")
    ap.add_argument('--base-url', default='https://physionet.org/files/nch-sleep/3.1.0')
    ap.add_argument('--sample-pos', type=int, default=50)
    ap.add_argument('--sample-neg', type=int, default=50)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out-dir', default='out')
    args = ap.parse_args()

    docs = Path(args.docs_dir).expanduser().resolve()
    diag_csv = docs / 'DIAGNOSIS.csv'
    sleep_csv = docs / 'SLEEP_STUDY.csv'
    if not diag_csv.exists():
        raise FileNotFoundError(f"Falta {diag_csv}")
    if not sleep_csv.exists():
        raise FileNotFoundError(f"Falta {sleep_csv}")

    out = Path(args.out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    # 1) IDs com apneia
    pos_ids = get_apnea_ids(diag_csv)

    # 2) Todos os pacientes presentes em SLEEP_STUDY
    ss = pd.read_csv(sleep_csv, usecols=['STUDY_PAT_ID','SLEEP_STUDY_ID'])
    all_ids = sorted(ss['STUDY_PAT_ID'].unique().tolist())
    neg_ids = sorted(set(all_ids) - set(pos_ids))

    # 3) Amostras equilibradas
    random.seed(args.seed)
    pos_sample = sorted(random.sample(pos_ids, min(args.sample_pos, len(pos_ids))))
    neg_sample = sorted(random.sample(neg_ids, min(args.sample_neg, len(neg_ids))))

    # 4) Prefixos STUDY_PAT_ID_SLEEP_STUDY_ID
    def to_prefix(df):
        return (df['STUDY_PAT_ID'].astype(str) + '_' + df['SLEEP_STUDY_ID'].astype(str)).unique().tolist()

    pos_records = to_prefix(ss[ss['STUDY_PAT_ID'].isin(pos_sample)])
    neg_records = to_prefix(ss[ss['STUDY_PAT_ID'].isin(neg_sample)])

    # 5) Guardar IDs/prefixos
    (out/'com_apneia_ids.txt').write_text('\n'.join(map(str, pos_sample)))
    (out/'sem_apneia_ids.txt').write_text('\n'.join(map(str, neg_sample)))
    (out/'com_apneia_records.txt').write_text('\n'.join(pos_records))
    (out/'sem_apneia_records.txt').write_text('\n'.join(neg_records))

    # 6) URLs para wget -i
    def urls(prefixes):
        L=[]
        for p in prefixes:
            L += [f"{args.base_url}/Sleep_Data/{p}.edf",
                  f"{args.base_url}/Sleep_Data/{p}.tsv",
                  f"{args.base_url}/Sleep_Data/{p}.atr"]
        return L
    (out/'com_apneia_urls.txt').write_text('\n'.join(urls(pos_records)))
    (out/'sem_apneia_urls.txt').write_text('\n'.join(urls(neg_records)))

    print(f"IDs com apneia (total): {len(pos_ids)} | sem apneia: {len(neg_ids)}")
    print(f"Amostras: +{len(pos_sample)} / -{len(neg_sample)}")
    print(f"Listas em: {out}")

if __name__ == "__main__":
    main()
