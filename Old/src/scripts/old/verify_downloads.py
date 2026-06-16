#!/usr/bin/env python3
""" 
# USAGE
## With Apnea:
python3 verify_downloads.py \
  --records ../../../lists/com_apneia_records.txt \
  --data-dir ../../../datasets/nch-sleep 
## Without Apnea:
python3 verify_downloads.py \
  --records ../../../lists/sem_apneia_records.txt \
  --data-dir ../../../datasets/nch-sleep
"""

import argparse
from pathlib import Path

def verify_records(records_file, base_dir):
    base_dir = Path(base_dir).expanduser().resolve()
    prefixes = Path(records_file).read_text().splitlines()

    missing = {}
    for p in prefixes:
        pat_id = p.split('_')[0]
        pat_dir = base_dir / pat_id
        expected = [pat_dir / f"{p}.edf", pat_dir / f"{p}.tsv", pat_dir / f"{p}.atr"]

        not_found = [str(f) for f in expected if not f.exists()]
        if not_found:
            missing[p] = not_found
    return missing

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True, help="Ficheiro *_records.txt gerado pelo diagnosisFilter.py")
    ap.add_argument("--data-dir", required=True, help="Diretório onde foram guardados os ficheiros (ex: ../../../datasets/nch-sleep)")
    args = ap.parse_args()

    missing = verify_records(args.records, args.data_dir)
    if not missing:
        print("✅ Todos os ficheiros estão completos!")
    else:
        print(f"⚠️ {len(missing)} estudos com ficheiros em falta:\n")
        for prefix, files in missing.items():
            print(f"{prefix}: faltam {', '.join(files)}")

if __name__ == "__main__":
    main()

