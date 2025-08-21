# fix_missing_small.sh
#!/usr/bin/env bash

# Usage:
# bash fixFiles.sh \
#   path/to/nch-sleep/{positives|negatives} \
#   {physionet_username}

set -euo pipefail

BASE="https://physionet.org/files/nch-sleep/3.1.0/Sleep_Data"
DIR="$1"
USR="$2"

cd "$DIR" || { echo "❌ Dir not found: $DIR"; exit 1; }
echo "📁 Folder: $PWD"
echo "👤 PhysioNet user: $USR"

# 1) For each .edf that misses the corresponding .tsv, download it
shopt -s nullglob
edfs=( *.edf )
if (( ${#edfs[@]} )); then
  for edf in "${edfs[@]}"; do
    stem="${edf%.edf}"
    if [[ ! -f "$stem.tsv" ]]; then
      echo "↳ Missing TSV for $stem  → downloading…"
      wget -c -N --user="$USR" --ask-password "$BASE/$stem.tsv"
    fi
  done
else
  echo "⚠️  No .edf found in this folder."
fi

# 2) re-download pairs marked as "small" (if ../small_list.txt exists)
SMALL_LIST="../small_list.txt"
if [[ -f "$SMALL_LIST" ]]; then
  echo "📄 Found $SMALL_LIST — re-downloading listed EDF+TSV pairs."
  while IFS= read -r stem; do
    [[ -z "$stem" ]] && continue
    echo "↳ Re-downloading $stem"
    wget -c -N --user="$USR" --ask-password \
      "$BASE/$stem.edf" "$BASE/$stem.tsv"
  done < "$SMALL_LIST"
fi

echo "✅ Done."
