cd $1
echo "EDF: $(ls -1 *.edf | wc -l)"
echo "TSV: $(ls -1 *.tsv | wc -l)"
echo "ATR: $(ls -1 *.atr 2>/dev/null | wc -l)"
echo "TOTAL: $(ls -1 *.* | wc -l)"
