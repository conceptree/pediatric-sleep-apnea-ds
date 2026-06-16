# ✅ verifyUnified.py

Script to **verify the integrity** of PhysioNet-style datasets (like NCH Sleep), containing `.edf` and `.tsv` files.

Supports two modes:
- `--fast`: quick check, based on file presence and minimal size
- `--full`: thorough check, parsing `.edf` and `.tsv` files to inspect metadata, channel counts, durations, etc.

---

## 📁 Expected folder structure

```
datasets/
└── nch-sleep/
    ├── positives/
    │   ├── 145_1537.edf
    │   ├── 145_1537.tsv
    │   └── ...
    ├── negatives/
    │   ├── 1261_23965.edf
    │   ├── 1261_23965.tsv
    │   └── ...
    └── all_raw/
        ├── *.edf
        └── *.tsv
```

---

## ⚡ Fast verification (`--fast`)

### Purpose:
- Checks if both `.edf` and `.tsv` files **exist** and meet a **minimum file size**
- Ideal for quick validation after downloads

### Command:

```bash
python3 verifyUnified.py   --base-dir /path/to/nch-sleep/all_raw   --fast
```

### Optional parameters:

- `--min-edf`: Minimum size in bytes for `.edf` files (default = 50MB)
- `--min-tsv`: Minimum size in bytes for `.tsv` files (default = 10KB)

---

## 🔍 Full verification (`--full`)

### Purpose:
- Parses `.edf` files using `mne.io.read_raw_edf()`
  - Checks number of channels and total duration
- Parses `.tsv` files as event logs
  - Checks number of rows and column structure
- Saves a detailed **CSV report** (optional)

### Command:

```bash
python3 verifyUnified.py   --base-dir /path/to/nch-sleep/all_raw   --full   --out-csv /path/to/save/report.csv
```

> 💾 The CSV contains one row per `stem`, with columns like:
> - `stem`, `has_tsv`, `n_channels`, `duration_s`, `n_tsv_rows`, etc.

---

## ✅ Example usages

### Example 1: Fast check for `positives`
```bash
python3 verifyUnified.py --base-dir ./datasets/nch-sleep/positives --fast
```

### Example 2: Full check with CSV output
```bash
python3 verifyUnified.py --base-dir ./datasets/nch-sleep/all_raw --full --out-csv ./reports/full_check.csv
```

### Example 3: Custom file size thresholds
```bash
python3 verifyUnified.py --base-dir ./datasets/nch-sleep/all_raw --fast --min-edf 40000000 --min-tsv 8000
```

---

## 📊 Example output

### Fast mode
```text
📊 Fast check: 112 valid pairs, 4 missing EDF, 2 missing TSV
```

### Full mode
```text
📊 Full check: 120 EDFs verified
💾 CSV report saved to ./reports/full_check.csv
```

---

## ⚙️ Dependencies

- Python ≥ 3.8
- `pandas`
- `mne`
- `argparse`

Install with pip:

```bash
pip install pandas mne
```

---

## 🚧 Optional future improvements

- [ ] Duplicate stem detection (label flip across classes)
- [ ] JSON summary output
- [ ] Orphan detection (`.edf` without `.tsv`, or vice versa)

---

## 👤 Author

- 🧠 MSc Thesis – Nuno Rodrigues  
- 🛠️ Engineering support: ChatGPT + PhysioNet best practices