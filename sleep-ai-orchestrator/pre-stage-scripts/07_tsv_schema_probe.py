from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


ENCODING_CANDIDATES = ["utf-8-sig", "utf-8", "cp1252", "latin-1"]
SAMPLE_SIZE = 200
KEYWORDS = ("ahi", "oahi", "apnea", "hypopnea", "rdi", "arousal", "desat", "spo2")


def load_paths_config(config_path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip().strip('"').strip("'")
    return data


def open_with_fallback(path: Path):
    last_error: Exception | None = None
    for encoding in ENCODING_CANDIDATES:
        try:
            f = path.open("r", encoding=encoding, newline="")
            f.read(2048)
            f.seek(0)
            return f, encoding
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
    raise UnicodeDecodeError(
        "unknown",
        b"",
        0,
        1,
        f"Unable to decode {path} using {ENCODING_CANDIDATES}. Last error: {last_error}",
    )


def read_manifest_tsv_paths(manifest_csv: Path) -> list[Path]:
    tsv_paths: list[Path] = []
    with manifest_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            p = Path((row.get("tsv_path") or "").strip())
            if p:
                tsv_paths.append(p)
    return tsv_paths


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    cfg = load_paths_config(project_root / "configs" / "paths.yaml")
    outputs_dir = Path(cfg["outputs_dir"])

    manifest_csv = outputs_dir / "inventory" / "paired_sleep_records_step4_splits.csv"
    if not manifest_csv.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_csv}. Run step 4 first.")

    tsv_paths = read_manifest_tsv_paths(manifest_csv)
    sample_paths = tsv_paths[:SAMPLE_SIZE]

    schema_counter: Counter[str] = Counter()
    column_counter: Counter[str] = Counter()
    candidate_counter: Counter[str] = Counter()
    encoding_counter: Counter[str] = Counter()
    unreadable: list[str] = []

    for p in sample_paths:
        if not p.exists():
            unreadable.append(str(p))
            continue

        try:
            f, used_encoding = open_with_fallback(p)
            encoding_counter[used_encoding] += 1
            with f:
                reader = csv.reader(f, delimiter="\t")
                header = next(reader, [])
        except Exception:
            unreadable.append(str(p))
            continue

        normalized = [h.strip() for h in header if h is not None]
        schema_key = "|".join(normalized)
        schema_counter[schema_key] += 1

        for col in normalized:
            if not col:
                continue
            column_counter[col] += 1
            low = col.lower()
            if any(k in low for k in KEYWORDS):
                candidate_counter[col] += 1

    out_file = outputs_dir / "inventory" / "tsv_schema_step7.txt"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append(f"manifest_csv: {manifest_csv}")
    lines.append(f"tsv_total_in_manifest: {len(tsv_paths)}")
    lines.append(f"sample_size_requested: {SAMPLE_SIZE}")
    lines.append(f"sample_size_effective: {len(sample_paths)}")
    lines.append(f"unreadable_count: {len(unreadable)}")
    lines.append("")

    lines.append("encodings_used:")
    for enc, count in encoding_counter.most_common():
        lines.append(f"  {enc}: {count}")
    lines.append("")

    lines.append(f"distinct_header_schemas: {len(schema_counter)}")
    lines.append("top_header_schemas:")
    for idx, (schema, count) in enumerate(schema_counter.most_common(5), start=1):
        lines.append(f"  {idx}. count={count}")
        lines.append(f"     header={schema}")
    lines.append("")

    lines.append("top_columns:")
    for col, count in column_counter.most_common(40):
        lines.append(f"  {count:4d} {col}")
    lines.append("")

    lines.append("candidate_target_like_columns:")
    if candidate_counter:
        for col, count in candidate_counter.most_common():
            lines.append(f"  {count:4d} {col}")
    else:
        lines.append("  (none found in sampled TSV headers)")

    if unreadable:
        lines.append("")
        lines.append("unreadable_sample:")
        for p in unreadable[:20]:
            lines.append(f"  - {p}")

    out_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
