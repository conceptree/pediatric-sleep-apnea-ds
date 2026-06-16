from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


ENCODING_CANDIDATES = ["utf-8-sig", "utf-8", "cp1252", "latin-1"]
SAMPLE_SIZE = 300
TOP_N = 200
KEYWORDS = ("apnea", "hypopnea", "rera", "desat", "snore", "arousal", "obstructive", "central", "mixed")


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
            f.read(1024)
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
            raw_path = (row.get("tsv_path") or "").strip()
            if raw_path:
                tsv_paths.append(Path(raw_path))
    return tsv_paths


def normalize_description(text: str) -> str:
    return " ".join(text.strip().split())


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    cfg = load_paths_config(project_root / "configs" / "paths.yaml")
    outputs_dir = Path(cfg["outputs_dir"])

    manifest_csv = outputs_dir / "inventory" / "paired_sleep_records_step4_splits.csv"
    if not manifest_csv.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_csv}. Run step 4 first.")

    tsv_paths = read_manifest_tsv_paths(manifest_csv)
    sample_paths = tsv_paths[:SAMPLE_SIZE]

    desc_counter: Counter[str] = Counter()
    keyword_counter: Counter[str] = Counter()
    files_with_keyword = 0
    unreadable_count = 0
    total_rows = 0

    for tsv_path in sample_paths:
        if not tsv_path.exists():
            unreadable_count += 1
            continue

        file_has_keyword = False

        try:
            f, _ = open_with_fallback(tsv_path)
            with f:
                reader = csv.DictReader(f, delimiter="\t")
                # Expect columns: onset, duration, description
                for row in reader:
                    total_rows += 1
                    desc = normalize_description((row.get("description") or ""))
                    if not desc:
                        continue
                    desc_counter[desc] += 1

                    low = desc.lower()
                    matched = [kw for kw in KEYWORDS if kw in low]
                    if matched:
                        file_has_keyword = True
                        for kw in matched:
                            keyword_counter[kw] += 1
        except Exception:
            unreadable_count += 1
            continue

        if file_has_keyword:
            files_with_keyword += 1

    out_dir = outputs_dir / "inventory"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_txt = out_dir / "tsv_description_profile_step8.txt"
    out_csv = out_dir / "tsv_description_top_values_step8.csv"

    lines: list[str] = []
    lines.append(f"manifest_csv: {manifest_csv}")
    lines.append(f"tsv_total_in_manifest: {len(tsv_paths)}")
    lines.append(f"sample_size_requested: {SAMPLE_SIZE}")
    lines.append(f"sample_size_effective: {len(sample_paths)}")
    lines.append(f"unreadable_count: {unreadable_count}")
    lines.append(f"total_rows_scanned: {total_rows}")
    lines.append(f"distinct_descriptions: {len(desc_counter)}")
    lines.append(f"files_with_keyword: {files_with_keyword}")
    lines.append("")

    lines.append("keyword_hits:")
    if keyword_counter:
        for kw, cnt in keyword_counter.most_common():
            lines.append(f"  {kw}: {cnt}")
    else:
        lines.append("  (none)")
    lines.append("")

    lines.append(f"top_descriptions_first_{TOP_N}:")
    for desc, cnt in desc_counter.most_common(TOP_N):
        lines.append(f"  {cnt:6d} | {desc}")

    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["description", "count"])
        for desc, cnt in desc_counter.most_common(TOP_N):
            writer.writerow([desc, cnt])

    print(f"Wrote {out_txt}")
    print(f"Wrote {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
