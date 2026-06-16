from __future__ import annotations

import csv
from pathlib import Path


ENCODING_CANDIDATES = ["utf-8-sig", "utf-8", "cp1252", "latin-1"]


def load_paths_config(config_path: Path) -> dict[str, str]:
    """Load a minimal key/value YAML config without external dependencies."""
    data: dict[str, str] = {}

    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue

        key, value = line.split(":", 1)
        data[key.strip()] = value.strip().strip('"').strip("'")

    return data


def open_csv_with_fallback(csv_path: Path):
    last_error: Exception | None = None

    for encoding in ENCODING_CANDIDATES:
        try:
            f = csv_path.open("r", encoding=encoding, newline="")
            # Probe decode and rewind so DictReader reads from start.
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
        f"Unable to decode {csv_path} with candidates {ENCODING_CANDIDATES}. Last error: {last_error}",
    )


def normalize_id(value: str | None) -> str:
    if value is None:
        return ""
    return value.strip().strip('"').strip("'")


def collect_manifest_ids(manifest_csv: Path) -> tuple[set[str], set[str]]:
    patient_ids: set[str] = set()
    sleep_study_ids: set[str] = set()

    with manifest_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            record_id = normalize_id(row.get("record_id"))
            if not record_id or "_" not in record_id:
                continue

            pid, sid = record_id.split("_", 1)
            if pid:
                patient_ids.add(pid)
            if sid:
                sleep_study_ids.add(sid)

    return patient_ids, sleep_study_ids


def collect_column_values(csv_path: Path, column_name: str) -> tuple[set[str], str]:
    values: set[str] = set()

    f, used_encoding = open_csv_with_fallback(csv_path)
    with f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return values, used_encoding

        # Normalize accidental BOM/empty first column headers.
        fieldnames = [fn.strip() if fn else "" for fn in reader.fieldnames]
        reader.fieldnames = fieldnames

        if column_name not in fieldnames:
            return values, used_encoding

        for row in reader:
            val = normalize_id(row.get(column_name))
            if val:
                values.add(val)

    return values, used_encoding


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    cfg = load_paths_config(project_root / "configs" / "paths.yaml")

    outputs_dir = Path(cfg["outputs_dir"])
    health_data_dir = Path(cfg["health_data_dir"])

    manifest_csv = outputs_dir / "inventory" / "paired_sleep_records_step4_splits.csv"
    demographic_csv = health_data_dir / "DEMOGRAPHIC.csv"
    sleep_study_csv = health_data_dir / "SLEEP_STUDY.csv"

    if not manifest_csv.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_csv}. Run step 4 first.")
    if not demographic_csv.exists() or not sleep_study_csv.exists():
        raise FileNotFoundError("Missing required health tables: DEMOGRAPHIC.csv and/or SLEEP_STUDY.csv")

    manifest_patients, manifest_sleep_studies = collect_manifest_ids(manifest_csv)

    demographic_patients, dem_encoding = collect_column_values(demographic_csv, "STUDY_PAT_ID")
    sleep_study_patients, ss_encoding = collect_column_values(sleep_study_csv, "STUDY_PAT_ID")
    sleep_study_ids, _ = collect_column_values(sleep_study_csv, "SLEEP_STUDY_ID")

    missing_in_demographic = sorted(manifest_patients - demographic_patients)
    missing_in_sleep_study_pat = sorted(manifest_patients - sleep_study_patients)
    missing_in_sleep_study_id = sorted(manifest_sleep_studies - sleep_study_ids)

    out_file = outputs_dir / "inventory" / "linkage_coverage_step6.txt"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    def pct(covered: int, total: int) -> float:
        if total == 0:
            return 0.0
        return (covered / total) * 100.0

    covered_dem = len(manifest_patients & demographic_patients)
    covered_ss_pat = len(manifest_patients & sleep_study_patients)
    covered_ss_id = len(manifest_sleep_studies & sleep_study_ids)

    lines = [
        f"manifest_csv: {manifest_csv}",
        f"demographic_csv: {demographic_csv}",
        f"sleep_study_csv: {sleep_study_csv}",
        f"demographic_encoding_used: {dem_encoding}",
        f"sleep_study_encoding_used: {ss_encoding}",
        "",
        f"manifest_unique_patients: {len(manifest_patients)}",
        f"manifest_unique_sleep_study_ids: {len(manifest_sleep_studies)}",
        "",
        f"coverage_patient_in_demographic: {covered_dem}/{len(manifest_patients)} ({pct(covered_dem, len(manifest_patients)):.2f}%)",
        f"coverage_patient_in_sleep_study: {covered_ss_pat}/{len(manifest_patients)} ({pct(covered_ss_pat, len(manifest_patients)):.2f}%)",
        f"coverage_sleep_study_id_in_sleep_study: {covered_ss_id}/{len(manifest_sleep_studies)} ({pct(covered_ss_id, len(manifest_sleep_studies)):.2f}%)",
        "",
        f"missing_patient_in_demographic_count: {len(missing_in_demographic)}",
        f"missing_patient_in_demographic_sample: {missing_in_demographic[:20]}",
        f"missing_patient_in_sleep_study_count: {len(missing_in_sleep_study_pat)}",
        f"missing_patient_in_sleep_study_sample: {missing_in_sleep_study_pat[:20]}",
        f"missing_sleep_study_id_in_sleep_study_count: {len(missing_in_sleep_study_id)}",
        f"missing_sleep_study_id_in_sleep_study_sample: {missing_in_sleep_study_id[:20]}",
    ]

    out_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
