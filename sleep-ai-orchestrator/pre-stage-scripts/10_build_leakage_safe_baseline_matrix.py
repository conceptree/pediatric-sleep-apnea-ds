from __future__ import annotations

import csv
from pathlib import Path


TARGET_COLUMN = "label_respiratory_burden_ge_5"

# Features deliberately selected to reduce direct target leakage.
SAFE_FEATURE_COLUMNS = [
    "event_rows_total",
    "recording_hours_est",
    "n_oxygen_desaturation",
    "n_eeg_arousal",
]

# Columns derived directly from apnea/hypopnea/rera burden and excluded from baseline inputs.
LEAKAGE_PRONE_COLUMNS = [
    "n_obstructive_apnea",
    "n_obstructive_hypopnea",
    "n_hypopnea_any",
    "n_apnea_any",
    "n_central_apnea",
    "n_mixed_apnea",
    "n_rera",
    "n_obstructive_events",
    "n_respiratory_events",
    "obstructive_event_rate_per_hour",
    "respiratory_event_rate_per_hour",
    "label_obstructive_any",
]


def load_paths_config(config_path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip().strip('"').strip("'")
    return data


def parse_float(value: str | None) -> float:
    if value is None:
        return 0.0
    text = value.strip()
    if not text:
        return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def parse_int(value: str | None) -> int:
    if value is None:
        return 0
    text = value.strip()
    if not text:
        return 0
    try:
        return int(float(text))
    except ValueError:
        return 0


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    cfg = load_paths_config(project_root / "configs" / "paths.yaml")
    outputs_dir = Path(cfg["outputs_dir"])

    input_csv = outputs_dir / "inventory" / "record_event_features_step9.csv"
    out_dir = outputs_dir / "inventory"
    out_dir.mkdir(parents=True, exist_ok=True)

    output_csv = out_dir / "baseline_matrix_step10.csv"
    guardrail_txt = out_dir / "baseline_matrix_step10_guardrails.txt"

    if not input_csv.exists():
        raise FileNotFoundError(f"Missing input: {input_csv}. Run step 9 first.")

    rows_out: list[dict[str, str]] = []
    split_counts = {"train": 0, "val": 0, "test": 0}
    target_counts = {0: 0, 1: 0}

    with input_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        expected = set(["record_id", "patient_id", "split", TARGET_COLUMN, *SAFE_FEATURE_COLUMNS])
        missing = [c for c in expected if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Missing required columns in step9 file: {missing}")

        for row in reader:
            split = (row.get("split") or "").strip()
            if split not in split_counts:
                continue

            target = parse_int(row.get(TARGET_COLUMN))
            if target not in (0, 1):
                continue

            out_row = {
                "record_id": (row.get("record_id") or "").strip(),
                "patient_id": (row.get("patient_id") or "").strip(),
                "split": split,
                "target_respiratory_burden_ge_5": str(target),
                "feature_event_rows_total": str(parse_int(row.get("event_rows_total"))),
                "feature_recording_hours_est": f"{parse_float(row.get('recording_hours_est')):.6f}",
                "feature_n_oxygen_desaturation": str(parse_int(row.get("n_oxygen_desaturation"))),
                "feature_n_eeg_arousal": str(parse_int(row.get("n_eeg_arousal"))),
            }

            rows_out.append(out_row)
            split_counts[split] += 1
            target_counts[target] += 1

    fieldnames = [
        "record_id",
        "patient_id",
        "split",
        "target_respiratory_burden_ge_5",
        "feature_event_rows_total",
        "feature_recording_hours_est",
        "feature_n_oxygen_desaturation",
        "feature_n_eeg_arousal",
    ]

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    lines = [
        f"input_csv: {input_csv}",
        f"output_csv: {output_csv}",
        f"rows_output: {len(rows_out)}",
        f"split_train: {split_counts['train']}",
        f"split_val: {split_counts['val']}",
        f"split_test: {split_counts['test']}",
        f"target_positive: {target_counts[1]}",
        f"target_negative: {target_counts[0]}",
        "",
        f"target_column: {TARGET_COLUMN}",
        f"safe_feature_columns: {SAFE_FEATURE_COLUMNS}",
        "",
        "excluded_leakage_prone_columns:",
    ]
    lines.extend([f"  - {c}" for c in LEAKAGE_PRONE_COLUMNS])

    guardrail_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {output_csv}")
    print(f"Wrote {guardrail_txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
