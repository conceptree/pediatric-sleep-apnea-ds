from __future__ import annotations

import csv
import random
from pathlib import Path


RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


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


def extract_patient_id(record_id: str) -> str:
    """Extract patient identifier from record_id pattern 'patient_study'."""
    return record_id.split("_", 1)[0]


def assign_grouped_splits(patient_ids: list[str]) -> dict[str, str]:
    if abs((TRAIN_RATIO + VAL_RATIO + TEST_RATIO) - 1.0) > 1e-9:
        raise ValueError("Split ratios must sum to 1.0")

    shuffled = patient_ids[:]
    rnd = random.Random(RANDOM_SEED)
    rnd.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    splits: dict[str, str] = {}
    for pid in shuffled[:n_train]:
        splits[pid] = "train"
    for pid in shuffled[n_train : n_train + n_val]:
        splits[pid] = "val"
    for pid in shuffled[n_train + n_val :]:
        splits[pid] = "test"

    return splits


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    cfg = load_paths_config(project_root / "configs" / "paths.yaml")
    outputs_dir = Path(cfg["outputs_dir"])

    inventory_dir = outputs_dir / "inventory"
    input_manifest = inventory_dir / "paired_sleep_records_step3.csv"
    output_manifest = inventory_dir / "paired_sleep_records_step4_splits.csv"
    output_qc = inventory_dir / "paired_sleep_records_step4_splits_qc.txt"

    if not input_manifest.exists():
        raise FileNotFoundError(
            f"Missing input manifest: {input_manifest}. Run step 3 first."
        )

    rows: list[dict[str, str]] = []
    with input_manifest.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        raise ValueError("Input manifest is empty")

    patient_ids = sorted({extract_patient_id(r["record_id"]) for r in rows})
    patient_to_split = assign_grouped_splits(patient_ids)

    # Persist split assignment on each record while keeping all original columns.
    fieldnames = ["record_id", "patient_id", "split", "edf_path", "tsv_path"]
    output_rows: list[dict[str, str]] = []
    for r in rows:
        patient_id = extract_patient_id(r["record_id"])
        split = patient_to_split[patient_id]
        output_rows.append(
            {
                "record_id": r["record_id"],
                "patient_id": patient_id,
                "split": split,
                "edf_path": r["edf_path"],
                "tsv_path": r["tsv_path"],
            }
        )

    with output_manifest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    rec_counts = {"train": 0, "val": 0, "test": 0}
    pat_counts = {"train": 0, "val": 0, "test": 0}

    for row in output_rows:
        rec_counts[row["split"]] += 1

    for _, split in patient_to_split.items():
        pat_counts[split] += 1

    qc_lines = [
        f"random_seed: {RANDOM_SEED}",
        f"ratios: train={TRAIN_RATIO}, val={VAL_RATIO}, test={TEST_RATIO}",
        f"records_total: {len(output_rows)}",
        f"patients_total: {len(patient_to_split)}",
        f"records_train: {rec_counts['train']}",
        f"records_val: {rec_counts['val']}",
        f"records_test: {rec_counts['test']}",
        f"patients_train: {pat_counts['train']}",
        f"patients_val: {pat_counts['val']}",
        f"patients_test: {pat_counts['test']}",
    ]
    output_qc.write_text("\n".join(qc_lines) + "\n", encoding="utf-8")

    print(f"Wrote {output_manifest}")
    print(f"Wrote {output_qc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
