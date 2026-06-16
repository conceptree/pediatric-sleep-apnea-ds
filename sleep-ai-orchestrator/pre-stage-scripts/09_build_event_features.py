from __future__ import annotations

import csv
from pathlib import Path


ENCODING_CANDIDATES = ["utf-8-sig", "utf-8", "cp1252", "latin-1"]


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
            return f
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


def parse_seconds(value: str | None) -> float | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None

    try:
        return float(text)
    except ValueError:
        pass

    if ":" in text:
        parts = text.split(":")
        try:
            if len(parts) == 3:
                h = float(parts[0])
                m = float(parts[1])
                s = float(parts[2])
                return h * 3600 + m * 60 + s
            if len(parts) == 2:
                m = float(parts[0])
                s = float(parts[1])
                return m * 60 + s
        except ValueError:
            return None

    return None


def as_rate(count: int, hours: float) -> float:
    if hours <= 0:
        return 0.0
    return count / hours


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    cfg = load_paths_config(project_root / "configs" / "paths.yaml")
    outputs_dir = Path(cfg["outputs_dir"])

    manifest_csv = outputs_dir / "inventory" / "paired_sleep_records_step4_splits.csv"
    if not manifest_csv.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_csv}. Run step 4 first.")

    out_dir = outputs_dir / "inventory"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "record_event_features_step9.csv"
    out_qc = out_dir / "record_event_features_step9_qc.txt"

    feature_rows: list[dict[str, str]] = []
    unreadable = 0

    with manifest_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            record_id = (row.get("record_id") or "").strip()
            patient_id = (row.get("patient_id") or "").strip()
            split = (row.get("split") or "").strip()
            tsv_path_text = (row.get("tsv_path") or "").strip()
            tsv_path = Path(tsv_path_text)

            if not record_id or not tsv_path_text or not tsv_path.exists():
                unreadable += 1
                continue

            n_rows = 0
            n_obstructive_apnea = 0
            n_obstructive_hypopnea = 0
            n_hypopnea_any = 0
            n_central_apnea = 0
            n_mixed_apnea = 0
            n_apnea_any = 0
            n_rera = 0
            n_oxygen_desaturation = 0
            n_eeg_arousal = 0

            min_onset: float | None = None
            max_end: float | None = None

            try:
                f_tsv = open_with_fallback(tsv_path)
                with f_tsv:
                    tsv_reader = csv.DictReader(f_tsv, delimiter="\t")
                    for ev in tsv_reader:
                        n_rows += 1

                        onset = parse_seconds(ev.get("onset"))
                        duration = parse_seconds(ev.get("duration"))
                        if onset is not None:
                            if min_onset is None or onset < min_onset:
                                min_onset = onset
                            if duration is not None and duration >= 0:
                                end = onset + duration
                                if max_end is None or end > max_end:
                                    max_end = end

                        desc = (ev.get("description") or "").strip().lower()
                        if not desc:
                            continue

                        if "obstructive apnea" in desc:
                            n_obstructive_apnea += 1
                        if "obstructive hypopnea" in desc:
                            n_obstructive_hypopnea += 1
                        if "hypopnea" in desc:
                            n_hypopnea_any += 1
                        if "central apnea" in desc:
                            n_central_apnea += 1
                        if "mixed apnea" in desc:
                            n_mixed_apnea += 1
                        if "apnea" in desc:
                            n_apnea_any += 1
                        if "rera" in desc:
                            n_rera += 1
                        if "oxygen desaturation" in desc or "desat" in desc:
                            n_oxygen_desaturation += 1
                        if "eeg arousal" in desc:
                            n_eeg_arousal += 1
            except Exception:
                unreadable += 1
                continue

            span_seconds = 0.0
            if min_onset is not None and max_end is not None and max_end >= min_onset:
                span_seconds = max_end - min_onset
            recording_hours = span_seconds / 3600.0 if span_seconds > 0 else 0.0

            n_obstructive_events = n_obstructive_apnea + n_obstructive_hypopnea
            n_respiratory_events = n_apnea_any + n_hypopnea_any + n_rera

            respiratory_event_rate_per_hour = as_rate(n_respiratory_events, recording_hours)
            obstructive_event_rate_per_hour = as_rate(n_obstructive_events, recording_hours)

            label_obstructive_any = 1 if n_obstructive_events >= 1 else 0
            # Exploratory threshold only for baseline prototyping (not a clinical decision rule).
            label_respiratory_burden_ge_5 = 1 if respiratory_event_rate_per_hour >= 5.0 else 0

            feature_rows.append(
                {
                    "record_id": record_id,
                    "patient_id": patient_id,
                    "split": split,
                    "tsv_path": str(tsv_path),
                    "event_rows_total": str(n_rows),
                    "recording_hours_est": f"{recording_hours:.4f}",
                    "n_obstructive_apnea": str(n_obstructive_apnea),
                    "n_obstructive_hypopnea": str(n_obstructive_hypopnea),
                    "n_hypopnea_any": str(n_hypopnea_any),
                    "n_apnea_any": str(n_apnea_any),
                    "n_central_apnea": str(n_central_apnea),
                    "n_mixed_apnea": str(n_mixed_apnea),
                    "n_rera": str(n_rera),
                    "n_oxygen_desaturation": str(n_oxygen_desaturation),
                    "n_eeg_arousal": str(n_eeg_arousal),
                    "n_obstructive_events": str(n_obstructive_events),
                    "n_respiratory_events": str(n_respiratory_events),
                    "obstructive_event_rate_per_hour": f"{obstructive_event_rate_per_hour:.4f}",
                    "respiratory_event_rate_per_hour": f"{respiratory_event_rate_per_hour:.4f}",
                    "label_obstructive_any": str(label_obstructive_any),
                    "label_respiratory_burden_ge_5": str(label_respiratory_burden_ge_5),
                }
            )

    fieldnames = [
        "record_id",
        "patient_id",
        "split",
        "tsv_path",
        "event_rows_total",
        "recording_hours_est",
        "n_obstructive_apnea",
        "n_obstructive_hypopnea",
        "n_hypopnea_any",
        "n_apnea_any",
        "n_central_apnea",
        "n_mixed_apnea",
        "n_rera",
        "n_oxygen_desaturation",
        "n_eeg_arousal",
        "n_obstructive_events",
        "n_respiratory_events",
        "obstructive_event_rate_per_hour",
        "respiratory_event_rate_per_hour",
        "label_obstructive_any",
        "label_respiratory_burden_ge_5",
    ]

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(feature_rows)

    n_total = len(feature_rows)
    n_obstructive_any = sum(int(r["label_obstructive_any"]) for r in feature_rows)
    n_burden_ge_5 = sum(int(r["label_respiratory_burden_ge_5"]) for r in feature_rows)

    qc_lines = [
        f"manifest_csv: {manifest_csv}",
        f"output_csv: {out_csv}",
        f"rows_output: {n_total}",
        f"unreadable_or_missing_tsv_count: {unreadable}",
        f"label_obstructive_any_positive: {n_obstructive_any}",
        f"label_obstructive_any_negative: {n_total - n_obstructive_any}",
        f"label_respiratory_burden_ge_5_positive: {n_burden_ge_5}",
        f"label_respiratory_burden_ge_5_negative: {n_total - n_burden_ge_5}",
    ]
    out_qc.write_text("\n".join(qc_lines) + "\n", encoding="utf-8")

    print(f"Wrote {out_csv}")
    print(f"Wrote {out_qc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
