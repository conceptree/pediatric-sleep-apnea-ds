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


def detect_id_like_columns(columns: list[str]) -> list[str]:
    id_tokens = ("id", "patient", "subject", "record", "study", "encounter", "visit")
    lowered = [(c, c.lower()) for c in columns]
    return [original for original, low in lowered if any(tok in low for tok in id_tokens)]


def read_header_and_count_rows(csv_path: Path) -> tuple[list[str], int, str]:
    """Read CSV header and row count using a safe encoding fallback strategy."""
    last_error: Exception | None = None

    for encoding in ENCODING_CANDIDATES:
        try:
            with csv_path.open("r", encoding=encoding, newline="") as f:
                reader = csv.reader(f)
                header = next(reader, [])
                row_count = sum(1 for _ in reader)
            return header, row_count, encoding
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


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    cfg = load_paths_config(project_root / "configs" / "paths.yaml")

    health_data_dir = Path(cfg["health_data_dir"])
    outputs_dir = Path(cfg["outputs_dir"])

    csv_files = sorted(
        [
            p
            for p in health_data_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() == ".csv"
            and not p.name.startswith("._")
            and not p.name.startswith(".")
        ]
    )

    out_dir = outputs_dir / "inventory"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "health_data_schema_step5.txt"

    lines: list[str] = []
    lines.append(f"health_data_dir: {health_data_dir}")
    lines.append(f"csv_files_count: {len(csv_files)}")
    lines.append("")

    for csv_path in csv_files:
        header, row_count, encoding_used = read_header_and_count_rows(csv_path)
        id_like = detect_id_like_columns(header)

        lines.append(f"== {csv_path.name} ==")
        lines.append(f"encoding_used: {encoding_used}")
        lines.append(f"rows: {row_count}")
        lines.append(f"columns_count: {len(header)}")
        lines.append(f"id_like_columns: {id_like}")
        lines.append("columns:")
        for col in header:
            lines.append(f"  - {col}")
        lines.append("")

    out_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
