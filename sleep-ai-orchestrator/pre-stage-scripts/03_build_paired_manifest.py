from __future__ import annotations

from pathlib import Path


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


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    cfg = load_paths_config(project_root / "configs" / "paths.yaml")

    sleep_data_dir = Path(cfg["sleep_data_dir"])
    outputs_dir = Path(cfg["outputs_dir"])

    edf_map: dict[str, Path] = {}
    tsv_map: dict[str, Path] = {}

    for p in sleep_data_dir.iterdir():
        if not p.is_file():
            continue

        ext = p.suffix.lower()
        if ext == ".edf":
            edf_map[p.stem] = p
        elif ext == ".tsv":
            tsv_map[p.stem] = p

    paired_ids = sorted(set(edf_map) & set(tsv_map))
    only_edf = sorted(set(edf_map) - set(tsv_map))
    only_tsv = sorted(set(tsv_map) - set(edf_map))

    inventory_dir = outputs_dir / "inventory"
    inventory_dir.mkdir(parents=True, exist_ok=True)

    manifest_csv = inventory_dir / "paired_sleep_records_step3.csv"
    qc_txt = inventory_dir / "paired_sleep_records_step3_qc.txt"

    lines = ["record_id,edf_path,tsv_path"]
    for rid in paired_ids:
        lines.append(f"{rid},{edf_map[rid]},{tsv_map[rid]}")
    manifest_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")

    qc_lines = [
        f"sleep_data_dir: {sleep_data_dir}",
        f"paired_count: {len(paired_ids)}",
        f"only_edf_count: {len(only_edf)}",
        f"only_tsv_count: {len(only_tsv)}",
        f"only_edf_sample: {only_edf[:10]}",
        f"only_tsv_sample: {only_tsv[:10]}",
    ]
    qc_txt.write_text("\n".join(qc_lines) + "\n", encoding="utf-8")

    print(f"Wrote {manifest_csv}")
    print(f"Wrote {qc_txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
