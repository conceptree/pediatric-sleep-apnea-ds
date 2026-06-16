from __future__ import annotations

from collections import Counter
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
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        data[key] = value

    return data


def analyze_root(root: Path) -> dict[str, object]:
    files = [p for p in root.iterdir() if p.is_file()]
    exts = Counter(p.suffix.lower().lstrip(".") or "(no_ext)" for p in files)

    stems_by_ext: dict[str, set[str]] = {}
    for p in files:
        ext = p.suffix.lower().lstrip(".") or "(no_ext)"
        stems_by_ext.setdefault(ext, set()).add(p.stem)

    edf = stems_by_ext.get("edf", set())
    tsv = stems_by_ext.get("tsv", set())
    paired = edf & tsv
    only_edf = sorted(edf - tsv)
    only_tsv = sorted(tsv - edf)

    return {
        "root": str(root),
        "total_files": len(files),
        "exts": exts,
        "edf_count": len(edf),
        "tsv_count": len(tsv),
        "paired_count": len(paired),
        "only_edf_count": len(only_edf),
        "only_tsv_count": len(only_tsv),
        "only_edf_sample": only_edf[:10],
        "only_tsv_sample": only_tsv[:10],
    }


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "configs" / "paths.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")

    cfg = load_paths_config(config_path)
    sleep_data_dir = Path(cfg["sleep_data_dir"])
    health_data_dir = Path(cfg["health_data_dir"])
    outputs_dir = Path(cfg["outputs_dir"])

    results = [analyze_root(sleep_data_dir), analyze_root(health_data_dir)]

    out_file = outputs_dir / "inventory" / "pairing_check_step2.txt"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    for r in results:
        lines.append(f"== {r['root']} ==")
        lines.append(f"total_files: {r['total_files']}")
        lines.append("extensions:")

        exts = r["exts"]
        assert isinstance(exts, Counter)
        for ext, count in sorted(exts.items(), key=lambda x: (-x[1], x[0])):
            lines.append(f"  {ext}: {count}")

        lines.append(f"edf_count: {r['edf_count']}")
        lines.append(f"tsv_count: {r['tsv_count']}")
        lines.append(f"paired_count: {r['paired_count']}")
        lines.append(f"only_edf_count: {r['only_edf_count']}")
        lines.append(f"only_tsv_count: {r['only_tsv_count']}")
        lines.append(f"only_edf_sample: {r['only_edf_sample']}")
        lines.append(f"only_tsv_sample: {r['only_tsv_sample']}")
        lines.append("")

    out_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
