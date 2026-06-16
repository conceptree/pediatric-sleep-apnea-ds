from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
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


def list_dirs_maxdepth_2(root: Path) -> list[Path]:
    if not root.exists() or not root.is_dir():
        return []

    results = [root]
    for child in sorted(root.iterdir()):
        if child.is_dir():
            results.append(child)
            for grandchild in sorted(child.iterdir()):
                if grandchild.is_dir():
                    results.append(grandchild)
    return results


def top_extensions(roots: list[Path], top_n: int = 20) -> list[tuple[str, int]]:
    counts: Counter[str] = Counter()

    for root in roots:
        if not root.exists() or not root.is_dir():
            continue

        for p in root.rglob("*"):
            if not p.is_file():
                continue
            suffix = p.suffix.lower().lstrip(".")
            counts[suffix if suffix else "(no_ext)"] += 1

    return sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:top_n]


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "configs" / "paths.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")

    cfg = load_paths_config(config_path)
    sleep_data_dir = Path(cfg["sleep_data_dir"])
    health_data_dir = Path(cfg["health_data_dir"])
    outputs_dir = Path(cfg["outputs_dir"])

    inventory_dir = outputs_dir / "inventory"
    inventory_dir.mkdir(parents=True, exist_ok=True)
    output_file = inventory_dir / "dataset_inventory_step1.txt"

    sleep_dirs = list_dirs_maxdepth_2(sleep_data_dir)
    health_dirs = list_dirs_maxdepth_2(health_data_dir)
    ext_rows = top_extensions([sleep_data_dir, health_data_dir], top_n=20)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    lines: list[str] = []
    lines.append(f"timestamp: {ts}")
    lines.append("")
    lines.append("== Sleep_Data: pastas (maxdepth 2) ==")
    lines.extend(str(p) for p in sleep_dirs)
    lines.append("")
    lines.append("== Health_Data: pastas (maxdepth 2) ==")
    lines.extend(str(p) for p in health_dirs)
    lines.append("")
    lines.append("== Top 20 extensoes de ficheiro ==")
    lines.extend(f"{count:4d} {ext}" for ext, count in ext_rows)
    lines.append("")

    output_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
