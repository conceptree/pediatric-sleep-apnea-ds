from __future__ import annotations

from pathlib import Path


def load_paths_config(config_path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip().strip('"').strip("'")
    return data


def parse_selected_baseline(comparison_file: Path) -> str:
    selected = "unknown"
    for line in comparison_file.read_text(encoding="utf-8").splitlines():
        if line.startswith("selected_baseline:"):
            selected = line.split(":", 1)[1].strip()
            break
    return selected


def extract_section_block(text: str, section_header: str) -> str:
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.strip() == section_header:
            start = i
            break
    if start is None:
        return ""

    end = len(lines)
    for j in range(start + 1, len(lines)):
        if lines[j].startswith("== ") and lines[j] != section_header:
            end = j
            break

    return "\n".join(lines[start:end]).strip()


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    cfg = load_paths_config(project_root / "configs" / "paths.yaml")
    outputs_dir = Path(cfg["outputs_dir"])

    comparison_file = outputs_dir / "baselines" / "baseline_comparison_step13.txt"
    if not comparison_file.exists():
        raise FileNotFoundError(f"Missing comparison file: {comparison_file}. Run step 13 first.")

    selected = parse_selected_baseline(comparison_file)
    text = comparison_file.read_text(encoding="utf-8")
    val_block = extract_section_block(text, "== VAL ==")
    test_block = extract_section_block(text, "== TEST ==")

    out_file = outputs_dir / "baselines" / "baseline_decision_step14.md"

    md_lines = [
        "# MVP Baseline Decision (Step 14)",
        "",
        "## Scope",
        "This project is framed as clinical decision support for pediatric sleep apnea research, not automated diagnosis.",
        "",
        "## Selected Baseline",
        f"Selected baseline: {selected}",
        "Selection rule: primary metric = test_balanced_accuracy, tie-breaker = test_f1.",
        "",
        "## Validation Snapshot",
        "```text",
        val_block,
        "```",
        "",
        "## Test Snapshot",
        "```text",
        test_block,
        "```",
        "",
        "## Interpretation",
        "The selected baseline prioritizes sensitivity/specificity balance under class imbalance, which is appropriate for screening-oriented clinical decision support research.",
        "",
        "## Guardrails",
        "- Data split is patient-grouped to reduce leakage.",
        "- Baseline matrix excludes leakage-prone respiratory burden features.",
        "- Metrics are reported on validation and held-out test splits.",
        "",
        "## Immediate Next Step",
        "Run the same evaluation protocol with a tree-based classical model (for example, Random Forest with class balancing) and compare against this selected baseline without changing the data contract.",
        "",
    ]

    out_file.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Wrote {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
