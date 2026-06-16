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


def read_text(path: Path) -> str:
    if not path.exists():
        return f"MISSING: {path}"
    return path.read_text(encoding="utf-8")


def find_line(text: str, prefix: str) -> str:
    for line in text.splitlines():
        if line.strip().startswith(prefix):
            return line.strip()
    return f"{prefix} <not found>"


def value_for_line(text: str, prefix: str) -> str:
    line = find_line(text, prefix)
    if ":" in line:
        return line.split(":", 1)[1].strip()
    return ""


def extract_test_metric_for_model(comparison_text: str, model_name: str, metric_name: str) -> str:
    lines = comparison_text.splitlines()
    in_test = False
    in_model = False

    for line in lines:
        stripped = line.strip()

        if stripped == "== TEST ==":
            in_test = True
            in_model = False
            continue

        if stripped.startswith("== ") and stripped != "== TEST ==" and in_test:
            break

        if not in_test:
            continue

        if stripped == f"model: {model_name}":
            in_model = True
            continue

        if stripped.startswith("model: ") and in_model:
            in_model = False

        if in_model and stripped.startswith(f"{metric_name}:"):
            return stripped

    return f"{metric_name}: <not found>"


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    cfg = load_paths_config(project_root / "configs" / "paths.yaml")
    outputs_dir = Path(cfg["outputs_dir"])

    step18_file = outputs_dir / "baselines" / "baseline_comparison_step18_all_models_with_xgboost.txt"
    step17_file = outputs_dir / "baselines" / "baseline_metrics_step17_xgboost.txt"
    step15_file = outputs_dir / "baselines" / "baseline_metrics_step15_rf.txt"
    step12_file = outputs_dir / "baselines" / "baseline_metrics_step12_logreg.txt"
    step11_file = outputs_dir / "baselines" / "baseline_metrics_step11.txt"
    step10_guardrails = outputs_dir / "inventory" / "baseline_matrix_step10_guardrails.txt"

    t18 = read_text(step18_file)
    t17 = read_text(step17_file)
    t10 = read_text(step10_guardrails)

    selected_baseline = find_line(t18, "selected_baseline:")
    selected_baseline_name = value_for_line(t18, "selected_baseline:")

    xgb_test_bacc = extract_test_metric_for_model(t18, "xgboost", "balanced_accuracy")
    xgb_test_f1 = extract_test_metric_for_model(t18, "xgboost", "f1")
    xgb_test_conf = extract_test_metric_for_model(t18, "xgboost", "tp_fp_tn_fn")

    rf_test_bacc = extract_test_metric_for_model(t18, "random_forest", "balanced_accuracy")
    rf_test_f1 = extract_test_metric_for_model(t18, "random_forest", "f1")

    lr_test_bacc = extract_test_metric_for_model(t18, "logreg", "balanced_accuracy")
    lr_test_f1 = extract_test_metric_for_model(t18, "logreg", "f1")

    thr_test_bacc = extract_test_metric_for_model(t18, "threshold_oxygen_desat", "balanced_accuracy")
    thr_test_f1 = extract_test_metric_for_model(t18, "threshold_oxygen_desat", "f1")

    out_file = outputs_dir / "baselines" / "mvp_v1_summary_step19.md"

    lines = [
        "# MVP v1 Baseline Summary",
        "",
        "## Scope",
        "Clinical decision support for pediatric sleep apnea research (not automated diagnosis).",
        "",
        "## Decision",
        selected_baseline,
        "Selection rule: primary metric = test balanced accuracy; tie-breaker = test F1.",
        "",
        "## Final Baseline Metrics",
        f"Selected model ({selected_baseline_name}):",
        f"- {xgb_test_bacc if selected_baseline_name == 'xgboost' else '<see selected model metrics file>'}",
        f"- {xgb_test_f1 if selected_baseline_name == 'xgboost' else '<see selected model metrics file>'}",
        f"- {xgb_test_conf if selected_baseline_name == 'xgboost' else '<see selected model metrics file>'}",
        "",
        "Comparison references:",
        f"- XGBoost: {xgb_test_bacc}; {xgb_test_f1}",
        f"- Random Forest: {rf_test_bacc}; {rf_test_f1}",
        f"- Logistic Regression: {lr_test_bacc}; {lr_test_f1}",
        f"- Threshold oxygen desaturation: {thr_test_bacc}; {thr_test_f1}",
        "",
        "## Guardrails Confirmed",
        "- Patient-grouped split to reduce leakage.",
        "- Leakage-prone features excluded from baseline matrix.",
        "- Validation and held-out test metrics reported.",
        "",
        "Guardrails source snapshot:",
        "```text",
        t10.strip(),
        "```",
        "",
        "## Artifacts",
        f"- Comparison: {step18_file}",
        f"- XGBoost metrics: {step17_file}",
        f"- XGBoost predictions: {outputs_dir / 'baselines' / 'baseline_predictions_step17_xgboost.csv'}",
        f"- Random Forest metrics: {step15_file}",
        f"- Baseline matrix: {outputs_dir / 'inventory' / 'baseline_matrix_step10.csv'}",
        "",
        "## Next Increment",
        f"Add probability calibration and threshold policy analysis for the selected {selected_baseline_name} while keeping the same data contract and split strategy.",
        "",
    ]

    out_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
