from __future__ import annotations

import csv
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


def safe_div(num: float, den: float) -> float:
    return num / den if den != 0 else 0.0


def confusion_counts(y_true: list[int], y_pred: list[int]) -> tuple[int, int, int, int]:
    tp = fp = tn = fn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 1:
            tp += 1
        elif yt == 0 and yp == 1:
            fp += 1
        elif yt == 0 and yp == 0:
            tn += 1
        elif yt == 1 and yp == 0:
            fn += 1
    return tp, fp, tn, fn


def compute_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    tp, fp, tn, fn = confusion_counts(y_true, y_pred)
    total = tp + fp + tn + fn

    accuracy = safe_div(tp + tn, total)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    f1 = safe_div(2 * precision * recall, precision + recall)
    balanced_accuracy = (recall + specificity) / 2.0

    return {
        "n": float(total),
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
    }


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    cfg = load_paths_config(project_root / "configs" / "paths.yaml")
    outputs_dir = Path(cfg["outputs_dir"])

    step11_preds = outputs_dir / "baselines" / "baseline_predictions_step11.csv"
    step12_preds = outputs_dir / "baselines" / "baseline_predictions_step12_logreg.csv"
    step15_preds = outputs_dir / "baselines" / "baseline_predictions_step15_rf.csv"

    if not step11_preds.exists() or not step12_preds.exists() or not step15_preds.exists():
        raise FileNotFoundError("Missing predictions from step 11, step 12 or step 15.")

    merged: dict[tuple[str, str], dict[str, int | str]] = {}

    with step11_preds.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = ((row.get("record_id") or "").strip(), (row.get("split") or "").strip())
            merged[key] = {
                "record_id": key[0],
                "split": key[1],
                "y_true": parse_int(row.get("y_true")),
                "pred_majority": parse_int(row.get("y_pred_majority")),
                "pred_threshold": parse_int(row.get("y_pred_threshold_oxygen_desaturation")),
            }

    with step12_preds.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = ((row.get("record_id") or "").strip(), (row.get("split") or "").strip())
            if key in merged:
                merged[key]["pred_logreg"] = parse_int(row.get("y_pred_logreg"))

    with step15_preds.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = ((row.get("record_id") or "").strip(), (row.get("split") or "").strip())
            if key in merged:
                merged[key]["pred_rf"] = parse_int(row.get("y_pred_rf"))

    rows = list(merged.values())

    models = [
        ("majority", "pred_majority"),
        ("threshold_oxygen_desat", "pred_threshold"),
        ("logreg", "pred_logreg"),
        ("random_forest", "pred_rf"),
    ]

    splits = ["val", "test"]
    result: dict[tuple[str, str], dict[str, float]] = {}

    for split in splits:
        split_rows = [r for r in rows if r.get("split") == split]
        y_true = [int(r["y_true"]) for r in split_rows]
        for model_name, pred_col in models:
            y_pred = [int(r.get(pred_col, 0)) for r in split_rows]
            result[(split, model_name)] = compute_metrics(y_true, y_pred)

    out_file = outputs_dir / "baselines" / "baseline_comparison_step16_all_models.txt"

    lines: list[str] = []
    lines.append(f"step11_predictions: {step11_preds}")
    lines.append(f"step12_predictions: {step12_preds}")
    lines.append(f"step15_predictions: {step15_preds}")
    lines.append("")

    for split in splits:
        lines.append(f"== {split.upper()} ==")
        for model_name, _ in models:
            m = result[(split, model_name)]
            lines.append(f"model: {model_name}")
            lines.append(f"  accuracy: {m['accuracy']:.6f}")
            lines.append(f"  balanced_accuracy: {m['balanced_accuracy']:.6f}")
            lines.append(f"  precision: {m['precision']:.6f}")
            lines.append(f"  recall: {m['recall']:.6f}")
            lines.append(f"  f1: {m['f1']:.6f}")
            lines.append(f"  tp_fp_tn_fn: {int(m['tp'])},{int(m['fp'])},{int(m['tn'])},{int(m['fn'])}")
        lines.append("")

    test_rank = sorted(
        [
            (
                model_name,
                result[("test", model_name)]["balanced_accuracy"],
                result[("test", model_name)]["f1"],
            )
            for model_name, _ in models
        ],
        key=lambda x: (x[1], x[2]),
        reverse=True,
    )

    winner = test_rank[0][0]
    lines.append("selection_rule:")
    lines.append("  primary_metric: test_balanced_accuracy")
    lines.append("  tie_breaker: test_f1")
    lines.append(f"selected_baseline: {winner}")

    out_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
