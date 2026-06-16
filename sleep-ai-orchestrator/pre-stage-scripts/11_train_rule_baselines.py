from __future__ import annotations

import csv
from pathlib import Path


TARGET_COL = "target_respiratory_burden_ge_5"
FEATURE_COL = "feature_n_oxygen_desaturation"


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


def safe_div(num: float, den: float) -> float:
    return num / den if den != 0 else 0.0


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


def select_best_threshold(train_rows: list[dict[str, str]]) -> tuple[int, dict[str, float]]:
    thresholds = sorted({parse_int(r.get(FEATURE_COL)) for r in train_rows})
    y_true = [parse_int(r.get(TARGET_COL)) for r in train_rows]

    best_threshold = 0
    best_metrics: dict[str, float] | None = None

    for t in thresholds:
        y_pred = [1 if parse_int(r.get(FEATURE_COL)) >= t else 0 for r in train_rows]
        metrics = compute_metrics(y_true, y_pred)

        if best_metrics is None:
            best_threshold = t
            best_metrics = metrics
            continue

        # Optimize balanced accuracy first; break ties with F1.
        if (
            metrics["balanced_accuracy"] > best_metrics["balanced_accuracy"]
            or (
                metrics["balanced_accuracy"] == best_metrics["balanced_accuracy"]
                and metrics["f1"] > best_metrics["f1"]
            )
        ):
            best_threshold = t
            best_metrics = metrics

    assert best_metrics is not None
    return best_threshold, best_metrics


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    cfg = load_paths_config(project_root / "configs" / "paths.yaml")

    outputs_dir = Path(cfg["outputs_dir"])
    input_csv = outputs_dir / "inventory" / "baseline_matrix_step10.csv"

    if not input_csv.exists():
        raise FileNotFoundError(f"Missing input: {input_csv}. Run step 10 first.")

    out_dir = outputs_dir / "baselines"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_file = out_dir / "baseline_metrics_step11.txt"
    preds_file = out_dir / "baseline_predictions_step11.csv"

    rows: list[dict[str, str]] = []
    with input_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    train_rows = [r for r in rows if (r.get("split") or "") == "train"]
    val_rows = [r for r in rows if (r.get("split") or "") == "val"]
    test_rows = [r for r in rows if (r.get("split") or "") == "test"]

    if not train_rows or not val_rows or not test_rows:
        raise ValueError("Expected train, val and test rows in split column.")

    y_train = [parse_int(r.get(TARGET_COL)) for r in train_rows]
    train_pos_rate = safe_div(sum(y_train), len(y_train))

    # Baseline A: majority-class classifier learned on train split.
    majority_class = 1 if train_pos_rate >= 0.5 else 0

    def predict_majority(input_rows: list[dict[str, str]]) -> list[int]:
        return [majority_class for _ in input_rows]

    # Baseline B: one-feature threshold model tuned on train.
    best_threshold, threshold_train_metrics = select_best_threshold(train_rows)

    def predict_threshold(input_rows: list[dict[str, str]]) -> list[int]:
        return [1 if parse_int(r.get(FEATURE_COL)) >= best_threshold else 0 for r in input_rows]

    y_val = [parse_int(r.get(TARGET_COL)) for r in val_rows]
    y_test = [parse_int(r.get(TARGET_COL)) for r in test_rows]

    maj_val = compute_metrics(y_val, predict_majority(val_rows))
    maj_test = compute_metrics(y_test, predict_majority(test_rows))

    thr_val = compute_metrics(y_val, predict_threshold(val_rows))
    thr_test = compute_metrics(y_test, predict_threshold(test_rows))

    lines = [
        f"input_csv: {input_csv}",
        f"train_rows: {len(train_rows)}",
        f"val_rows: {len(val_rows)}",
        f"test_rows: {len(test_rows)}",
        f"train_positive_rate: {train_pos_rate:.6f}",
        "",
        "baseline_A_majority:",
        f"  learned_majority_class: {majority_class}",
        f"  val_accuracy: {maj_val['accuracy']:.6f}",
        f"  val_balanced_accuracy: {maj_val['balanced_accuracy']:.6f}",
        f"  val_f1: {maj_val['f1']:.6f}",
        f"  val_tp_fp_tn_fn: {int(maj_val['tp'])},{int(maj_val['fp'])},{int(maj_val['tn'])},{int(maj_val['fn'])}",
        f"  test_accuracy: {maj_test['accuracy']:.6f}",
        f"  test_balanced_accuracy: {maj_test['balanced_accuracy']:.6f}",
        f"  test_f1: {maj_test['f1']:.6f}",
        f"  test_tp_fp_tn_fn: {int(maj_test['tp'])},{int(maj_test['fp'])},{int(maj_test['tn'])},{int(maj_test['fn'])}",
        "",
        "baseline_B_threshold_on_oxygen_desaturation:",
        f"  feature: {FEATURE_COL}",
        f"  selected_threshold_from_train: {best_threshold}",
        f"  train_balanced_accuracy: {threshold_train_metrics['balanced_accuracy']:.6f}",
        f"  train_f1: {threshold_train_metrics['f1']:.6f}",
        f"  val_accuracy: {thr_val['accuracy']:.6f}",
        f"  val_balanced_accuracy: {thr_val['balanced_accuracy']:.6f}",
        f"  val_f1: {thr_val['f1']:.6f}",
        f"  val_tp_fp_tn_fn: {int(thr_val['tp'])},{int(thr_val['fp'])},{int(thr_val['tn'])},{int(thr_val['fn'])}",
        f"  test_accuracy: {thr_test['accuracy']:.6f}",
        f"  test_balanced_accuracy: {thr_test['balanced_accuracy']:.6f}",
        f"  test_f1: {thr_test['f1']:.6f}",
        f"  test_tp_fp_tn_fn: {int(thr_test['tp'])},{int(thr_test['fp'])},{int(thr_test['tn'])},{int(thr_test['fn'])}",
    ]
    metrics_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with preds_file.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "record_id",
            "split",
            "y_true",
            "y_pred_majority",
            "y_pred_threshold_oxygen_desaturation",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in rows:
            y_true = parse_int(r.get(TARGET_COL))
            y_pred_maj = majority_class
            y_pred_thr = 1 if parse_int(r.get(FEATURE_COL)) >= best_threshold else 0
            writer.writerow(
                {
                    "record_id": (r.get("record_id") or "").strip(),
                    "split": (r.get("split") or "").strip(),
                    "y_true": y_true,
                    "y_pred_majority": y_pred_maj,
                    "y_pred_threshold_oxygen_desaturation": y_pred_thr,
                }
            )

    print(f"Wrote {metrics_file}")
    print(f"Wrote {preds_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
