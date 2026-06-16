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


def parse_float(value: str | None) -> float:
    if value is None:
        return 0.0
    text = value.strip()
    if not text:
        return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


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


def format_metrics(prefix: str, m: dict[str, float]) -> list[str]:
    return [
        f"  {prefix}_accuracy: {m['accuracy']:.6f}",
        f"  {prefix}_balanced_accuracy: {m['balanced_accuracy']:.6f}",
        f"  {prefix}_precision: {m['precision']:.6f}",
        f"  {prefix}_recall: {m['recall']:.6f}",
        f"  {prefix}_f1: {m['f1']:.6f}",
        f"  {prefix}_tp_fp_tn_fn: {int(m['tp'])},{int(m['fp'])},{int(m['tn'])},{int(m['fn'])}",
    ]


def main() -> int:
    try:
        import numpy as np
        from xgboost import XGBClassifier
    except Exception as exc:
        raise RuntimeError(
            "This step requires numpy and xgboost. Install with: python3 -m pip install numpy xgboost"
        ) from exc

    project_root = Path(__file__).resolve().parents[1]
    cfg = load_paths_config(project_root / "configs" / "paths.yaml")

    outputs_dir = Path(cfg["outputs_dir"])
    input_csv = outputs_dir / "inventory" / "baseline_matrix_step10.csv"

    if not input_csv.exists():
        raise FileNotFoundError(f"Missing input: {input_csv}. Run step 10 first.")

    out_dir = outputs_dir / "baselines"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_file = out_dir / "baseline_metrics_step17_xgboost.txt"
    preds_file = out_dir / "baseline_predictions_step17_xgboost.csv"

    feature_cols = [
        "feature_event_rows_total",
        "feature_recording_hours_est",
        "feature_n_oxygen_desaturation",
        "feature_n_eeg_arousal",
    ]
    target_col = "target_respiratory_burden_ge_5"

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

    def to_xy(input_rows: list[dict[str, str]]):
        x = []
        y = []
        for r in input_rows:
            x.append([parse_float(r.get(c)) for c in feature_cols])
            y.append(parse_int(r.get(target_col)))
        return np.asarray(x, dtype=float), np.asarray(y, dtype=int)

    x_train, y_train = to_xy(train_rows)
    x_val, y_val = to_xy(val_rows)
    x_test, y_test = to_xy(test_rows)

    positives = int((y_train == 1).sum())
    negatives = int((y_train == 0).sum())
    scale_pos_weight = safe_div(negatives, positives) if positives > 0 else 1.0

    model = XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
    )
    model.fit(x_train, y_train)

    val_proba = model.predict_proba(x_val)[:, 1]
    test_proba = model.predict_proba(x_test)[:, 1]

    best_t = 0.5
    best_val_bacc = -1.0
    best_val_f1 = -1.0

    for i in range(5, 96):
        t = i / 100.0
        val_pred = (val_proba >= t).astype(int)
        m = compute_metrics(y_val.tolist(), val_pred.tolist())
        if m["balanced_accuracy"] > best_val_bacc or (
            m["balanced_accuracy"] == best_val_bacc and m["f1"] > best_val_f1
        ):
            best_t = t
            best_val_bacc = m["balanced_accuracy"]
            best_val_f1 = m["f1"]

    val_pred_final = (val_proba >= best_t).astype(int)
    test_pred_final = (test_proba >= best_t).astype(int)

    val_metrics = compute_metrics(y_val.tolist(), val_pred_final.tolist())
    test_metrics = compute_metrics(y_test.tolist(), test_pred_final.tolist())

    lines = [
        f"input_csv: {input_csv}",
        f"train_rows: {len(train_rows)}",
        f"val_rows: {len(val_rows)}",
        f"test_rows: {len(test_rows)}",
        f"features: {feature_cols}",
        f"target: {target_col}",
        f"scale_pos_weight: {scale_pos_weight:.6f}",
        f"threshold_selected_on_val: {best_t:.2f}",
        "",
        "validation_metrics:",
    ]
    lines.extend(format_metrics("val", val_metrics))
    lines.append("")
    lines.append("test_metrics:")
    lines.extend(format_metrics("test", test_metrics))
    lines.append("")
    lines.append("feature_importance_gain:")

    booster = model.get_booster()
    gain_map = booster.get_score(importance_type="gain")
    for i, name in enumerate(feature_cols):
        gain = float(gain_map.get(f"f{i}", 0.0))
        lines.append(f"  {name}: {gain:.6f}")

    metrics_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with preds_file.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "record_id",
            "split",
            "y_true",
            "y_proba_xgboost",
            "y_pred_xgboost",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for split_name, split_rows, x_s, y_true_arr in [
            ("train", train_rows, x_train, y_train),
            ("val", val_rows, x_val, y_val),
            ("test", test_rows, x_test, y_test),
        ]:
            proba = model.predict_proba(x_s)[:, 1]
            pred = (proba >= best_t).astype(int)
            for idx, r in enumerate(split_rows):
                writer.writerow(
                    {
                        "record_id": (r.get("record_id") or "").strip(),
                        "split": split_name,
                        "y_true": int(y_true_arr[idx]),
                        "y_proba_xgboost": f"{float(proba[idx]):.6f}",
                        "y_pred_xgboost": int(pred[idx]),
                    }
                )

    print(f"Wrote {metrics_file}")
    print(f"Wrote {preds_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
