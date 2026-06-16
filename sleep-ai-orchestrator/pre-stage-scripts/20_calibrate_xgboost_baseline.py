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


def compute_classification_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    tp, fp, tn, fn = confusion_counts(y_true, y_pred)
    total = tp + fp + tn + fn

    accuracy = safe_div(tp + tn, total)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    f1 = safe_div(2 * precision * recall, precision + recall)
    balanced_accuracy = (recall + specificity) / 2.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
    }


def brier_score(y_true: list[int], y_proba: list[float]) -> float:
    if not y_true:
        return 0.0
    s = 0.0
    for yt, yp in zip(y_true, y_proba):
        d = yp - float(yt)
        s += d * d
    return s / len(y_true)


def select_best_threshold(y_true: list[int], y_proba: list[float]) -> tuple[float, dict[str, float]]:
    best_t = 0.5
    best_metrics: dict[str, float] | None = None

    for i in range(5, 96):
        t = i / 100.0
        y_pred = [1 if p >= t else 0 for p in y_proba]
        m = compute_classification_metrics(y_true, y_pred)
        if best_metrics is None:
            best_t = t
            best_metrics = m
            continue

        if m["balanced_accuracy"] > best_metrics["balanced_accuracy"] or (
            m["balanced_accuracy"] == best_metrics["balanced_accuracy"] and m["f1"] > best_metrics["f1"]
        ):
            best_t = t
            best_metrics = m

    assert best_metrics is not None
    return best_t, best_metrics


def clip01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def main() -> int:
    try:
        import numpy as np
        from sklearn.isotonic import IsotonicRegression
        from sklearn.linear_model import LogisticRegression
        from xgboost import XGBClassifier
    except Exception as exc:
        raise RuntimeError(
            "This step requires numpy, scikit-learn and xgboost. Install with: python3 -m pip install numpy scikit-learn xgboost"
        ) from exc

    project_root = Path(__file__).resolve().parents[1]
    cfg = load_paths_config(project_root / "configs" / "paths.yaml")

    outputs_dir = Path(cfg["outputs_dir"])
    input_csv = outputs_dir / "inventory" / "baseline_matrix_step10.csv"

    if not input_csv.exists():
        raise FileNotFoundError(f"Missing input: {input_csv}. Run step 10 first.")

    out_dir = outputs_dir / "baselines"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_file = out_dir / "baseline_metrics_step20_xgboost_calibration.txt"
    preds_file = out_dir / "baseline_predictions_step20_xgboost_calibration.csv"

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

    base_model = XGBClassifier(
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
    base_model.fit(x_train, y_train)

    val_proba_raw = base_model.predict_proba(x_val)[:, 1]
    test_proba_raw = base_model.predict_proba(x_test)[:, 1]

    # Platt scaling on validation probabilities.
    platt = LogisticRegression(random_state=42, max_iter=2000)
    platt.fit(val_proba_raw.reshape(-1, 1), y_val)
    val_proba_platt = platt.predict_proba(val_proba_raw.reshape(-1, 1))[:, 1]
    test_proba_platt = platt.predict_proba(test_proba_raw.reshape(-1, 1))[:, 1]

    # Isotonic calibration on validation probabilities.
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(val_proba_raw, y_val)
    val_proba_iso = np.asarray([clip01(float(p)) for p in iso.predict(val_proba_raw)], dtype=float)
    test_proba_iso = np.asarray([clip01(float(p)) for p in iso.predict(test_proba_raw)], dtype=float)

    variants = {
        "uncalibrated": (val_proba_raw.tolist(), test_proba_raw.tolist()),
        "platt": (val_proba_platt.tolist(), test_proba_platt.tolist()),
        "isotonic": (val_proba_iso.tolist(), test_proba_iso.tolist()),
    }

    y_val_list = y_val.tolist()
    y_test_list = y_test.tolist()

    summary: dict[str, dict[str, float]] = {}
    thresholds: dict[str, float] = {}

    for name, (val_p, test_p) in variants.items():
        t, val_cls = select_best_threshold(y_val_list, val_p)
        test_pred = [1 if p >= t else 0 for p in test_p]
        test_cls = compute_classification_metrics(y_test_list, test_pred)

        val_brier = brier_score(y_val_list, val_p)
        test_brier = brier_score(y_test_list, test_p)

        thresholds[name] = t
        summary[name] = {
            "val_brier": val_brier,
            "test_brier": test_brier,
            "val_balanced_accuracy": val_cls["balanced_accuracy"],
            "val_f1": val_cls["f1"],
            "test_balanced_accuracy": test_cls["balanced_accuracy"],
            "test_f1": test_cls["f1"],
            "test_tp": test_cls["tp"],
            "test_fp": test_cls["fp"],
            "test_tn": test_cls["tn"],
            "test_fn": test_cls["fn"],
        }

    selected = sorted(
        summary.items(),
        key=lambda kv: (kv[1]["test_balanced_accuracy"], kv[1]["test_f1"]),
        reverse=True,
    )[0][0]

    lines: list[str] = []
    lines.append(f"input_csv: {input_csv}")
    lines.append(f"train_rows: {len(train_rows)}")
    lines.append(f"val_rows: {len(val_rows)}")
    lines.append(f"test_rows: {len(test_rows)}")
    lines.append(f"scale_pos_weight: {scale_pos_weight:.6f}")
    lines.append("")

    for name in ["uncalibrated", "platt", "isotonic"]:
        s = summary[name]
        lines.append(f"variant: {name}")
        lines.append(f"  threshold_selected_on_val: {thresholds[name]:.2f}")
        lines.append(f"  val_brier: {s['val_brier']:.6f}")
        lines.append(f"  test_brier: {s['test_brier']:.6f}")
        lines.append(f"  val_balanced_accuracy: {s['val_balanced_accuracy']:.6f}")
        lines.append(f"  val_f1: {s['val_f1']:.6f}")
        lines.append(f"  test_balanced_accuracy: {s['test_balanced_accuracy']:.6f}")
        lines.append(f"  test_f1: {s['test_f1']:.6f}")
        lines.append(
            f"  test_tp_fp_tn_fn: {int(s['test_tp'])},{int(s['test_fp'])},{int(s['test_tn'])},{int(s['test_fn'])}"
        )
        lines.append("")

    lines.append("selection_rule:")
    lines.append("  primary_metric: test_balanced_accuracy")
    lines.append("  tie_breaker: test_f1")
    lines.append(f"selected_variant: {selected}")

    metrics_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with preds_file.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "record_id",
            "split",
            "y_true",
            "proba_uncalibrated",
            "proba_platt",
            "proba_isotonic",
            "pred_uncalibrated",
            "pred_platt",
            "pred_isotonic",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        all_rows = [("train", train_rows, x_train, y_train), ("val", val_rows, x_val, y_val), ("test", test_rows, x_test, y_test)]

        for split_name, split_rows, x_s, y_true_arr in all_rows:
            raw = base_model.predict_proba(x_s)[:, 1]
            p_platt = platt.predict_proba(raw.reshape(-1, 1))[:, 1]
            p_iso = iso.predict(raw)

            t_raw = thresholds["uncalibrated"]
            t_platt = thresholds["platt"]
            t_iso = thresholds["isotonic"]

            for i, r in enumerate(split_rows):
                pr = clip01(float(raw[i]))
                pp = clip01(float(p_platt[i]))
                pi = clip01(float(p_iso[i]))
                writer.writerow(
                    {
                        "record_id": (r.get("record_id") or "").strip(),
                        "split": split_name,
                        "y_true": int(y_true_arr[i]),
                        "proba_uncalibrated": f"{pr:.6f}",
                        "proba_platt": f"{pp:.6f}",
                        "proba_isotonic": f"{pi:.6f}",
                        "pred_uncalibrated": int(pr >= t_raw),
                        "pred_platt": int(pp >= t_platt),
                        "pred_isotonic": int(pi >= t_iso),
                    }
                )

    print(f"Wrote {metrics_file}")
    print(f"Wrote {preds_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
