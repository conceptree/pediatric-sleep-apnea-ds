# MVP v1 Baseline Summary

## Scope
Clinical decision support for pediatric sleep apnea research (not automated diagnosis).

## Decision
selected_baseline: xgboost
Selection rule: primary metric = test balanced accuracy; tie-breaker = test F1.

## Final Baseline Metrics
Selected model (xgboost):
- balanced_accuracy: 0.839669
- f1: 0.481481
- tp_fp_tn_fn: 26,53,191,3

Comparison references:
- XGBoost: balanced_accuracy: 0.839669; f1: 0.481481
- Random Forest: balanced_accuracy: 0.825325; f1: 0.452174
- Logistic Regression: balanced_accuracy: 0.765263; f1: 0.521739
- Threshold oxygen desaturation: balanced_accuracy: 0.819531; f1: 0.484848

## Guardrails Confirmed
- Patient-grouped split to reduce leakage.
- Leakage-prone features excluded from baseline matrix.
- Validation and held-out test metrics reported.

Guardrails source snapshot:
```text
input_csv: /Volumes/CORSAIR/tese/sleep-ai-orchestrator/outputs/inventory/record_event_features_step9.csv
output_csv: /Volumes/CORSAIR/tese/sleep-ai-orchestrator/outputs/inventory/baseline_matrix_step10.csv
rows_output: 1833
split_train: 1291
split_val: 269
split_test: 273
target_positive: 187
target_negative: 1646

target_column: label_respiratory_burden_ge_5
safe_feature_columns: ['event_rows_total', 'recording_hours_est', 'n_oxygen_desaturation', 'n_eeg_arousal']

excluded_leakage_prone_columns:
  - n_obstructive_apnea
  - n_obstructive_hypopnea
  - n_hypopnea_any
  - n_apnea_any
  - n_central_apnea
  - n_mixed_apnea
  - n_rera
  - n_obstructive_events
  - n_respiratory_events
  - obstructive_event_rate_per_hour
  - respiratory_event_rate_per_hour
  - label_obstructive_any
```

## Artifacts
- Comparison: /Volumes/CORSAIR/tese/sleep-ai-orchestrator/outputs/baselines/baseline_comparison_step18_all_models_with_xgboost.txt
- XGBoost metrics: /Volumes/CORSAIR/tese/sleep-ai-orchestrator/outputs/baselines/baseline_metrics_step17_xgboost.txt
- XGBoost predictions: /Volumes/CORSAIR/tese/sleep-ai-orchestrator/outputs/baselines/baseline_predictions_step17_xgboost.csv
- Random Forest metrics: /Volumes/CORSAIR/tese/sleep-ai-orchestrator/outputs/baselines/baseline_metrics_step15_rf.txt
- Baseline matrix: /Volumes/CORSAIR/tese/sleep-ai-orchestrator/outputs/inventory/baseline_matrix_step10.csv

## Next Increment
Add probability calibration and threshold policy analysis for the selected xgboost while keeping the same data contract and split strategy.
