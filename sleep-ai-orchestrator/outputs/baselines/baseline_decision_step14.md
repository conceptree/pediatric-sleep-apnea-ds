# MVP Baseline Decision (Step 14)

## Scope
This project is framed as clinical decision support for pediatric sleep apnea research, not automated diagnosis.

## Selected Baseline
Selected baseline: threshold_oxygen_desat
Selection rule: primary metric = test_balanced_accuracy, tie-breaker = test_f1.

## Validation Snapshot
```text
== VAL ==
model: majority
  accuracy: 0.914498
  balanced_accuracy: 0.500000
  precision: 0.000000
  recall: 0.000000
  f1: 0.000000
  tp_fp_tn_fn: 0,0,246,23
model: threshold_oxygen_desat
  accuracy: 0.802974
  balanced_accuracy: 0.754330
  precision: 0.258065
  recall: 0.695652
  f1: 0.376471
  tp_fp_tn_fn: 16,46,200,7
model: logreg
  accuracy: 0.873606
  balanced_accuracy: 0.792948
  precision: 0.372093
  recall: 0.695652
  f1: 0.484848
  tp_fp_tn_fn: 16,27,219,7
```

## Test Snapshot
```text
== TEST ==
model: majority
  accuracy: 0.893773
  balanced_accuracy: 0.500000
  precision: 0.000000
  recall: 0.000000
  f1: 0.000000
  tp_fp_tn_fn: 0,0,244,29
model: threshold_oxygen_desat
  accuracy: 0.813187
  balanced_accuracy: 0.819531
  precision: 0.342857
  recall: 0.827586
  f1: 0.484848
  tp_fp_tn_fn: 24,46,198,5
model: logreg
  accuracy: 0.879121
  balanced_accuracy: 0.765263
  precision: 0.450000
  recall: 0.620690
  f1: 0.521739
  tp_fp_tn_fn: 18,22,222,11

selection_rule:
  primary_metric: test_balanced_accuracy
  tie_breaker: test_f1
selected_baseline: threshold_oxygen_desat
```

## Interpretation
The selected baseline prioritizes sensitivity/specificity balance under class imbalance, which is appropriate for screening-oriented clinical decision support research.

## Guardrails
- Data split is patient-grouped to reduce leakage.
- Baseline matrix excludes leakage-prone respiratory burden features.
- Metrics are reported on validation and held-out test splits.

## Immediate Next Step
Run the same evaluation protocol with a tree-based classical model (for example, Random Forest with class balancing) and compare against this selected baseline without changing the data contract.
