#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
XGBoost + GroupKFold (por paciente) + threshold tuning (Youden).
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    classification_report, roc_auc_score, accuracy_score,
    precision_recall_fscore_support, roc_curve
)
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
import joblib

# --------------------
# Paths
# --------------------
BASE = Path("/Users/nunorodrigues/dev/tese/datasets/splits")
FEATS = BASE / "features.parquet"
MODELS = BASE / "models"
OUTDIR = BASE / "reports"
MODELS.mkdir(exist_ok=True, parents=True)
OUTDIR.mkdir(exist_ok=True, parents=True)

print(f"📦 Lendo features: {FEATS}")
df = pd.read_parquet(FEATS)

# --------------------
# Preparação X / y
# --------------------
meta_cols = ["subset", "label", "stem", "error"]
X = df.drop(columns=[c for c in meta_cols if c in df.columns], errors="ignore")
y = df["label"].astype(int)

# Patient ID = prefixo antes do "_"
df["patient_id"] = df["stem"].str.split("_").str[0]

# Remover colunas com >10% de NaN
valid_mask = X.isna().mean() < 0.10
X = X.loc[:, valid_mask].copy()
feat_names = X.columns.tolist()

is_trainval = df["subset"].isin(["train", "val"])
is_test = df["subset"] == "test"

X_trainval = X[is_trainval].reset_index(drop=True)
y_trainval = y[is_trainval].reset_index(drop=True)
groups_trainval = df.loc[is_trainval, "patient_id"].reset_index(drop=True)

X_test = X[is_test].reset_index(drop=True)
y_test = y[is_test].reset_index(drop=True)

print(
    f"Registos: train+val={len(y_trainval)} | test={len(y_test)} | "
    f"features usadas={X_trainval.shape[1]}"
)

# --------------------
# Config XGBoost
# --------------------
base_params = dict(
    n_estimators=600,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=2,
    reg_lambda=1.5,
    reg_alpha=0.1,
    objective="binary:logistic",
    eval_metric="auc",
    random_state=42,
    n_jobs=4,
)

# --------------------
# GroupKFold CV
# --------------------
gkf = GroupKFold(n_splits=5)
fold_metrics = []
thresholds = []

for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_trainval, y_trainval, groups=groups_trainval), 1):
    X_tr, X_va = X_trainval.iloc[tr_idx], X_trainval.iloc[va_idx]
    y_tr, y_va = y_trainval.iloc[tr_idx], y_trainval.iloc[va_idx]

    imputer = SimpleImputer(strategy="median")
    X_tr_imp = imputer.fit_transform(X_tr)
    X_va_imp = imputer.transform(X_va)

    pos = int(y_tr.sum())
    neg = int((1 - y_tr).sum())
    spw = (neg / max(pos, 1)) if pos > 0 else 1.0

    clf = XGBClassifier(**{**base_params, "scale_pos_weight": spw})
    clf.fit(X_tr_imp, y_tr, eval_set=[(X_va_imp, y_va)], early_stopping_rounds=50, verbose=False)

    proba = clf.predict_proba(X_va_imp)[:, 1]

    # Threshold Youden
    fpr, tpr, thr = roc_curve(y_va, proba)
    youden = tpr - fpr
    best_thr = thr[youden.argmax()]
    thresholds.append(best_thr)

    pred = (proba >= best_thr).astype(int)

    auc = roc_auc_score(y_va, proba)
    acc = accuracy_score(y_va, pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_va, pred, average="binary", zero_division=0
    )

    fold_metrics.append({
        "fold": fold,
        "n_train": len(y_tr),
        "n_val": len(y_va),
        "pos_train": int(y_tr.sum()),
        "pos_val": int(y_va.sum()),
        "scale_pos_weight": round(spw, 3),
        "AUC": auc, "Accuracy": acc, "Precision": prec,
        "Recall": rec, "F1": f1, "best_thr": best_thr,
    })

cv_df = pd.DataFrame(fold_metrics)
cv_df.to_csv(OUTDIR / "xgb_group_cv_metrics.csv", index=False)
print("\n🔁 5-fold GroupKFold CV (train+val por paciente):")
print(cv_df.round(3))
print("\nMédias CV:", cv_df.drop(columns=["fold"]).mean(numeric_only=True).round(3).to_dict())

thr_final = float(np.mean(thresholds))
print(f"\n🔎 Threshold médio (Youden): {thr_final:.3f}")

# --------------------
# Treino final em todo o train+val
# --------------------
imputer_final = SimpleImputer(strategy="median")
X_tv_imp = imputer_final.fit_transform(X_trainval)

pos_tot = int(y_trainval.sum())
neg_tot = int((1 - y_trainval).sum())
spw_tot = (neg_tot / max(pos_tot, 1)) if pos_tot > 0 else 1.0

final_clf = XGBClassifier(**{**base_params, "scale_pos_weight": spw_tot})
final_clf.fit(X_tv_imp, y_trainval, verbose=False)

# Avaliação no test com threshold médio
X_test_imp = imputer_final.transform(X_test)
proba_test = final_clf.predict_proba(X_test_imp)[:, 1]
pred_test = (proba_test >= thr_final).astype(int)

print("\n🧪 Test set report")
print(classification_report(y_test, pred_test, digits=3))
print("ROC AUC (test):", round(roc_auc_score(y_test, proba_test), 3))

# Importâncias (gain)
importances = final_clf.get_booster().get_score(importance_type="gain")
imp_named = {}
for k, v in importances.items():
    try:
        i = int(k.replace("f", ""))
        imp_named[feat_names[i]] = v
    except Exception:
        imp_named[k] = v
pd.Series(imp_named).sort_values(ascending=False).to_csv(OUTDIR / "xgb_group_feature_importance.csv")

# Guardar modelo + meta
to_save = {
    "imputer": imputer_final,
    "model": final_clf,
    "feature_names": feat_names,
    "threshold": thr_final,
}
model_path = MODELS / "xgb_group_final.joblib"
joblib.dump(to_save, model_path)

meta = {
    "features_used": feat_names,
    "params": base_params | {"scale_pos_weight": spw_tot},
    "cv_summary": cv_df.drop(columns=["fold"]).mean(numeric_only=True).to_dict(),
    "threshold_final": thr_final,
    "paths": {"features": str(FEATS), "model": str(model_path)},
}
with open(OUTDIR / "xgb_group_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print(f"\n💾 Modelo salvo em: {model_path}")
print(f"📝 CV metrics: {OUTDIR/'xgb_group_cv_metrics.csv'}")
print(f"⭐ Importância de features: {OUTDIR/'xgb_group_feature_importance.csv'}")
