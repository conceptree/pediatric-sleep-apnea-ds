#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Baseline training (Logistic Regression + Random Forest)
with features extracted in extractFeatures.py

Saves the trained models to disk using joblib.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, RocCurveDisplay
import matplotlib.pyplot as plt
import joblib

# =====================
# Load data
# =====================
BASE = Path("/Users/nunorodrigues/dev/tese/datasets/splits")
FEATS = BASE / "features.parquet"
MODELS = BASE / "models"
MODELS.mkdir(exist_ok=True)

df = pd.read_parquet(FEATS)

# drop non-numeric or meta columns
X = df.drop(columns=["subset","label","stem","error"], errors="ignore")
y = df["label"]

# remove columns with NaN in >10% of the rows
mask = X.isna().mean() < 0.1
X = X.loc[:, mask]
X = X.fillna(X.median())

# split train/val/test according to 'subset'
X_train = X[df["subset"]=="train"]
y_train = y[df["subset"]=="train"]
X_val   = X[df["subset"]=="val"]
y_val   = y[df["subset"]=="val"]
X_test  = X[df["subset"]=="test"]
y_test  = y[df["subset"]=="test"]

print(f"Train={len(y_train)}, Val={len(y_val)}, Test={len(y_test)}")

# =====================
# Logistic Regression
# =====================
pipe_lr = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=200, class_weight="balanced"))
])
pipe_lr.fit(X_train, y_train)

print("\n=== Logistic Regression ===")
print("Val report:\n", classification_report(y_val, pipe_lr.predict(X_val)))
print("Test report:\n", classification_report(y_test, pipe_lr.predict(X_test)))
print("ROC AUC (test):", roc_auc_score(y_test, pipe_lr.predict_proba(X_test)[:,1]))

# save model
joblib.dump(pipe_lr, MODELS / "logreg_model.joblib")
print(f"💾 Logistic Regression saved at {MODELS/'logreg_model.joblib'}")

# =====================
# Random Forest
# =====================
rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
rf.fit(X_train, y_train)

print("\n=== Random Forest ===")
print("Val report:\n", classification_report(y_val, rf.predict(X_val)))
print("Test report:\n", classification_report(y_test, rf.predict(X_test)))
print("ROC AUC (test):", roc_auc_score(y_test, rf.predict_proba(X_test)[:,1]))

# save model
joblib.dump(rf, MODELS / "rf_model.joblib")
print(f"💾 Random Forest saved at {MODELS/'rf_model.joblib'}")

# =====================
# ROC Curve plot
# =====================
RocCurveDisplay.from_estimator(pipe_lr, X_test, y_test, name="LogReg")
RocCurveDisplay.from_estimator(rf, X_test, y_test, name="RandomForest")
plt.title("ROC curves (Test set)")
plt.show()
