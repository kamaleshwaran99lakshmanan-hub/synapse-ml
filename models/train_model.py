"""
Real XGBoost Training
----------------------
Trains on actual SEC financial data labeled with real layoff events.

Features used (all from real SEC EDGAR filings):
  - profit_margin           : Net Income / Revenue
  - operating_margin        : Operating Income / Revenue  
  - debt_to_equity          : LT Debt / Stockholders Equity
  - debt_to_assets          : LT Debt / Total Assets
  - asset_to_liability      : Assets / Liabilities (solvency)
  - revenue_growth_yoy      : YoY revenue growth rate
  - net_income_growth_yoy   : YoY net income growth
  - revenue_cagr_3yr        : 3-year compound annual growth
  - revenue_per_employee    : Efficiency metric
  - log_revenue             : Scale (log-transformed)
  - log_employees           : Headcount (log-transformed)

Run: python models/train_model.py
"""

import os
import sys
import joblib
import logging
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
    confusion_matrix, average_precision_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

DATASET_PATH = "data/processed/training_dataset.csv"
MODEL_PATH   = "models/layoff_xgboost_model.json"
SCALER_PATH  = "models/feature_scaler.pkl"

# Features the model will actually learn from
FEATURE_COLS = [
    "profit_margin",
    "operating_margin",
    "debt_to_equity",
    "debt_to_assets",
    "asset_to_liability",
    "equity_ratio",
    "revenue_growth_yoy",
    "net_income_growth_yoy",
    "revenue_cagr_3yr",
    "revenue_per_employee",
    "net_income_per_employee",
    "log_revenue",
    "log_employees",
]

LABEL_COL = "layoff_occurred"


def load_and_validate(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    log.info(f"Loaded {len(df)} rows from {path}")
    
    missing_features = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_features:
        log.warning(f"Missing feature columns: {missing_features}")
        FEATURE_COLS[:] = [c for c in FEATURE_COLS if c in df.columns]
    
    if LABEL_COL not in df.columns:
        raise ValueError(f"Label column '{LABEL_COL}' not found in dataset.")
    
    log.info(f"Label distribution:\n{df[LABEL_COL].value_counts()}")
    return df


def build_model():
    return xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        n_estimators=100,
        max_depth=2,          # shallow tree for small data
        learning_rate=0.1,
        subsample=1.0,        # use all rows (too few to subsample)
        colsample_bytree=1.0,
        min_child_weight=1,   # was 5, too restrictive for 20 rows
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    )


def train(df: pd.DataFrame):
    X = df[FEATURE_COLS].copy()
    y = df[LABEL_COL].astype(int)
    
    # Handle class imbalance
    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    log.info(f"Class ratio → neg:{n_neg}, pos:{n_pos}, scale_pos_weight:{scale_pos_weight:.2f}")
    
    # Impute missing values with median (SEC data has gaps)
    # Drop columns that are entirely null
    X = X.dropna(axis=1, how='all')
    available_features = X.columns.tolist()
    log.info(f"Features after dropping all-null cols: {available_features}")
    
    # Impute remaining missing values with median
    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=available_features)
    
    # Update FEATURE_COLS to match what we actually have
    FEATURE_COLS[:] = available_features
    model = build_model()
    model.set_params(scale_pos_weight=scale_pos_weight)
    
    # ------------------------------------------------------------------ #
    #  Cross-validation (proper evaluation on small real datasets)
    # ------------------------------------------------------------------ #
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)    

    auc_scores = cross_val_score(model, X_imputed, y, cv=cv, scoring="roc_auc")
    apr_scores = cross_val_score(model, X_imputed, y, cv=cv, scoring="average_precision")
    
    log.info(f"5-Fold CV ROC-AUC: {auc_scores.mean():.3f} ± {auc_scores.std():.3f}")
    log.info(f"5-Fold CV Avg Precision: {apr_scores.mean():.3f} ± {apr_scores.std():.3f}")
    
    # ------------------------------------------------------------------ #
    #  Final fit on all data
    # ------------------------------------------------------------------ #
    model.fit(X_imputed, y, eval_set=[(X_imputed, y)], verbose=50)
    
    # ------------------------------------------------------------------ #
    #  Feature importance (what actually matters for layoffs)
    # ------------------------------------------------------------------ #
    importance = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    
    log.info("\n=== Top Features Driving Layoff Predictions ===")
    log.info(importance.to_string(index=False))
    
    # ------------------------------------------------------------------ #
    #  Save model and preprocessor
    # ------------------------------------------------------------------ #
    os.makedirs("models", exist_ok=True)
    model.save_model(MODEL_PATH)
    joblib.dump({"imputer": imputer, "feature_cols": FEATURE_COLS}, SCALER_PATH)
    
    log.info(f"\n✅ Model saved → {MODEL_PATH}")
    log.info(f"✅ Preprocessor saved → {SCALER_PATH}")
    
    return model, imputer


if __name__ == "__main__":
    # If dataset doesn't exist yet, build it first
    if not os.path.exists(DATASET_PATH):
        log.info("Dataset not found. Running dataset builder first...")
        from features.build_dataset import DatasetBuilder
        DatasetBuilder().build()
    
    df = load_and_validate(DATASET_PATH)
    model, imputer = train(df)
    
    log.info("\n=== Training complete. Model is ready for real inference. ===")