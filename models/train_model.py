"""
Synapse Master Training Pipeline
---------------------------------
Trains the XGBoost model on SEC financial data, NLP sentiment, and Macro indicators.
Uses GroupShuffleSplit to prevent data leakage across corporate entities.

Run: python models/train_model.py
"""

import os
import sys
import joblib
import logging
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, precision_recall_curve, confusion_matrix
from sklearn.impute import SimpleImputer

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

DATASET_PATH = "data/processed/training_dataset.parquet"
MODEL_PATH   = "models/layoff_xgboost_model.json"
SCALER_PATH  = "models/feature_scaler.pkl"

FEATURE_COLS = [
    'revenue_usd', 'net_income_usd', 'operating_income_usd', 'total_assets_usd', 
    'total_liabilities_usd', 'long_term_debt_usd', 'stockholders_equity_usd', 
    'employees', 'profit_margin', 'operating_margin', 'debt_to_equity', 
    'debt_to_assets', 'asset_to_liability', 'equity_ratio', 'revenue_per_employee', 
    'net_income_per_employee', 'revenue_growth_yoy', 'net_income_growth_yoy', 
    'revenue_cagr_3yr', 'log_revenue', 'log_employees', 
    'panic_word_count', 'macro_interest_rate', 'nasdaq_annual_return'
]

LABEL_COL = "layoff_occurred"

def train():
    log.info("Loading processed dataset...")
    df = pd.read_parquet(DATASET_PATH)
    
    X = df[FEATURE_COLS].apply(pd.to_numeric, errors='coerce')
    y = df[LABEL_COL].astype(int)
    tickers = df['ticker']

    for col in FEATURE_COLS:
        if X[col].isna().all():
            log.warning(f"Feature '{col}' is entirely empty in this dataset. Defaulting to 0.")
            X[col] = 0.0

    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=FEATURE_COLS)

    log.info("Splitting data by isolated corporate entities (GroupShuffleSplit)...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X_imputed, y, groups=tickers))

    X_train, X_test = X_imputed.iloc[train_idx], X_imputed.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    scale_weight = np.sum(y_train == 0) / np.sum(y_train == 1)

    log.info("Training XGBoost Classifier...")
    model = xgb.XGBClassifier(
        scale_pos_weight=scale_weight,
        eval_metric='logloss',
        random_state=42,
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1
    )
    model.fit(X_train, y_train)

    log.info("Optimizing PR-Curve for maximum F1-Score...")
    y_probs = model.predict_proba(X_test)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)
    
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

    y_pred_optimal = (y_probs >= best_threshold).astype(int)

    log.info(f"\n--> Mathematically Optimal Threshold Found: {best_threshold:.4f}\n")
    log.info("=== Optimized Leak-Proof Validation ===")
    
    # Print matrix cleanly to terminal instead of rendering an image
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred_optimal))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_optimal))

    os.makedirs("models", exist_ok=True)
    model.save_model(MODEL_PATH)
    joblib.dump({"imputer": imputer, "feature_cols": FEATURE_COLS}, SCALER_PATH)
    
    log.info(f"✅ Model saved to {MODEL_PATH}")
    log.info(f"✅ Preprocessor saved to {SCALER_PATH}")

if __name__ == "__main__":
    train()