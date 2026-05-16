"""
FastAPI Prediction Server
--------------------------
Serves live predictions by:
  1. Accepting a ticker (e.g. "AMZN")
  2. Fetching LIVE financial data from SEC EDGAR
  3. Running XGBoost inference
  4. Returning structured risk assessment

Start: uvicorn api.main:app --reload
"""

import os
import sys
import joblib
import logging
import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from scrapers.sec_scraper import SECScraper

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

MODEL_PATH      = "models/layoff_xgboost_model.json"
PREPROCESSOR_PATH = "models/feature_scaler.pkl"

# ------------------------------------------------------------------ #
#  Startup: load model + SEC scraper once
# ------------------------------------------------------------------ #
app = FastAPI(
    title="TechHaven Stability API",
    description="Predicts layoff risk from real SEC financial data",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

log.info("🧠 Loading XGBoost model...")
model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)

log.info("📦 Loading preprocessor...")
preprocessor = joblib.load(PREPROCESSOR_PATH)
imputer      = preprocessor["imputer"]
FEATURE_COLS = preprocessor["feature_cols"]

log.info("🌐 Initializing SEC EDGAR scraper...")
sec_scraper = SECScraper()

log.info("✅ Server ready.")


# ------------------------------------------------------------------ #
#  Request/Response schemas
# ------------------------------------------------------------------ #
class PredictRequest(BaseModel):
    ticker: str
    
class FeatureSnapshot(BaseModel):
    profit_margin: float | None
    operating_margin: float | None
    debt_to_equity: float | None
    revenue_growth_yoy: float | None
    net_income_growth_yoy: float | None
    revenue_per_employee: float | None
    log_revenue: float | None

class PredictionResponse(BaseModel):
    ticker: str
    company_name: str
    risk_score: float           # 0.0–1.0 probability of layoff
    risk_label: str             # "High Risk" / "Medium Risk" / "Low Risk"
    confidence: str             # "High" / "Medium" / "Low"
    key_signals: list[str]      # Human-readable red/green flags
    features: dict              # Raw financial features used
    data_source: str
    disclaimer: str


# ------------------------------------------------------------------ #
#  Inference logic
# ------------------------------------------------------------------ #
def classify_risk(score: float) -> tuple[str, str]:
    if score >= 0.65:
        return "High Risk", "High"
    elif score >= 0.40:
        return "Medium Risk", "Medium"
    else:
        return "Low Risk", "High"

def generate_signals(features: dict, score: float) -> list[str]:
    """Convert raw features into human-readable insight strings."""
    signals = []
    
    pm = features.get("profit_margin")
    if pm is not None:
        if pm < -0.05:
            signals.append(f"🔴 Negative profit margin ({pm:.1%}) — company burning cash")
        elif pm < 0.05:
            signals.append(f"🟡 Thin profit margin ({pm:.1%})")
        else:
            signals.append(f"🟢 Healthy profit margin ({pm:.1%})")
    
    rg = features.get("revenue_growth_yoy")
    if rg is not None:
        if rg < -0.10:
            signals.append(f"🔴 Revenue declining ({rg:.1%} YoY) — major contraction signal")
        elif rg < 0.03:
            signals.append(f"🟡 Sluggish revenue growth ({rg:.1%} YoY)")
        else:
            signals.append(f"🟢 Revenue growing ({rg:.1%} YoY)")
    
    de = features.get("debt_to_equity")
    if de is not None:
        if de > 3.0:
            signals.append(f"🔴 Very high debt load (D/E: {de:.1f}x) — restructuring risk")
        elif de > 1.5:
            signals.append(f"🟡 Elevated debt (D/E: {de:.1f}x)")
        else:
            signals.append(f"🟢 Manageable debt (D/E: {de:.1f}x)")
    
    rpe = features.get("revenue_per_employee")
    if rpe is not None:
        rpe_k = rpe / 1000
        if rpe_k < 300:
            signals.append(f"🔴 Low revenue per employee (${rpe_k:.0f}K) — overstaffed")
        elif rpe_k < 600:
            signals.append(f"🟡 Moderate revenue per employee (${rpe_k:.0f}K)")
        else:
            signals.append(f"🟢 High revenue per employee (${rpe_k:.0f}K)")
    
    nig = features.get("net_income_growth_yoy")
    if nig is not None and nig < -0.30:
        signals.append(f"🔴 Profit shrinking fast ({nig:.1%} YoY) — earnings pressure")
    
    return signals if signals else ["⚪ Insufficient signal data for detailed analysis"]


@app.get("/")
def root():
    return {"status": "ok", "service": "TechHaven Stability API v1.0"}


@app.get("/predict/{ticker}", response_model=PredictionResponse)
def predict(ticker: str):
    ticker = ticker.upper().strip()
    
    # 1. Fetch LIVE data from SEC EDGAR
    log.info(f"[{ticker}] Fetching live SEC data...")
    sec_data = sec_scraper.get_company_financials(ticker)
    
    if "error" in sec_data:
        raise HTTPException(
            status_code=404,
            detail=f"Could not fetch SEC data for '{ticker}': {sec_data['error']}"
        )
    
    # 2. Build feature row
    from features.build_dataset import DatasetBuilder
    builder = DatasetBuilder()
    feature_row = builder._to_feature_row(sec_data, label=0)  # label=0 placeholder
    
    if not feature_row:
        raise HTTPException(
            status_code=422,
            detail=f"Insufficient financial data for '{ticker}' (missing revenue/income)"
        )
    
    # 3. Prepare feature vector
    X = pd.DataFrame([feature_row])[FEATURE_COLS]
    X_imputed = pd.DataFrame(imputer.transform(X), columns=FEATURE_COLS)
    
    # 4. Predict
    prob = float(model.predict_proba(X_imputed)[0][1])
    risk_label, confidence = classify_risk(prob)
    signals = generate_signals(feature_row, prob)
    
    # 5. Build response
    return PredictionResponse(
        ticker=ticker,
        company_name=sec_data.get("entity_name", ticker),
        risk_score=round(prob, 3),
        risk_label=risk_label,
        confidence=confidence,
        key_signals=signals,
        features={
            "profit_margin": feature_row.get("profit_margin"),
            "operating_margin": feature_row.get("operating_margin"),
            "debt_to_equity": feature_row.get("debt_to_equity"),
            "revenue_growth_yoy": feature_row.get("revenue_growth_yoy"),
            "net_income_growth_yoy": feature_row.get("net_income_growth_yoy"),
            "revenue_per_employee": feature_row.get("revenue_per_employee"),
            "revenue_usd": sec_data.get("revenue_usd"),
            "net_income_usd": sec_data.get("net_income_usd"),
            "employees": sec_data.get("employees"),
        },
        data_source="SEC EDGAR (XBRL) — 10-K Annual Filing",
        disclaimer=(
            "Prediction based on public SEC filings. "
            "Not financial advice. Past patterns may not predict future events."
        )
    )


@app.get("/batch")
def batch_predict(tickers: str):
    """Predict for multiple tickers. Usage: /batch?tickers=AMZN,META,SNAP"""
    ticker_list = [t.strip().upper() for t in tickers.split(",")]
    results = []
    for t in ticker_list[:10]:  # cap at 10
        try:
            result = predict(t)
            results.append(result)
        except HTTPException as e:
            results.append({"ticker": t, "error": e.detail})
    return results