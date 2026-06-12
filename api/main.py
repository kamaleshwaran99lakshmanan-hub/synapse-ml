"""
Synapse ML FastAPI Engine
--------------------------
Serves LIVE predictions synced with Macro, Contagion, and NLP features.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import yfinance as yf
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from scrapers.sec_scraper import SECScraper

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Synapse ML API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Load the leak-proof model
MODEL_PATH = "models/layoff_xgboost_model.json"
model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)
OPTIMAL_THRESHOLD = 0.2137  # From our leak-proof evaluation

sec_scraper = SECScraper()

# The exact feature order the XGBoost model expects
EXPECTED_FEATURES = [
    'revenue_usd', 'net_income_usd', 'operating_income_usd', 'total_assets_usd', 
    'total_liabilities_usd', 'long_term_debt_usd', 'stockholders_equity_usd', 
    'employees', 'profit_margin', 'operating_margin', 'debt_to_equity', 
    'debt_to_assets', 'asset_to_liability', 'equity_ratio', 'revenue_per_employee', 
    'net_income_per_employee', 'revenue_growth_yoy', 'net_income_growth_yoy', 
    'revenue_cagr_3yr', 'log_revenue', 'log_employees', 'panic_word_count', 
    'macro_interest_rate', 'nasdaq_annual_return'
]

class PredictRequest(BaseModel):
    ticker: str

def fetch_live_macro_and_market():
    """Fetches real-time FED rate and NASDAQ contagion features."""
    try:
        # Fallback rates if APIs are slow
        macro_rate = 4.25 
        ndx_trend = 0.10
        
        # Live NASDAQ Fetch
        ndx = yf.Ticker("^NDX").history(period="1y")
        if not ndx.empty:
            ndx_trend = (ndx['Close'].iloc[-1] - ndx['Close'].iloc[0]) / ndx['Close'].iloc[0]
            
        # Live FED Fetch (Using pandas to read FRED CSV)
        try:
            url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=FEDFUNDS"
            macro_df = pd.read_csv(url)
            macro_rate = float(macro_df['FEDFUNDS'].iloc[-1])
        except Exception:
            pass
            
        return macro_rate, ndx_trend
    except Exception as e:
        log.warning(f"Using fallback macro features: {e}")
        return 4.25, 0.10

@app.post("/predict")
async def predict_risk(request: PredictRequest):
    ticker = sec_scraper.resolve_company_to_ticker(request.ticker)

    print("User entered:", request.ticker)
    print("Resolved ticker:", ticker)

    if ticker is None:
     raise HTTPException(
        status_code=404,
        detail="Company or ticker not found."
      )
    
    try:
        # 1. Fetch SEC Data & NLP Panic Words
        log.info(f"[{ticker}] Fetching SEC & NLP data...")
        historical_data = sec_scraper.get_company_financials_historical(ticker)
        
        if not historical_data or "error" in historical_data[0]:
            raise HTTPException(status_code=404, detail="Company SEC data not found.")
            
        # Grab the most recent fiscal year's data
        latest_data = historical_data[-1] 
        
        # 2. Fetch Live External Features
        log.info(f"[{ticker}] Fetching live Macro & Market Contagion...")
        macro_rate, ndx_trend = fetch_live_macro_and_market()
        latest_data['macro_interest_rate'] = macro_rate
        latest_data['nasdaq_annual_return'] = ndx_trend
        
        # 3. Format strictly for XGBoost
        df_features = pd.DataFrame([latest_data])
        for col in EXPECTED_FEATURES:
            if col not in df_features.columns:
                df_features[col] = 0.0 # Safety fallback
                
        X = df_features[EXPECTED_FEATURES].apply(pd.to_numeric, errors='coerce').fillna(0)
        
        # 4. Inference
        prob = float(model.predict_proba(X)[0][1])
        risk_level = "High Risk" if prob >= OPTIMAL_THRESHOLD else "Low Risk"
        
        return {
            "ticker": ticker,
            "company_name": latest_data.get("entity_name", ticker),
            "fiscal_year_analyzed": latest_data.get("fiscal_year", "Unknown"),
            "layoff_probability": round(prob, 4),
            "risk_assessment": risk_level,
            "intelligence_signals": {
    "ceo_panic_words": latest_data.get("panic_word_count", 0),

    "panic_words": latest_data.get(
        "panic_words",
        []
    ),

    "panic_word_frequencies": latest_data.get(
        "panic_word_frequencies",
        {}
    ),

    "fed_interest_rate": round(macro_rate, 2),

    "nasdaq_trend": round(ndx_trend, 2)
}
        }
    except Exception as e:
        log.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search/{query}")
def search_ticker(query: str):
    query = query.lower()

    try:
        # Access ticker map (we’ll fix this below)
        ticker_map = sec_scraper._ticker_map  

        matches = [
            {"ticker": ticker, "name": name}
            for name, ticker in ticker_map.items()
            if query in name.lower() or query in ticker.lower()
        ]

        return matches[:10]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)