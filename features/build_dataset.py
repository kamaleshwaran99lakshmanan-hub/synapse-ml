"""
Dataset Builder
----------------
Joins SEC financial data with layoff.fyi ground-truth labels
to produce a real training dataset.

Pipeline:
  1. Fetch layoff events (label=1) from layoffs.fyi
  2. Map company names → SEC tickers
  3. Fetch financial data from EDGAR for each company
  4. Fetch stable companies (label=0) as negative examples
  5. Compute derived features
  6. Export to data/processed/training_dataset.csv

Run: python -m features.build_dataset
"""

import os
import sys
import time
import logging
import pandas as pd
import numpy as np

# Allow running from root
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from scrapers.sec_scraper import SECScraper
from scrapers.layoffs_fyi_scraper import LayoffsScraper, NegativeSampleBuilder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

OUTPUT_PATH = "data/processed/training_dataset.csv"

# Manual curated mapping: company name → SEC ticker
# (layoffs.fyi uses company names, SEC uses tickers)
COMPANY_TO_TICKER = {
    # From live scraper - exact names
    "Cisco":        "CSCO",
    "LinkedIn":     "MSFT",   # owned by Microsoft
    "GitLab":       "GTLB",
    "ZoomInfo":     "ZI",
    "Cloudflare":   "NET",
    "Bill.com":     "BILL",
    "Ticketmaster": "LYV",    # owned by Live Nation
    "Upwork":       "UPWK",
    "PayPal":       "PYPL",
    "Coinbase":     "COIN",
    "Freshworks":   "FRSH",
    "Guesty":       None,     # private
    "Breadfast":    None,     # private
    "MRI Software": None,     # private
    "Adda247":      None,     # private
    "Parker":       None,     # private (filed bankruptcy)
    "Staffbase":    None,     # private
    "DeepL":        None,     # private
    "Truecaller":   None,     # not US-listed
    "Arctic Wolf":  None,     # private
    "Pocket FM":    None,     # private
    "LSports":      None,     # private
    "ApnaMart":     None,     # private
    "reAlpha":      "AIRE",
    # Existing ones - keep these for broader dataset
    "Amazon":       "AMZN",
    "Meta":         "META",
    "Snap":         "SNAP",
    "Lyft":         "LYFT",
    "Salesforce":   "CRM",
    "Alphabet":     "GOOGL",
    "Google":       "GOOGL",
    "Microsoft":    "MSFT",
    "Intel":        "INTC",
    "Zoom":         "ZM",
    "Peloton":      "PTON",
    "Robinhood":    "HOOD",
    "Redfin":       "RDFN",
    "Opendoor":     "OPEN",
    "Unity":        "U",
    "DocuSign":     "DOCU",
    "Shopify":      "SHOP",
    "Wayfair":      "W",
    "Disney":       "DIS",
    "Netflix":      "NFLX",
    "Spotify":      "SPOT",
    "Block":        "SQ",
    "Twilio":       "TWLO",
    "HP":           "HPQ",
    "Dell":         "DELL",
    "IBM":          "IBM",
    "Oracle":       "ORCL",
    "Tesla":        "TSLA",
    "Nvidia":       "NVDA",
    "Qualcomm":     "QCOM",
}


class DatasetBuilder:
    def __init__(self):
        self.sec = SECScraper()
        self.layoffs_scraper = LayoffsScraper()
        self.neg_builder = NegativeSampleBuilder()

    # ------------------------------------------------------------------ #
    #  Step 1: Get layoff company names from layoffs.fyi
    # ------------------------------------------------------------------ #
    def _get_layoff_tickers(self, layoff_df: pd.DataFrame) -> list:
        """Map company names from layoffs.fyi to SEC tickers."""
        if "company" not in layoff_df.columns:
            return []
        
        tickers = []
        for company in layoff_df["company"].unique():
            ticker = COMPANY_TO_TICKER.get(company)
            if ticker:
                tickers.append({"company": company, "ticker": ticker, "layoff_occurred": 1})
        
        log.info(f"Mapped {len(tickers)} companies to SEC tickers.")
        return tickers

    # ------------------------------------------------------------------ #
    #  Step 2: Fetch SEC financials for each ticker
    # ------------------------------------------------------------------ #
    def _fetch_financials_batch(self, company_list: list) -> pd.DataFrame:
        """
        Fetch SEC financials for a list of {company, ticker, layoff_occurred} dicts.
        Returns flattened DataFrame with ML features.
        """
        rows = []
        for item in company_list:
            ticker = item.get("ticker")
            if not ticker:
                continue
            
            log.info(f"Fetching SEC data for {ticker}...")
            try:
                data = self.sec.get_company_financials(ticker)
                if "error" in data:
                    log.warning(f"  Skip {ticker}: {data['error']}")
                    continue
                
                # Flatten to ML-ready feature row
                row = self._to_feature_row(data, label=item["layoff_occurred"])
                if row:
                    rows.append(row)
                
                time.sleep(0.2)  # Respect SEC rate limits
            except Exception as e:
                log.error(f"  Failed {ticker}: {e}")
        
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ #
    #  Step 3: Feature engineering
    # ------------------------------------------------------------------ #
    def _to_feature_row(self, sec_data: dict, label: int) -> dict:
        """Convert raw SEC data dict → single ML-ready feature row."""
        
        rev = sec_data.get("revenue_usd")
        ni = sec_data.get("net_income_usd")
        oi = sec_data.get("operating_income_usd")
        emp = sec_data.get("employees")
        ltd = sec_data.get("long_term_debt_usd")
        assets = sec_data.get("total_assets_usd")
        liab = sec_data.get("total_liabilities_usd")
        equity = sec_data.get("stockholders_equity_usd")
        
        # Skip companies with missing core data
        if not rev or not ni:
            return {}
        
        # Revenue trend: compute 2-year CAGR from series
        rev_series = sec_data.get("revenue_series", [])
        revenue_cagr = None
        if len(rev_series) >= 3:
            oldest_val = rev_series[0][1]
            newest_val = rev_series[-1][1]
            n_years = len(rev_series) - 1
            if oldest_val > 0:
                revenue_cagr = (newest_val / oldest_val) ** (1 / n_years) - 1
        
        # Consecutive loss years
        ni_series = sec_data.get("net_income_growth_yoy")
        
        # Safe division helper
        def safe_div(a, b):
            try:
                return float(a) / float(b) if b and b != 0 else None
            except (TypeError, ZeroDivisionError):
                return None

        return {
            # Identity
            "ticker": sec_data.get("ticker"),
            "entity_name": sec_data.get("entity_name"),
            
            # Core financial features (normalized)
            "revenue_usd": rev,
            "net_income_usd": ni,
            "profit_margin": safe_div(ni, rev),
            "operating_margin": safe_div(oi, rev),
            
            # Solvency features
            "debt_to_equity": safe_div(ltd, equity),
            "debt_to_assets": safe_div(ltd, assets),
            "asset_to_liability": safe_div(assets, liab),
            "equity_ratio": safe_div(equity, assets),
            
            # Efficiency features
            "revenue_per_employee": safe_div(rev, emp),
            "net_income_per_employee": safe_div(ni, emp),
            
            # Growth features
            "revenue_growth_yoy": sec_data.get("revenue_growth_yoy"),
            "net_income_growth_yoy": sec_data.get("net_income_growth_yoy"),
            "revenue_cagr_3yr": revenue_cagr,
            
            # Scale (log-transformed to handle wide range)
            "log_revenue": np.log1p(rev) if rev and rev > 0 else None,
            "log_employees": np.log1p(emp) if emp and emp > 0 else None,
            "employees": emp,
            
            # Label
            "layoff_occurred": label,
        }

    # ------------------------------------------------------------------ #
    #  Main pipeline
    # ------------------------------------------------------------------ #
    def build(self) -> pd.DataFrame:
        os.makedirs("data/processed", exist_ok=True)
        
        # --- Positive samples (layoff=1) ---
        log.info("=== Step 1: Fetching layoff labels ===")
        layoff_df = self.layoffs_scraper.fetch_layoff_labels()
        
        log.info("=== Step 2: Mapping companies to SEC tickers ===")
        layoff_companies = self._get_layoff_tickers(layoff_df)
        
        log.info("=== Step 3: Fetching SEC data for layoff companies ===")
        positive_df = self._fetch_financials_batch(layoff_companies)
        log.info(f"Positive samples collected: {len(positive_df)}")
        
        # --- Negative samples (layoff=0) ---
        log.info("=== Step 4: Fetching SEC data for stable companies ===")
        stable_list = self.neg_builder.build_negative_samples()
        negative_df = self._fetch_financials_batch(stable_list)
        log.info(f"Negative samples collected: {len(negative_df)}")
        
        # --- Combine ---
        log.info("=== Step 5: Assembling final dataset ===")
        dataset = pd.concat([positive_df, negative_df], ignore_index=True)
        dataset = dataset.dropna(subset=["profit_margin", "revenue_growth_yoy"])
        
        log.info(f"Dataset shape: {dataset.shape}")
        log.info(f"Label distribution:\n{dataset['layoff_occurred'].value_counts()}")
        
        dataset.to_csv(OUTPUT_PATH, index=False)
        log.info(f"✅ Saved to {OUTPUT_PATH}")
        
        return dataset


if __name__ == "__main__":
    builder = DatasetBuilder()
    df = builder.build()
    print(f"\nFinal dataset: {len(df)} rows, {df.columns.tolist()}")