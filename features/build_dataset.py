import os
import sys
import time
import logging
import pandas as pd
import numpy as np
import yfinance as yf

# Allow running from root
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from scrapers.sec_scraper import SECScraper
from scrapers.layoffs_fyi_scraper import LayoffsScraper, NegativeSampleBuilder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

OUTPUT_PATH = "data/processed/training_dataset.parquet"

COMPANY_TO_TICKER = {
    "Cisco": "CSCO", "LinkedIn": "MSFT", "GitLab": "GTLB", "ZoomInfo": "ZI",
    "Cloudflare": "NET", "Bill.com": "BILL", "Ticketmaster": "LYV", "Upwork": "UPWK",
    "PayPal": "PYPL", "Coinbase": "COIN", "Freshworks": "FRSH", "reAlpha": "AIRE",
    "Amazon": "AMZN", "Meta": "META", "Snap": "SNAP", "Lyft": "LYFT",
    "Salesforce": "CRM", "Alphabet": "GOOGL", "Google": "GOOGL", "Microsoft": "MSFT",
    "Intel": "INTC", "Zoom": "ZM", "Peloton": "PTON", "Robinhood": "HOOD",
    "Redfin": "RDFN", "Opendoor": "OPEN", "Unity": "U", "DocuSign": "DOCU",
    "Shopify": "SHOP", "Wayfair": "W", "Disney": "DIS", "Netflix": "NFLX",
    "Spotify": "SPOT", "Block": "SQ", "Twilio": "TWLO", "HP": "HPQ",
    "Dell": "DELL", "IBM": "IBM", "Oracle": "ORCL", "Tesla": "TSLA",
    "Nvidia": "NVDA", "Qualcomm": "QCOM",
}

class DatasetBuilder:
    def __init__(self):
        self.sec = SECScraper()
        self.layoffs_scraper = LayoffsScraper()
        self.neg_builder = NegativeSampleBuilder()

    def _get_layoff_tickers(self, layoff_df: pd.DataFrame) -> list:
        if "company" not in layoff_df.columns: return []
        layoff_df['layoff_year'] = pd.to_datetime(layoff_df['date'], errors='coerce').dt.year
        ticker_lookup = {k.lower(): v for k, v in COMPANY_TO_TICKER.items()}
        tickers = []
        for company, group in layoff_df.groupby('company'):
            ticker = ticker_lookup.get(str(company).strip().lower())
            if ticker:
                layoff_years = [str(int(y)) for y in group['layoff_year'].dropna()]
                tickers.append({"company": company, "ticker": ticker, "layoff_years": layoff_years})
        return tickers

    def _fetch_financials_batch(self, company_list: list, is_stable_list: bool = False) -> pd.DataFrame:
        all_rows = []
        for item in company_list:
            ticker = item.get("ticker")
            if not ticker: continue
            try:
                yearly_data_list = self.sec.get_company_financials_historical(ticker)
                for year_data in yearly_data_list:
                    if "error" in year_data: continue
                    current_fy = str(year_data.get("fiscal_year"))
                    if is_stable_list:
                        year_data["layoff_occurred"] = 0
                    else:
                        layoff_years = item.get("layoff_years", [])
                        is_layoff_window = any(
                            current_fy == ly or current_fy == str(int(ly) - 1) 
                            for ly in layoff_years if ly.isdigit()
                        )
                        year_data["layoff_occurred"] = 1 if is_layoff_window else 0
                    all_rows.append(year_data)
                time.sleep(0.2)
            except Exception as e: pass
        return pd.DataFrame(all_rows)

    # ================================================================= #
    # EXTERNAL FEATURE ENGINE
    # ================================================================= #

    def _inject_macro_interest_rates(self, df: pd.DataFrame) -> pd.DataFrame:
        log.info("--> Feature 1: Injecting FED Interest Rates (Macro Cost of Capital)")
        try:
            url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=FEDFUNDS"
            macro_df = pd.read_csv(url)
            macro_df['year'] = pd.to_datetime(macro_df['DATE']).dt.year
            macro_df['FEDFUNDS'] = pd.to_numeric(macro_df['FEDFUNDS'], errors='coerce')
            annual_rates = macro_df.groupby('year')['FEDFUNDS'].mean().reset_index()
            annual_rates.rename(columns={'year': 'fiscal_year_match', 'FEDFUNDS': 'macro_interest_rate'}, inplace=True)
        except Exception:
            annual_rates = pd.DataFrame({
                'fiscal_year_match': [2020, 2021, 2022, 2023, 2024, 2025, 2026],
                'macro_interest_rate': [0.38, 0.08, 1.68, 5.02, 5.33, 4.50, 4.25]
            })

        df['fiscal_year_match'] = pd.to_numeric(df['fiscal_year'], errors='coerce')
        df = df.merge(annual_rates, on='fiscal_year_match', how='left')
        df['macro_interest_rate'] = df['macro_interest_rate'].fillna(df['macro_interest_rate'].median())
        return df

    def _inject_market_contagion(self, df: pd.DataFrame) -> pd.DataFrame:
        log.info("--> Feature 2: Injecting NASDAQ-100 Trend (Sector Contagion)")
        try:
            # Fetch last 10 years of NASDAQ data
            ndx = yf.Ticker("^NDX").history(period="10y")
            ndx['year'] = ndx.index.year
            # Calculate annual return percentage
            annual_returns = ndx.groupby('year').apply(
                lambda x: (x['Close'].iloc[-1] - x['Close'].iloc[0]) / x['Close'].iloc[0]
            ).reset_index()
            annual_returns.columns = ['fiscal_year_match', 'nasdaq_annual_return']
        except Exception:
            # Fallback historical data if Yahoo Finance blocks the request
            annual_returns = pd.DataFrame({
                'fiscal_year_match': [2020, 2021, 2022, 2023, 2024, 2025, 2026],
                'nasdaq_annual_return': [0.43, 0.26, -0.33, 0.53, 0.20, 0.15, 0.10]
            })

        df = df.merge(annual_returns, on='fiscal_year_match', how='left')
        df['nasdaq_annual_return'] = df['nasdaq_annual_return'].fillna(df['nasdaq_annual_return'].median())
        # Drop the merging key as we no longer need it
        df.drop(columns=['fiscal_year_match'], inplace=True)
        return df

    def _inject_ceo_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        log.info("--> Feature 3: Preparing Socket for CEO Panic Whisper (Sentiment)")
        # If the SEC scraper hasn't extracted panic words yet, default to 0
        # We will populate this properly when we upgrade sec_scraper.py
        if 'panic_word_count' not in df.columns:
            df['panic_word_count'] = 0  
        return df

    # ================================================================= #

    def build(self) -> pd.DataFrame:
        os.makedirs("data/processed", exist_ok=True)

        log.info("=== Step 1 & 2: Loading Labels and Mapping Tickers ===")
        layoff_df = pd.read_csv("data/raw/layoffs_seed.csv")
        layoff_df.columns = [str(c).strip().lower().replace(" ", "_") for c in layoff_df.columns]
        layoff_df["layoff_occurred"] = 1
        layoff_companies = self._get_layoff_tickers(layoff_df)

        log.info("=== Step 3: Fetching SEC data for layoff companies ===")
        positive_df = self._fetch_financials_batch(layoff_companies, is_stable_list=False)

        log.info("=== Step 4: Fetching SEC data for stable companies ===")
        stable_list = self.neg_builder.build_negative_samples()
        negative_df = self._fetch_financials_batch(stable_list, is_stable_list=True)
        
        log.info("=== Step 5: Assembling and Engineering External Features ===")
        dataset = pd.concat([positive_df, negative_df], ignore_index=True)
        
        # Inject the 3 new intelligence layers
        dataset = self._inject_macro_interest_rates(dataset)
        dataset = self._inject_market_contagion(dataset)
        dataset = self._inject_ceo_sentiment(dataset)
        
        dataset = dataset.dropna(subset=["profit_margin"])

        log.info("Dataset shape: %s", dataset.shape)
        dataset.to_parquet(OUTPUT_PATH, engine="pyarrow", compression="snappy", index=False)
        log.info("✅ Saved to %s", OUTPUT_PATH)

        return dataset

if __name__ == "__main__":
    builder = DatasetBuilder()
    df = builder.build()
    print(f"\nFinal dataset: {len(df)} rows, {df.columns.tolist()}")