"""
Layoffs Ground Truth Scraper (Upgraded)
-----------------------------
Scrapes layoffs.fyi for real labeled layoff events.
Now using cloudscraper to bypass Cloudflare security blocks.
"""

import pandas as pd
import logging
import json
import re
import cloudscraper  # Ensure you ran: pip install cloudscraper
from io import StringIO
from bs4 import BeautifulSoup

log = logging.getLogger(__name__)

# Target URLs
LAYOFFS_FYI_CSV = "https://docs.google.com/spreadsheets/d/1LpseE3T5oGH-DVhQe3HL8yAyp5wBWsMzVKJoFuMW1F8/export?format=csv"
LAYOFFS_FYI_DIRECT = "https://layoffs.fyi/"

class LayoffsScraper:
    def __init__(self):
        # We initialize the cloudscraper browser object here
        self.scraper = cloudscraper.create_scraper(
            browser={
                'browser': 'chrome',
                'platform': 'darwin',
                'desktop': True
            }
        )

    def fetch_layoff_labels(self) -> pd.DataFrame:
        """
        Main entry point used by build_dataset.py. 
        Returns a DataFrame with confirmed layoff events.
        """
        log.info("🔍 Fetching live layoff labels...")
        
        # 1. Try the Google Sheet first (fastest if it works)
        try:
            resp = self.scraper.get(LAYOFFS_FYI_CSV, timeout=15)
            if resp.status_code == 200:
                df = pd.read_csv(StringIO(resp.text))
                log.info(f"✅ Loaded {len(df)} records from Google Sheet.")
                return self._clean_layoffs_df(df)
        except Exception as e:
            log.warning(f"⚠️ Google Sheet access failed: {e}")
        
        # 2. Fallback: Scrape the site directly using cloudscraper bypass
        return self._fetch_from_site()

    def _fetch_from_site(self) -> pd.DataFrame:
        log.info("🎭 Loading layoffs.fyi and reading Airtable iframe...")
        try:
            from playwright.sync_api import sync_playwright

            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                page.goto(LAYOFFS_FYI_DIRECT, timeout=60000, wait_until="domcontentloaded")

                page.wait_for_timeout(10000)

                frame = page.frame_locator('iframe').first

                # Scroll inside the iframe to load more rows
                table = frame.locator('#table')
                for _ in range(30):
                    table.evaluate('el => el.scrollTop += 800')
                    page.wait_for_timeout(500)

                txt = table.inner_text(timeout=10000)
                browser.close()

            lines = [l.strip() for l in txt.splitlines() if l.strip()]

            # Remove toolbar noise
            skip = {
                "Hide fields", "Filter", "Group", "Sort",
                "Drag to adjust the number of frozen columns",
                "Company", "Location HQ", "# Laid Off", "Date",
                "%", "Industry", "Source", "Stage",
                "$ Raised (mm)", "Country", "Date Added"
            }
            lines = [l for l in lines if l not in skip]

            # Split: numbered company list comes first, then all data values
            companies = []
            data_lines = []
            i = 0

            # Collect numbered company names (lines after a digit line)
            while i < len(lines):
                if lines[i].isdigit():
                    if i + 1 < len(lines):
                        companies.append(lines[i + 1])
                        i += 2
                    else:
                        i += 1
                else:
                    break  # end of company list, data starts here

            data_lines = lines[i:]

            # Each row = 10 fields: location, laid_off, date, pct,
            #                       industry, source, stage, raised, country, date_added
            FIELDS = ["location_hq", "employees_laid_off", "date", "pct_laid_off",
                      "industry", "source", "stage", "raised_mm", "country", "date_added"]
            N = len(FIELDS)

            records = []
            for idx, company in enumerate(companies):
                chunk = data_lines[idx * N: (idx + 1) * N]
                if len(chunk) == N:
                    record = {"company": company}
                    record.update(dict(zip(FIELDS, chunk)))
                    records.append(record)

            df = pd.DataFrame(records)
            log.info(f"✅ Extracted {len(df)} live rows from Airtable iframe.")
            return self._clean_layoffs_df(df)

        except Exception as e:
            log.error(f"❌ Airtable iframe scrape failed: {e}")
            return pd.DataFrame()

    def _clean_layoffs_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standard cleaning logic used by the rest of the pipeline."""
        df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]
        
        rename_map = {
            "company": "company",
            "date_added": "date",
            "laid_off": "employees_laid_off",
            "percentage": "pct_laid_off",
            "location_hq": "country",
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        df["layoff_occurred"] = 1
        
        for col in ["employees_laid_off", "pct_laid_off"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        
        return df

class NegativeSampleBuilder:
    """
    Kept for compatibility with build_dataset.py.
    Provides tickers for companies that did NOT have major layoffs.
    """
    STABLE_TICKERS = ["AAPL", "MSFT", "GOOGL", "V", "MA", "JNJ", "PG", "KO", "WMT"]
    
    def build_negative_samples(self) -> list:
        return [
            {"ticker": t, "layoff_occurred": 0, "source": "stable_list"}
            for t in self.STABLE_TICKERS
        ]

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    scraper = LayoffsScraper()
    df = scraper.fetch_layoff_labels()
    
    if not df.empty:
        print(f"\n✨ Success! Layoff Records: {len(df):,}")
        print(df.head(5).to_string())
    else:
        print("\n❌ Failed to fetch data. Check your connection or cloudscraper installation.")