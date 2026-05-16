"""
SEC EDGAR Scraper
-----------------
Fetches real financial data from SEC EDGAR for US-listed companies.
Uses the official EDGAR full-text search + company facts API (free, no key needed).

SEC EDGAR APIs used:
  - https://data.sec.gov/submissions/{CIK}.json        → filing metadata
  - https://data.sec.gov/api/xbrl/companyfacts/{CIK}.json → XBRL financial facts

Usage:
    scraper = SECScraper()
    data = scraper.get_company_financials("AMZN")   # ticker
    data = scraper.get_company_financials("0000018230")  # or CIK directly
"""

import requests
import json
import time
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# SEC requires a User-Agent identifying your app. Put your email here.
# See: https://www.sec.gov/os/accessing-edgar-data
HEADERS = {
    "User-Agent": "TechHaven-Stability-Platform contact@techhaven.io",
    "Accept-Encoding": "gzip, deflate",
}

TICKER_CIK_URL = "https://www.sec.gov/files/company_tickers.json"
COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"


class SECScraper:
    def __init__(self):
        self._ticker_map: dict = {}
        self._load_ticker_map()

    # ------------------------------------------------------------------ #
    #  Ticker → CIK resolution
    # ------------------------------------------------------------------ #
    def _load_ticker_map(self):
        """Download SEC's master ticker→CIK map once per session."""
        try:
            log.info("Loading SEC ticker→CIK map...")
            resp = requests.get(TICKER_CIK_URL, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            raw = resp.json()
            # raw is {index: {cik_str, ticker, title}, ...}
            self._ticker_map = {
                v["ticker"].upper(): str(v["cik_str"]).zfill(10)
                for v in raw.values()
            }
            log.info(f"Loaded {len(self._ticker_map):,} tickers from SEC.")
        except Exception as e:
            log.error(f"Failed to load ticker map: {e}")

    def _resolve_cik(self, identifier: str) -> Optional[str]:
        """Return zero-padded 10-digit CIK from ticker or raw CIK."""
        if identifier.isdigit():
            return identifier.zfill(10)
        cik = self._ticker_map.get(identifier.upper())
        if not cik:
            log.warning(f"Ticker '{identifier}' not found in SEC map.")
        return cik

    # ------------------------------------------------------------------ #
    #  Core fetchers
    # ------------------------------------------------------------------ #
    def _fetch_company_facts(self, cik: str) -> Optional[dict]:
        """Fetch all XBRL facts for a company (revenue, profit, headcount, etc.)."""
        url = COMPANY_FACTS_URL.format(cik=cik)
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as e:
            log.error(f"HTTP error fetching facts for CIK {cik}: {e}")
        except Exception as e:
            log.error(f"Error fetching facts for CIK {cik}: {e}")
        return None

    def _extract_latest_annual(self, facts: dict, concept: str, unit: str = "USD") -> Optional[float]:
        """
        Pull the most recent 10-K annual value for a given XBRL concept.
        
        Common concepts:
          Revenues / RevenueFromContractWithCustomerExcludingAssessedTax
          NetIncomeLoss
          OperatingIncomeLoss
          Assets
          Liabilities
          LongTermDebt
          StockholdersEquity
          EntityCommonStockSharesOutstanding
        """
        try:
            gaap = facts.get("facts", {}).get("us-gaap", {})
            concept_data = gaap.get(concept, {}).get("units", {}).get(unit, [])
            
            # Filter to 10-K annual filings only (form == "10-K")
            annual = [
                entry for entry in concept_data
                if entry.get("form") == "10-K" and entry.get("val") is not None
            ]
            
            if not annual:
                return None
            
            # Sort by end date descending and return the most recent
            annual.sort(key=lambda x: x.get("end", ""), reverse=True)
            return float(annual[0]["val"])
        except Exception:
            return None

    def _extract_time_series(self, facts: dict, concept: str, unit: str = "USD", n: int = 4) -> list:
        """
        Return last n annual values as [(year, value), ...] sorted oldest→newest.
        Used to compute YoY growth rates.
        """
        try:
            gaap = facts.get("facts", {}).get("us-gaap", {})
            concept_data = gaap.get(concept, {}).get("units", {}).get(unit, [])
            
            annual = [
                entry for entry in concept_data
                if entry.get("form") == "10-K" and entry.get("val") is not None
            ]
            annual.sort(key=lambda x: x.get("end", ""))
            
            # Deduplicate by end-year (keep last filing per year)
            seen_years = {}
            for entry in annual:
                year = entry["end"][:4]
                seen_years[year] = float(entry["val"])
            
            items = sorted(seen_years.items())[-n:]
            return items
        except Exception:
            return []

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #
    def get_company_financials(self, ticker_or_cik: str) -> dict:
        """
        Returns a flat dict of financial features ready for ML inference.
        All monetary values are in USD (not billions).
        """
        cik = self._resolve_cik(ticker_or_cik)
        if not cik:
            return {"error": f"CIK not found for '{ticker_or_cik}'"}

        log.info(f"Fetching SEC XBRL facts for CIK {cik}...")
        time.sleep(0.15)  # SEC rate limit: max 10 req/sec
        
        facts = self._fetch_company_facts(cik)
        if not facts:
            return {"error": f"No XBRL data for CIK {cik}"}

        entity_name = facts.get("entityName", ticker_or_cik)

        # ---- Pull raw values ----------------------------------------- #
        revenue = self._extract_latest_annual(facts, "Revenues") or \
                  self._extract_latest_annual(facts, "RevenueFromContractWithCustomerExcludingAssessedTax")
        
        net_income = self._extract_latest_annual(facts, "NetIncomeLoss")
        operating_income = self._extract_latest_annual(facts, "OperatingIncomeLoss")
        total_assets = self._extract_latest_annual(facts, "Assets")
        total_liabilities = self._extract_latest_annual(facts, "Liabilities")
        long_term_debt = self._extract_latest_annual(facts, "LongTermDebt")
        stockholders_equity = self._extract_latest_annual(facts, "StockholdersEquity")
        
        # Headcount is reported as shares but sometimes as employees
        employees = self._extract_latest_annual(
            facts, "EntityNumberOfEmployees", unit="pure"
        )
        
        # ---- Time-series for growth rates ---------------------------- #
        revenue_series = self._extract_time_series(facts, "Revenues", n=4)
        if not revenue_series:
            revenue_series = self._extract_time_series(
                facts, "RevenueFromContractWithCustomerExcludingAssessedTax", n=4
            )
        
        net_income_series = self._extract_time_series(facts, "NetIncomeLoss", n=4)

        # ---- Compute derived features -------------------------------- #
        def yoy_growth(series):
            """YoY growth rate from time-series [(year, val), ...]"""
            if len(series) < 2:
                return None
            prev = series[-2][1]
            curr = series[-1][1]
            if prev == 0:
                return None
            return (curr - prev) / abs(prev)

        revenue_growth_yoy = yoy_growth(revenue_series)
        net_income_growth_yoy = yoy_growth(net_income_series)
        
        profit_margin = (net_income / revenue) if (revenue and net_income is not None) else None
        operating_margin = (operating_income / revenue) if (revenue and operating_income is not None) else None
        debt_to_equity = (long_term_debt / stockholders_equity) if (long_term_debt and stockholders_equity and stockholders_equity != 0) else None
        revenue_per_employee = (revenue / employees) if (revenue and employees) else None
        asset_to_liability = (total_assets / total_liabilities) if (total_assets and total_liabilities and total_liabilities != 0) else None

        return {
            "ticker": ticker_or_cik.upper(),
            "entity_name": entity_name,
            "cik": cik,
            "source": "SEC_EDGAR",
            # Raw financials (USD)
            "revenue_usd": revenue,
            "net_income_usd": net_income,
            "operating_income_usd": operating_income,
            "total_assets_usd": total_assets,
            "total_liabilities_usd": total_liabilities,
            "long_term_debt_usd": long_term_debt,
            "stockholders_equity_usd": stockholders_equity,
            "employees": employees,
            # Derived ML features
            "profit_margin": profit_margin,
            "operating_margin": operating_margin,
            "debt_to_equity": debt_to_equity,
            "revenue_per_employee": revenue_per_employee,
            "asset_to_liability_ratio": asset_to_liability,
            "revenue_growth_yoy": revenue_growth_yoy,
            "net_income_growth_yoy": net_income_growth_yoy,
            # Revenue trend (last 4 years for feature engineering)
            "revenue_series": revenue_series,
        }


if __name__ == "__main__":
    scraper = SECScraper()
    
    for ticker in ["AMZN", "META", "SNAP", "LYFT"]:
        data = scraper.get_company_financials(ticker)
        print(f"\n{'='*60}")
        print(f"Company: {data.get('entity_name')} ({ticker})")
        print(f"Revenue: ${data.get('revenue_usd', 'N/A'):,.0f}" if data.get('revenue_usd') else "Revenue: N/A")
        print(f"Net Income: ${data.get('net_income_usd', 'N/A'):,.0f}" if data.get('net_income_usd') else "Net Income: N/A")
        print(f"Profit Margin: {data.get('profit_margin', 'N/A'):.2%}" if data.get('profit_margin') is not None else "Profit Margin: N/A")
        print(f"Debt/Equity: {data.get('debt_to_equity', 'N/A'):.2f}" if data.get('debt_to_equity') is not None else "Debt/Equity: N/A")
        print(f"Revenue Growth YoY: {data.get('revenue_growth_yoy', 'N/A'):.2%}" if data.get('revenue_growth_yoy') is not None else "Revenue Growth: N/A")
        time.sleep(0.5)  # Be polite to SEC servers