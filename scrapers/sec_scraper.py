"""
SEC EDGAR Scraper (with NLP Sentiment Engine)
---------------------------------------------
Fetches real financial data and performs NLP on SEC filings.

SEC EDGAR APIs used:
  - https://data.sec.gov/submissions/{CIK}.json        → filing metadata & URLs
  - https://data.sec.gov/api/xbrl/companyfacts/{CIK}.json → XBRL financial facts
"""

import requests
import json
import time
import logging
import re
import numpy as np
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "TechHaven-Stability-Platform contact@techhaven.io",
    "Accept-Encoding": "gzip, deflate",
}

TICKER_CIK_URL = "https://www.sec.gov/files/company_tickers.json"
COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"

# The "CEO Panic" Dictionary
PANIC_WORDS = [
    r"\brestructuring\b", 
    r"\bheadwinds\b", 
    r"\bmacroeconomic\b", 
    r"\bdownsizing\b", 
    r"\bcost reduction\b", 
    r"\bseverance\b",
    r"\buncertainty\b"
]


class SECScraper:
    def __init__(self):
        self._ticker_map: dict = {}
        self._company_map: dict = {}   # NEW
        self._aliases = {
        "GOOGLE": "GOOGL",
        "ALPHABET": "GOOGL",

        "FACEBOOK": "META",
        "META": "META",

        "AMAZON": "AMZN",

        "APPLE": "AAPL",

        "MICROSOFT": "MSFT",

        "TESLA": "TSLA",
    }
        self._load_ticker_map()

        self._submissions_cache: dict = {}

    def _load_ticker_map(self):
        try:
            log.info("Loading SEC ticker→CIK map...")
            resp = requests.get(TICKER_CIK_URL, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            raw = resp.json()
            self._ticker_map = {}
            self._company_map = {}

            for v in raw.values():
                ticker = v["ticker"].upper()
                cik = str(v["cik_str"]).zfill(10)
                company = v["title"].upper()

                self._ticker_map[ticker] = cik
                self._company_map[company] = ticker
        except Exception as e:
            log.error(f"Failed to load ticker map: {e}")
    
    def _resolve_cik(self, identifier: str) -> Optional[str]:
        if identifier.isdigit():
            return identifier.zfill(10)
        return self._ticker_map.get(identifier.upper())

    def resolve_company_to_ticker(self, text: str):

     text = text.upper().strip()

    # Already a ticker
     if text in self._ticker_map:
        return text

    # 👇 Check aliases first
     if text in self._aliases:
        return self._aliases[text]

    # Exact SEC company title
     if text in self._company_map:
        return self._company_map[text]

    # Partial SEC company title
     for company_name, ticker in self._company_map.items():
        if text in company_name:
            return ticker

     return None
    # ------------------------------------------------------------------ #
    #  NLP Sentiment Engine
    # ------------------------------------------------------------------ #
    def _fetch_10k_accession_for_year(self, cik: str, target_year: str) -> Optional[str]:
        """Finds the SEC document ID (Accession Number) for a specific year's 10-K."""
        try:
            if cik not in self._submissions_cache:
                url = SUBMISSIONS_URL.format(cik=cik)
                resp = requests.get(url, headers=HEADERS, timeout=10)
                resp.raise_for_status()
                self._submissions_cache[cik] = resp.json()
                time.sleep(0.15) # Rate limit respect
            
            recent = self._submissions_cache[cik].get("filings", {}).get("recent", {})
            if not recent:
                return None
                
            forms = recent.get("form", [])
            dates = recent.get("filingDate", [])
            accessions = recent.get("accessionNumber", [])
            
            for i in range(len(forms)):
                # Match 10-K filings with the fiscal year we are looking for
                if forms[i] == "10-K" and dates[i].startswith(target_year):
                    return accessions[i]
            return None
        except Exception:
            return None

    def _count_panic_words_in_filing(self, cik: str, accession_number: str):
        """
        Downloads the raw SEC filing and analyzes panic words.

        Returns:
        {
            "count": int,
            "words": list[str],
            "frequencies": dict[str, int]
        }
        """
        try:
            # SEC constructs URLs using the CIK (without leading zeros)
            # and accession number without dashes
            clean_cik = str(int(cik))
            clean_accession = accession_number.replace("-", "")

            url = (
                f"https://www.sec.gov/Archives/edgar/data/"
                f"{clean_cik}/{clean_accession}/{accession_number}.txt"
            )

            resp = requests.get(url, headers=HEADERS, timeout=15)
            resp.raise_for_status()

            # Limit processing size
            text_chunk = resp.text[:500000].lower()

            total_panic_score = 0
            panic_word_frequencies = {}

            for word_pattern in PANIC_WORDS:

                matches = re.findall(word_pattern, text_chunk)

                if matches:
                    # Extract readable word
                    clean_word = re.sub(r"\\b", "", word_pattern)

                    panic_word_frequencies[clean_word] = len(matches)

                    total_panic_score += len(matches)

            time.sleep(0.2)

            return {
                "count": total_panic_score,
                "words": list(panic_word_frequencies.keys()),
                "frequencies": panic_word_frequencies,
            }

        except Exception:
            return {
                "count": 0,
                "words": [],
                "frequencies": {},
            }
    # ------------------------------------------------------------------ #
    #  Core fetchers
    # ------------------------------------------------------------------ #
    def _fetch_company_facts(self, cik: str) -> Optional[dict]:
        url = COMPANY_FACTS_URL.format(cik=cik)
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except Exception:
            return None

    def get_company_financials_historical(self, ticker_or_cik: str) -> list[dict]:
        cik = self._resolve_cik(ticker_or_cik)
        if not cik:
            return [{"error": f"CIK not found for '{ticker_or_cik}'"}]

        log.info(f"Fetching historical SEC XBRL & Text facts for CIK {cik}...")
        time.sleep(0.15)

        facts = self._fetch_company_facts(cik)
        if not facts:
            return [{"error": f"No XBRL data for CIK {cik}"}]

        entity_name = facts.get("entityName", ticker_or_cik)

        def get_series(concept, unit="USD", n=8):
            try:
                gaap = facts.get("facts", {}).get("us-gaap", {})
                concept_data = gaap.get(concept, {}).get("units", {}).get(unit, [])
                annual = [e for e in concept_data if e.get("form") == "10-K" and e.get("val") is not None]
                annual.sort(key=lambda x: x.get("end", ""))
                seen = {}
                for e in annual:
                    year = e["end"][:4]
                    seen[year] = float(e["val"])
                return dict(sorted(seen.items())[-n:])
            except Exception:
                return {}

        # Merge both XBRL tags so we don't lose years when companies switch accounting standards
        revenue_by_year = get_series("Revenues")
        revenue_by_year.update(get_series("RevenueFromContractWithCustomerExcludingAssessedTax"))
        net_income_by_year = get_series("NetIncomeLoss")
        op_income_by_year = get_series("OperatingIncomeLoss")
        assets_by_year = get_series("Assets")
        liabilities_by_year = get_series("Liabilities")
        ltd_by_year = get_series("LongTermDebt")
        equity_by_year = get_series("StockholdersEquity")
        employees_by_year = get_series("EntityNumberOfEmployees", unit="pure")

        all_years = sorted(revenue_by_year.keys())

        def safe_div(a, b):
            try:
                return float(a) / float(b) if b and b != 0 else None
            except (TypeError, ZeroDivisionError):
                return None

        rows = []
        rev_years = sorted(revenue_by_year.keys())

        for i, year in enumerate(all_years):
            rev = revenue_by_year.get(year)
            if not rev: continue

            # --- NLP Feature Injection ---
            # Automatically find the document and count panic words for this specific fiscal year
            accession_number = self._fetch_10k_accession_for_year(cik, year)
            panic_result = {
                "count": 0,
                "words": [],
                "frequencies": {},
            }

            if accession_number:
                panic_result = self._count_panic_words_in_filing(
                    cik,
                    accession_number
                    )

            ni = net_income_by_year.get(year)
            oi = op_income_by_year.get(year)
            assets = assets_by_year.get(year)
            liab = liabilities_by_year.get(year)
            ltd = ltd_by_year.get(year)
            equity = equity_by_year.get(year)
            emp = employees_by_year.get(year)

            prev_rev = revenue_by_year.get(rev_years[i - 1]) if i > 0 else None
            prev_ni = net_income_by_year.get(rev_years[i - 1]) if i > 0 else None

            rev_cagr = None
            if i >= 3:
                base_year = rev_years[i - 3]
                base_rev = revenue_by_year.get(base_year)
                if base_rev and base_rev > 0:
                    rev_cagr = (rev / base_rev) ** (1 / 3) - 1

            rows.append({
                "ticker": ticker_or_cik.upper(),
                "entity_name": entity_name,
                "cik": cik,
                "fiscal_year": year,
                "source": "SEC_EDGAR",
                "revenue_usd": rev,
                "net_income_usd": ni,
                "operating_income_usd": oi,
                "total_assets_usd": assets,
                "total_liabilities_usd": liab,
                "long_term_debt_usd": ltd,
                "stockholders_equity_usd": equity,
                "employees": emp,
                "profit_margin": safe_div(ni, rev),
                "operating_margin": safe_div(oi, rev),
                "debt_to_equity": safe_div(ltd, equity),
                "debt_to_assets": safe_div(ltd, assets),
                "asset_to_liability": safe_div(assets, liab),
                "equity_ratio": safe_div(equity, assets),
                "revenue_per_employee": safe_div(rev, emp),
                "net_income_per_employee": safe_div(ni, emp),
                "revenue_growth_yoy": safe_div((rev - prev_rev), abs(prev_rev)) if prev_rev else None,
                "net_income_growth_yoy": safe_div((ni - prev_ni), abs(prev_ni)) if (ni and prev_ni) else None,
                "revenue_cagr_3yr": rev_cagr,
                "log_revenue": float(np.log1p(rev)) if rev and rev > 0 else None,
                "log_employees": float(np.log1p(emp)) if emp and emp > 0 else None,
                "panic_word_count": panic_result["count"],
                "panic_words": panic_result["words"],
                "panic_word_frequencies": panic_result["frequencies"],
            })

        return rows

if __name__ == "__main__":
    scraper = SECScraper()

    data = scraper.get_company_financials_historical("META")

    print(data[-1])