"""
layoffs_fyi_scraper.py
-----------------------
Provides two classes consumed by features/build_dataset.py:

  LayoffsScraper        – fetches live layoff events from layoffs.fyi
                          (label = 1 rows for the ML pipeline)

  NegativeSampleBuilder – returns a curated list of financially-stable
                          companies as negative training examples (label = 0)

Scraping strategy
-----------------
layoffs.fyi embeds its data in an Airtable iframe. Airtable's embed JS
makes an internal XHR to fetch grid row data as JSON. We intercept that
network response directly rather than scraping the rendered DOM, which
avoids both the networkidle timeout (too many 3rd-party trackers on the
outer page) and Airtable's bot-detection fingerprinting.

Fallback: if the network interceptor captures nothing within MAX_WAIT_S
seconds (e.g. Airtable rotated its internal API paths), we fall back to
DOM scraping of the iframe's rendered rows.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import pandas as pd
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AIRTABLE_EMBED_URL = (
    "https://airtable.com/embed/app1PaujS9zxVGUZ4/shroKsHx3SdYYOzeh"
    "?backgroundColor=green&viewControls=on"
)

# Substring patterns that identify Airtable's internal data XHR
_AIRTABLE_DATA_PATTERNS = ("viewData", "/v0.3/view/", "/v0.3/application/")

MAX_WAIT_S = 25          # seconds to wait for the XHR interceptor
POLL_INTERVAL_S = 0.5    # polling cadence while waiting


# ---------------------------------------------------------------------------
# Internal Playwright helpers
# ---------------------------------------------------------------------------

def _make_browser_context(playwright):
    """
    Launch Chromium with the flags and JS patches needed to avoid
    Airtable's bot-detection fingerprinting.
    """
    browser = playwright.chromium.launch(
        headless=True,
        args=[
            "--disable-blink-features=AutomationControlled",
            "--no-sandbox",
            "--disable-setuid-sandbox",
            "--disable-dev-shm-usage",
            "--disable-infobars",
            "--disable-background-timer-throttling",
            "--disable-backgrounding-occluded-windows",
            "--disable-renderer-backgrounding",
        ],
    )
    context = browser.new_context(
        user_agent=(
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        viewport={"width": 1920, "height": 1080},
        java_script_enabled=True,
        bypass_csp=True,          # our init script must run before Airtable's JS
    )
    # Patch navigator.webdriver in EVERY frame context (including the
    # Airtable iframe), not just the top-level page.
    context.add_init_script("""
        Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
        Object.defineProperty(navigator, 'plugins',   { get: () => [1, 2, 3, 4, 5] });
        Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
        window.chrome = {
            runtime: {}, loadTimes: function(){}, csi: function(){}, app: {}
        };
    """)
    return browser, context


def _fetch_raw_airtable_json() -> dict:
    """
    Open the Airtable embed URL in a headless browser and capture the
    internal JSON data response that the Airtable grid JS fires on load.

    Returns the first captured JSON body that contains row data, or {}
    if nothing was intercepted before MAX_WAIT_S.
    """
    intercepted: dict = {}

    with sync_playwright() as p:
        browser, context = _make_browser_context(p)
        page = context.new_page()

        def _on_response(response):
            if intercepted:            # already got what we need
                return
            url = response.url
            if not any(pat in url for pat in _AIRTABLE_DATA_PATTERNS):
                return
            if response.status != 200:
                return
            if "application/json" not in response.headers.get("content-type", ""):
                return
            try:
                body = response.json()
                if any(k in body for k in ("data", "rows", "records")):
                    intercepted.update(body)
                    log.info("[scraper] Captured Airtable XHR: %s", url)
            except Exception:
                pass

        page.on("response", _on_response)

        log.info("[scraper] Navigating to Airtable embed …")
        try:
            page.goto(AIRTABLE_EMBED_URL, wait_until="domcontentloaded", timeout=30_000)
        except PlaywrightTimeout:
            log.warning("[scraper] domcontentloaded timed out — continuing anyway")

        elapsed = 0.0
        while not intercepted and elapsed < MAX_WAIT_S:
            time.sleep(POLL_INTERVAL_S)
            elapsed += POLL_INTERVAL_S

        if not intercepted:
            log.warning("[scraper] XHR interceptor empty — trying DOM fallback")
            intercepted = _dom_fallback(page)

        browser.close()

    return intercepted


def _dom_fallback(page) -> dict:
    """
    Last-resort: scrape rendered row elements from the Airtable iframe DOM.
    Returns a dict shaped like {"records": [{"Company": ..., ...}, ...]}
    so the caller can treat it uniformly.
    """
    try:
        page.wait_for_selector('iframe[src*="airtable.com"]', timeout=20_000)
        frame = page.frame_locator('iframe[src*="airtable.com"]')
        row_loc = frame.locator('.dataRow, [data-rowindex], .rowContainer')
        row_loc.first.wait_for(state="visible", timeout=25_000)

        rows = row_loc.all()
        log.info("[scraper] DOM fallback found %d row elements", len(rows))

        records = []
        for row in rows:
            cells = row.locator('[data-columnindex], .cell, td').all_inner_texts()
            records.append({"raw_cells": cells})

        return {"records": records}

    except PlaywrightTimeout:
        log.error("[scraper] DOM fallback also timed out.")
        return {}


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_airtable_response(data: dict) -> pd.DataFrame:
    """
    Normalise the various JSON shapes that Airtable's internal API can
    return into a flat DataFrame.

    Known shapes
    ────────────
    Shape A  { "data": { "rows": [...], "columns": [...] } }
    Shape B  { "rows": [...], "columns": [...] }
    Shape C  { "records": [{ "id": ..., "fields": {...} }, ...] }
    """
    if not data:
        log.error("[scraper] Empty data dict — returning empty DataFrame")
        return pd.DataFrame()

    log.debug("[scraper] Top-level keys: %s", list(data.keys()))

    # Shape A
    if "data" in data and isinstance(data["data"], dict):
        inner = data["data"]
        rows    = inner.get("rows") or inner.get("records", [])
        columns = inner.get("columns") or inner.get("fields", [])
        return _rows_to_df(rows, columns)

    # Shape B
    if "rows" in data:
        return _rows_to_df(data["rows"], data.get("columns", []))

    # Shape C  (also used by the DOM fallback)
    if "records" in data:
        flat = [r.get("fields", r) for r in data["records"]]
        return pd.DataFrame(flat)

    log.error("[scraper] Unrecognised response shape: %s", list(data.keys()))
    return pd.DataFrame()


def _rows_to_df(rows: list, columns: list) -> pd.DataFrame:
    """Convert parallel row/column arrays into a named DataFrame."""
    if not rows:
        log.warning("[scraper] 0 rows returned from Airtable")
        return pd.DataFrame()

    col_names: list[str] = []
    for c in columns:
        if isinstance(c, str):
            col_names.append(c)
        elif isinstance(c, dict):
            col_names.append(c.get("name") or c.get("id") or str(c))

    parsed_rows = []
    for row in rows:
        if isinstance(row, dict):
            cell = (
                row.get("cellValuesByColumnId")
                or row.get("cells")
                or row
            )
            parsed_rows.append(cell)
        else:
            parsed_rows.append(row)

    if col_names and parsed_rows and isinstance(parsed_rows[0], list):
        df = pd.DataFrame(parsed_rows, columns=col_names[: len(parsed_rows[0])])
    else:
        df = pd.DataFrame(parsed_rows)

    log.info("[scraper] Parsed %d rows × %d columns", len(df), len(df.columns))
    return df


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename whatever column Airtable called 'Company' (case-insensitive)
    to the canonical 'company' name that build_dataset.py expects.
    Also lower-cases all column names for consistency.
    """
    df.columns = [str(c).strip().lower() for c in df.columns]

    # Common Airtable column aliases → canonical name
    rename_map: dict[str, str] = {}
    for col in df.columns:
        if col in ("company name", "company_name", "organization"):
            rename_map[col] = "company"
        elif col in ("# laid off", "num_laid_off", "employees_laid_off", "layoffs"):
            rename_map[col] = "num_laid_off"
        elif col in ("date", "date added", "date_added"):
            rename_map[col] = "date"
        elif col in ("industry", "sector"):
            rename_map[col] = "industry"
        elif col in ("country", "location country"):
            rename_map[col] = "country"

    if rename_map:
        df = df.rename(columns=rename_map)

    return df


# ---------------------------------------------------------------------------
# Public Classes
# ---------------------------------------------------------------------------

class LayoffsScraper:
    """
    Fetches live layoff events from layoffs.fyi and returns them as a
    DataFrame with at least a 'company' column.

    Usage (called by DatasetBuilder in build_dataset.py):
        scraper = LayoffsScraper()
        df = scraper.fetch_layoff_labels()
        # df.columns includes: company, num_laid_off, date, industry, country, …
    """

    def fetch_layoff_labels(self) -> pd.DataFrame:
        """
        Scrape layoffs.fyi and return a normalised DataFrame.

        Returns
        -------
        pd.DataFrame
            Columns guaranteed: 'company'
            Columns present when available: 'num_laid_off', 'date',
            'industry', 'country', 'percentage', 'stage', 'source'
        """
        log.info("[LayoffsScraper] Starting layoffs.fyi scrape …")

        raw = _fetch_raw_airtable_json()
        df  = _parse_airtable_response(raw)

        if df.empty:
            log.error(
                "[LayoffsScraper] Scrape returned 0 rows. "
                "Check AIRTABLE_EMBED_URL or bot-detection countermeasures."
            )
            return df

        df = _normalise_columns(df)

        if "company" not in df.columns:
            log.error(
                "[LayoffsScraper] 'company' column not found after normalisation. "
                "Actual columns: %s",
                df.columns.tolist(),
            )
            return pd.DataFrame()

        # Drop rows with no company name
        df = df[df["company"].notna() & (df["company"] != "")]
        log.info("[LayoffsScraper] Returning %d layoff records.", len(df))
        return df.reset_index(drop=True)


class NegativeSampleBuilder:
    """
    Produces a list of dicts representing financially-stable companies
    (label = 0) for use as negative training examples.

    The list is manually curated from well-known, consistently profitable
    public companies across diverse sectors.  No scraping required.

    Usage (called by DatasetBuilder in build_dataset.py):
        neg = NegativeSampleBuilder()
        stable_list = neg.build_negative_samples()
        # Returns: [{"company": "Apple", "ticker": "AAPL", "layoff_occurred": 0}, ...]
    """

    # Tickers chosen for consistent revenue growth, no large-scale layoffs
    # in recent years, and strong balance sheets — i.e. genuine label=0 signal.
    _STABLE_COMPANIES = [
    {"company":"Apple","ticker":"AAPL"},
    {"company":"Microsoft","ticker":"MSFT"},
    {"company":"Nvidia","ticker":"NVDA"},
    {"company":"Broadcom","ticker":"AVGO"},
    {"company":"Oracle","ticker":"ORCL"},
    {"company":"Adobe","ticker":"ADBE"},
    {"company":"Intuit","ticker":"INTU"},
    {"company":"ServiceNow","ticker":"NOW"},
    {"company":"Workday","ticker":"WDAY"},
    {"company":"Veeva","ticker":"VEEV"},
    {"company":"Datadog","ticker":"DDOG"},
    {"company":"CrowdStrike","ticker":"CRWD"},
    {"company":"ADP","ticker":"ADP"},
    {"company":"Roper","ticker":"ROP"},
    {"company":"Accenture","ticker":"ACN"},
    {"company":"Cognizant","ticker":"CTSH"},
    {"company":"TSMC","ticker":"TSM"},
    {"company":"ASML","ticker":"ASML"},
    {"company":"Visa","ticker":"V"},
    {"company":"Mastercard","ticker":"MA"},
    {"company":"UnitedHealth","ticker":"UNH"},
    {"company":"Costco","ticker":"COST"},
    {"company":"McDonalds","ticker":"MCD"},
    {"company":"PepsiCo","ticker":"PEP"},
    {"company":"Coca-Cola","ticker":"KO"},
    {"company":"Procter & Gamble","ticker":"PG"},
    {"company":"Johnson & Johnson","ticker":"JNJ"},
    {"company":"AbbVie","ticker":"ABBV"},
    {"company":"Merck","ticker":"MRK"},
    {"company":"Thermo Fisher","ticker":"TMO"},
]

    def build_negative_samples(self) -> list[dict[str, Any]]:
        """
        Return the stable-company list with 'layoff_occurred' = 0 attached.

        Returns
        -------
        list[dict]
            Each dict: {"company": str, "ticker": str, "layoff_occurred": 0}
            Compatible with DatasetBuilder._fetch_financials_batch().
        """
        samples = [
            {**entry, "layoff_occurred": 0}
            for entry in self._STABLE_COMPANIES
        ]
        log.info(
            "[NegativeSampleBuilder] Returning %d stable-company entries.",
            len(samples),
        )
        return samples