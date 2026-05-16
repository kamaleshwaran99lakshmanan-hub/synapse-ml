# Synapse ML — Real Layoff Risk Predictor

## What's different from the fake version

| Old (fake)                          | New (real)                              |
|-------------------------------------|-----------------------------------------|
| `np.random.uniform()` training data | Real SEC EDGAR 10-K filings             |
| Wikipedia revenue table             | XBRL financial facts API                |
| Math formula as "layoff label"      | Confirmed events from layoffs.fyi       |
| 3 features (rev, profit, employees) | 13 real financial ratios                |
| No generalization possible          | Cross-validated on held-out companies   |

## Run Order

### 1. Install dependencies
```bash
cd synapse-ml
pip install -r requirements.txt
```

### 2. Build the real dataset (takes ~5 min, hits SEC EDGAR)
```bash
python -m features.build_dataset
# Output: data/processed/training_dataset.csv
```

### 3. Train the real model
```bash
python models/train_model.py
# Output: models/layoff_xgboost_model.json
#         models/feature_scaler.pkl
```

### 4. Start the API
```bash
uvicorn api.main:app --reload --port 8000
```

### 5. Test it
```bash
# Single ticker
curl http://localhost:8000/predict/SNAP

# Batch
curl "http://localhost:8000/batch?tickers=AMZN,META,SNAP,LYFT,PTON"
```

## Data Sources

| Source           | What it provides             | URL                                   |
|------------------|------------------------------|---------------------------------------|
| SEC EDGAR        | Revenue, profit, debt, headcount | https://data.sec.gov/api/xbrl/       |
| layoffs.fyi      | Confirmed layoff events (labels) | https://layoffs.fyi                  |
| MCA/Tofler       | Indian companies (future)    | https://www.tofler.in                 |
| Ambitionbox      | Employee sentiment (future)  | https://www.ambitionbox.com           |

## Important Notes

- SEC EDGAR is rate-limited to ~10 req/sec. The scrapers have delays built in.
- SEC only has US public companies. For Indian companies (Zoho, Freshworks),
  MCA filings need a different scraper (see `scrapers/mca_scraper.py` — TODO).
- Private companies (Stripe, Klarna) are skipped — no public financial data.
- The dataset will be small initially (~30-50 companies). Expect CV AUC ~0.70-0.80.
  Quality beats quantity with real data.

## Adding Indian Companies (MCA)

Indian public companies file with BSE/NSE and MCA. Two paths:
1. **BSE API**: https://api.bseindia.com/BseIndiaAPI/api/Fundamental/
2. **Screener.in** (India's SEC equivalent, free tier): https://www.screener.in/api/

These require additional scrapers (planned for v2).