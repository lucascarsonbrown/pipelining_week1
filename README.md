# Listing Confidence Pipeline

Takes a CSV or XLSX of scraped product listings and scores how well each set of results actually matches the original search query — using Google Vision OCR on listing images and Gemini LLM to produce a `match_confidence` score (0.0–1.0) per search term. Results are uploaded to BigQuery.

---

## How it works

```text
your_listings.xlsx
        │
        ▼
  upload_scraped.py ──────────────────► BigQuery: scraped_listings
        │
        ▼
  enrich_scraped.py
    ├── Phase 1: Vision OCR (top 5 images per search term)
    └── Phase 2: Gemini LLM → match_confidence score
        │
        ▼
  BigQuery: enrichment_results
```

---

## Prerequisites

- Python 3.10+
- A Google Cloud project with these APIs enabled:
  - BigQuery
  - Cloud Storage
  - Cloud Vision API
  - Vertex AI API (for Gemini)
- Either a service account key JSON **or** `gcloud auth application-default login`

---

## Setup

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure GCP credentials

Copy `.env.example` to `.env` and fill it in:

```bash
cp .env.example .env
```

```env
# .env
GCP_PROJECT_ID=your-gcp-project-id
GCP_LOCATION=us-central1

# Option A — service account key file:
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json

# Option B — run this instead and leave the line above commented out:
#   gcloud auth application-default login
```

### 3. Configure the pipeline for your data

Open **`config.py`** — this is the only file you need to edit.

**Point it at your file:**

```python
INPUT_FILE       = "your_listings.xlsx"   # path to your CSV or XLSX
INPUT_FILE_SHEET = 0                      # sheet index or name (XLSX only)
```

**Map your column names:**

```python
COL_SEARCH_TERM = "search_term"     # what was searched
COL_TITLE       = "product_title"   # listing title
COL_IMAGE_URL   = "image_url"       # URL to the listing image
COL_SELLER      = "seller_name"     # seller identifier
COL_LISTING_URL = "product_url"     # URL to the listing page
COL_PRICE       = "price"           # listing price
```

Change the right-hand side of each line to match the actual column headers in your file.

**Name your BigQuery resources** (optional — defaults are fine):

```python
BQ_DATASET          = "listing_pipeline"
BQ_TABLE_SCRAPED    = "scraped_listings"
BQ_TABLE_ENRICHMENT = "enrichment_results"
GCS_BUCKET          = "your-project-id-listing-pipeline"
```

### 4. Create GCP resources (one-time)

```bash
python setup_gcp.py
```

This creates the BigQuery dataset + tables and the Cloud Storage bucket.

---

## Running the pipeline

### Upload your listings to BigQuery

```bash
python upload_scraped.py
```

Reads `INPUT_FILE`, cleans price columns, and loads the data into BigQuery.

### Run OCR + LLM confidence scoring

```bash
python enrich_scraped.py
```

For each search term in your file:

1. Selects the best images (deduped by seller, ranked by title match)
2. Runs Vision OCR to extract text from images
3. Sends titles + OCR text to Gemini and gets a `match_confidence` score
4. Uploads results to BigQuery and saves `data/enrichment_results.csv` locally

Progress is checkpointed to `data/enrich_checkpoint.json` — if the run is interrupted, restart the command and it will pick up where it left off.

**Useful flags:**

```bash
python enrich_scraped.py --test 20      # dry run on first 20 search terms
python enrich_scraped.py --ocr-only     # run only the OCR phase
python enrich_scraped.py --llm-only     # run only the LLM phase (OCR already done)
python enrich_scraped.py --retry-zeros  # re-score terms that got 0.0 confidence
```

---

## Output

| Destination | Contents |
| --- | --- |
| BigQuery `scraped_listings` | Raw listings from your input file |
| BigQuery `enrichment_results` | `search_term`, `match_confidence`, `image_urls`, `images_analyzed` |
| `data/enrichment_results.csv` | Same as above, saved locally |
| `data/enrich_checkpoint.json` | Resumable progress checkpoint |

---

## Tuning

All tuneable settings live in `config.py`:

| Setting | Default | Effect |
| --- | --- | --- |
| `OCR_IMAGES_PER_QUERY` | 5 | Images analyzed per search term |
| `OCR_WORKERS` | 10 | Concurrent Vision API calls |
| `LLM_WORKERS` | 3 | Concurrent Gemini calls (keep low) |
| `LLM_MODEL` | `gemini-2.0-flash` | Gemini model used for scoring |
| `LISTINGS_PER_QUERY` | 10 | Top N listings considered |
| `BATCH_SIZE` | 100 | Records per checkpoint save |
