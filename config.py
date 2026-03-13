"""
Pipeline Configuration
======================
Edit this file to configure the pipeline for your scraped listings data.

The pipeline takes a CSV or XLSX of scraped listings and runs:
  1. Vision OCR on listing images
  2. Gemini LLM to score match_confidence (0.0 - 1.0)
  3. Uploads results to BigQuery

QUICKSTART:
  1. Fill in .env with your GCP credentials  (see .env.example)
  2. Set INPUT_FILE below to your scraped listings file
  3. Set COLUMN MAPPINGS to match your file's column headers
  4. python setup_gcp.py          # one-time: creates BQ dataset + tables + GCS bucket
  5. python upload_scraped.py     # upload raw listings to BigQuery
  6. python enrich_scraped.py     # run OCR + LLM confidence scoring
"""
import os
from dotenv import load_dotenv

load_dotenv()


# =============================================================================
# INPUT FILE
# =============================================================================
# Path to your scraped listings file. Supports .csv and .xlsx.
INPUT_FILE = "scraped_listings.xlsx"

# For XLSX files: sheet name or 0-based index. Ignored for CSV files.
INPUT_FILE_SHEET = 0


# =============================================================================
# COLUMN MAPPINGS
# =============================================================================
# Map your file's column headers to what the pipeline expects.
# Change the RIGHT-HAND SIDE values to match your actual column names.
#
# Required:
COL_SEARCH_TERM = "search_term"     # What was searched (e.g. "2018 Honda Civic Brake Pad")
COL_TITLE       = "product_title"   # Listing title text
COL_IMAGE_URL   = "image_url"       # URL to the listing's main image
COL_SELLER      = "seller_name"     # Seller identifier — used to deduplicate images per query
COL_LISTING_URL = "product_url"     # URL of the listing page
COL_PRICE       = "price"           # Listing price (dollar signs / commas are auto-cleaned)

# Optional (leave as-is if the column does not exist in your file):
COL_ORIGINAL_PRICE = "original_price"  # Crossed-out / original price


# =============================================================================
# GCP SETTINGS
# =============================================================================
# Set these in your .env file (see .env.example):
#   GCP_PROJECT_ID              — your Google Cloud project ID
#   GCP_LOCATION                — region, e.g. us-central1
#   GOOGLE_APPLICATION_CREDENTIALS — path to a service-account key JSON
#                                    (not needed if using `gcloud auth application-default login`)
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GCP_LOCATION   = os.getenv("GCP_LOCATION", "us-central1")

_creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
if _creds_path and not os.path.exists(_creds_path):
    # Avoid setting a path that does not exist — ADC will be used instead.
    os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)


# =============================================================================
# BIGQUERY SETTINGS
# =============================================================================
BQ_DATASET          = "listing_pipeline"   # BigQuery dataset name
BQ_TABLE_SCRAPED    = "scraped_listings"   # Raw listings  (upload_scraped.py)
BQ_TABLE_ENRICHMENT = "enrichment_results" # OCR + LLM scores (enrich_scraped.py)


# =============================================================================
# LLM SETTINGS
# =============================================================================
LLM_MODEL       = "gemini-2.0-flash"  # Gemini model to use for confidence scoring
LLM_WORKERS     = 3                   # Concurrent LLM calls (keep low to avoid rate limits)
LLM_MAX_RETRIES = 5                   # Retry attempts when rate-limited


# =============================================================================
# OCR SETTINGS
# =============================================================================
OCR_WORKERS          = 10  # Concurrent Vision API calls
OCR_IMAGES_PER_QUERY = 5   # Number of images to OCR per search term


# =============================================================================
# PIPELINE SETTINGS
# =============================================================================
LISTINGS_PER_QUERY = 10   # Top N listings per search term considered by the pipeline
BATCH_SIZE         = 100  # Records written per checkpoint save

DATA_DIR       = os.path.join(os.path.dirname(__file__), "data")
CHECKPOINT_DIR = os.path.join(DATA_DIR, "checkpoints")


