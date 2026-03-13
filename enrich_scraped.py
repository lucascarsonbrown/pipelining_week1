"""
Enrichment pipeline for scraped eBay listings.

1. For each search term, pick the best 5 listing images using title-match scoring
2. Run Vision OCR on those images to extract part numbers/text
3. Run Gemini LLM to compute match_confidence
4. Write results to BigQuery

Usage:
    python enrich_scraped.py --test 10    # test with 10 search terms
    python enrich_scraped.py              # run all search terms
    python enrich_scraped.py --ocr-only     # just OCR, no LLM
    python enrich_scraped.py --llm-only     # just LLM (assumes OCR already done)
    python enrich_scraped.py --retry-zeros  # re-score terms that got 0.0 confidence
"""

import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from google.cloud import bigquery, vision
from google import genai
from google.genai import types

import config
import utils

# Concurrency settings — override in config.py
OCR_WORKERS     = config.OCR_WORKERS
LLM_WORKERS     = config.LLM_WORKERS
LLM_MAX_RETRIES = config.LLM_MAX_RETRIES


# ---------------------------------------------------------------------------
# Image selection
# ---------------------------------------------------------------------------

def score_title_match(search_term, title):
    search_words = set(search_term.lower().split())
    title_words = set(re.sub(r"[^\w\s]", " ", title.lower()).split())
    if not search_words:
        return 0
    return len(search_words & title_words) / len(search_words)


def _upgrade_image_url(url):
    if isinstance(url, str):
        url = url.replace("/s-l140.", "/s-l500.")
        url = url.replace("/s-l225.", "/s-l500.")
        url = url.replace("/s-l300.", "/s-l500.")
    return url


def select_best_images(listings_df, n=5):
    col_term   = config.COL_SEARCH_TERM
    col_img    = config.COL_IMAGE_URL
    col_title  = config.COL_TITLE
    col_seller = config.COL_SELLER

    search_term = listings_df[col_term].iloc[0]
    df = listings_df[
        listings_df[col_img].notna()
        & (listings_df[col_img].str.len() > 10)
    ].copy()
    if df.empty:
        return []

    df[col_img] = df[col_img].apply(_upgrade_image_url)
    df["title_score"] = df[col_title].fillna("").apply(
        lambda t: score_title_match(search_term, t)
    )
    df = df.sort_values("title_score", ascending=False)

    selected = []
    seen_sellers = set()
    for _, row in df.iterrows():
        seller = row.get(col_seller, "")
        if seller in seen_sellers:
            continue
        selected.append({
            "image_url": row[col_img],
            "product_title": row[col_title],
            "title_score": row["title_score"],
        })
        seen_sellers.add(seller)
        if len(selected) >= n:
            break

    # Fill remaining from top scores if needed
    if len(selected) < n:
        for _, row in df.iterrows():
            if any(s["image_url"] == row[col_img] for s in selected):
                continue
            selected.append({
                "image_url": row[col_img],
                "product_title": row[col_title],
                "title_score": row["title_score"],
            })
            if len(selected) >= n:
                break

    return selected


# ---------------------------------------------------------------------------
# Vision OCR
# ---------------------------------------------------------------------------

def run_ocr_on_image(vision_client, image_url):
    image = vision.Image()
    image.source.image_uri = image_url
    response = vision_client.text_detection(image=image)
    if response.error.message:
        raise Exception(response.error.message)
    texts = response.text_annotations
    return texts[0].description.strip() if texts else ""


# ---------------------------------------------------------------------------
# Gemini LLM
# ---------------------------------------------------------------------------

def _parse_llm_response(raw_text):
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass
    brace_match = re.search(r"\{[^{}]*\}", raw_text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass
    conf_match = re.search(r'"match_confidence"\s*:\s*([\d.]+)', raw_text)
    if conf_match:
        return {"match_confidence": float(conf_match.group(1))}
    return {"match_confidence": 0.0}


def build_prompt(search_term, titles, ocr_texts):
    parts = [
        "You are analyzing eBay auto parts listings. Return ONLY a JSON object.",
        f"SEARCH QUERY: {search_term}",
        f"LISTING TITLES ({len(titles)}):",
    ]
    for t in titles[:15]:
        parts.append(f"  - {t}")
    if ocr_texts:
        parts.append("OCR TEXT FROM IMAGES:")
        for t in ocr_texts:
            parts.append(f"  - {t[:200]}")
    parts.append('')
    parts.append('Return: {"match_confidence": <float 0.0-1.0>}')
    parts.append("1.0 = listings clearly match the intended part. 0.0 = no match.")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def load_scraped_data():
    print(f"Loading scraped data from {config.INPUT_FILE}...")
    path = config.INPUT_FILE
    if path.endswith(".xlsx") or path.endswith(".xls"):
        df = pd.read_excel(path, sheet_name=config.INPUT_FILE_SHEET)
    else:
        df = pd.read_csv(path)
    grouped = dict(list(df.groupby(config.COL_SEARCH_TERM)))
    print(f"  {len(df):,} listings across {len(grouped):,} search terms")
    return grouped


def load_checkpoint():
    path = os.path.join(config.DATA_DIR, "enrich_checkpoint.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"completed_ocr": [], "completed_llm": [], "results": {}}


def save_checkpoint(ckpt):
    os.makedirs(config.DATA_DIR, exist_ok=True)
    path = os.path.join(config.DATA_DIR, "enrich_checkpoint.json")
    with open(path, "w") as f:
        json.dump(ckpt, f)


def _ocr_one_term(vision_client, term, listings_df):
    """OCR a single search term's images. Returns (term, ocr_results)."""
    selected = select_best_images(listings_df, n=config.OCR_IMAGES_PER_QUERY)
    ocr_results = []
    for img in selected:
        try:
            ocr_text = run_ocr_on_image(vision_client, img["image_url"])
        except Exception:
            ocr_text = ""
        ocr_results.append({
            "image_url": img["image_url"],
            "ocr_text": ocr_text,
        })
    return term, ocr_results


def run_ocr_phase(grouped, checkpoint, limit=None):
    vision_client = vision.ImageAnnotatorClient()

    search_terms = sorted(grouped.keys())
    if limit:
        search_terms = search_terms[:limit]

    completed = set(checkpoint.get("completed_ocr", []))
    todo = [t for t in search_terms if t not in completed]
    total = len(todo)

    print(f"\n=== OCR Phase: {total} to process ({len(completed)} already done) ===\n")
    if not todo:
        return

    processed = 0
    # Process in batches of 50 for reliable checkpointing
    for batch_start in range(0, total, 50):
        batch = todo[batch_start:batch_start + 50]
        with ThreadPoolExecutor(max_workers=OCR_WORKERS) as pool:
            futures = {
                pool.submit(_ocr_one_term, vision_client, term, grouped[term]): term
                for term in batch
            }
            for future in as_completed(futures):
                term, ocr_results = future.result()
                if term not in checkpoint["results"]:
                    checkpoint["results"][term] = {}
                checkpoint["results"][term]["ocr"] = ocr_results
                checkpoint["completed_ocr"].append(term)
                processed += 1

        save_checkpoint(checkpoint)
        print(f"  [{processed}/{total}] OCR checkpoint saved")

    print(f"\nOCR complete. Processed {processed} search terms.")


def _llm_one_term(genai_client, term, titles, ocr_texts):
    """Score a single search term. Returns (term, match_confidence)."""
    prompt = build_prompt(term, titles, ocr_texts)
    for attempt in range(LLM_MAX_RETRIES):
        try:
            response = genai_client.models.generate_content(
                model=config.LLM_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1,
                    max_output_tokens=256,
                ),
            )
            result = _parse_llm_response(response.text or "")
            return term, float(result.get("match_confidence", 0))
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait = 2 ** attempt * 5  # 5s, 10s, 20s, 40s, 80s
                print(f"  Rate limited on {term}, retrying in {wait}s (attempt {attempt + 1}/{LLM_MAX_RETRIES})")
                time.sleep(wait)
            else:
                print(f"  LLM failed for {term}: {e}")
                return term, 0.0
    print(f"  LLM exhausted retries for {term}")
    return term, 0.0


def run_llm_phase(grouped, checkpoint, limit=None):
    genai_client = genai.Client(
        vertexai=True,
        project=config.GCP_PROJECT_ID,
        location=config.GCP_LOCATION,
    )

    search_terms = sorted(grouped.keys())
    if limit:
        search_terms = search_terms[:limit]

    completed = set(checkpoint.get("completed_llm", []))
    todo = [t for t in search_terms if t not in completed]
    total = len(todo)

    print(f"\n=== LLM Phase: {total} to process ({len(completed)} already done) ===\n")
    if not todo:
        return

    processed = 0
    for batch_start in range(0, total, 50):
        batch = todo[batch_start:batch_start + 50]
        with ThreadPoolExecutor(max_workers=LLM_WORKERS) as pool:
            futures = {}
            for term in batch:
                titles = grouped[term][config.COL_TITLE].dropna().tolist()[:15]
                ocr_data = checkpoint.get("results", {}).get(term, {}).get("ocr", [])
                ocr_texts = [r["ocr_text"] for r in ocr_data if r.get("ocr_text")]
                futures[pool.submit(_llm_one_term, genai_client, term, titles, ocr_texts)] = term

            for future in as_completed(futures):
                term, confidence = future.result()
                if term not in checkpoint["results"]:
                    checkpoint["results"][term] = {}
                checkpoint["results"][term]["match_confidence"] = confidence
                checkpoint["completed_llm"].append(term)
                processed += 1

        save_checkpoint(checkpoint)
        print(f"  [{processed}/{total}] LLM checkpoint saved")

    print(f"\nLLM complete. Processed {processed} search terms.")


def upload_results(checkpoint):
    client = bigquery.Client(project=config.GCP_PROJECT_ID)
    table_ref = f"{config.GCP_PROJECT_ID}.{config.BQ_DATASET}.{config.BQ_TABLE_ENRICHMENT}"

    rows = []
    for term, data in checkpoint.get("results", {}).items():
        ocr_data = data.get("ocr", [])
        image_urls = [r.get("image_url", "") for r in ocr_data if r.get("image_url")]
        rows.append({
            "search_term": term,
            "match_confidence": data.get("match_confidence"),
            "image_urls": " | ".join(image_urls),
            "images_analyzed": len(ocr_data),
        })

    if not rows:
        print("No results to upload.")
        return

    df = pd.DataFrame(rows)
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE", autodetect=True)
    job = client.load_table_from_dataframe(df, table_ref, job_config=job_config)
    job.result()
    print(f"Uploaded {len(rows)} enrichment results to {table_ref}")


def main():
    args = sys.argv[1:]
    limit = None
    for arg in args:
        if arg.startswith("--test"):
            idx = args.index(arg)
            if idx + 1 < len(args) and args[idx + 1].isdigit():
                limit = int(args[idx + 1])
            else:
                limit = 10

    ocr_only = "--ocr-only" in args
    llm_only = "--llm-only" in args
    retry_zeros = "--retry-zeros" in args
    run_both = not ocr_only and not llm_only and not retry_zeros

    utils.ensure_dirs()
    grouped = load_scraped_data()
    checkpoint = load_checkpoint()

    if retry_zeros:
        # Remove 0.0-confidence terms from completed_llm so they get re-processed
        zero_terms = [
            t for t, d in checkpoint.get("results", {}).items()
            if d.get("match_confidence") == 0.0
        ]
        checkpoint["completed_llm"] = [
            t for t in checkpoint.get("completed_llm", []) if t not in set(zero_terms)
        ]
        print(f"Retrying {len(zero_terms)} terms with 0.0 confidence")
        save_checkpoint(checkpoint)
        run_llm_phase(grouped, checkpoint, limit=limit)
    elif run_both or ocr_only:
        run_ocr_phase(grouped, checkpoint, limit=limit)

    if not retry_zeros and (run_both or llm_only):
        run_llm_phase(grouped, checkpoint, limit=limit)

    print("\n=== Uploading results to BigQuery ===")
    upload_results(checkpoint)

    results = checkpoint.get("results", {})
    if results:
        csv_rows = []
        for t, d in results.items():
            ocr_data = d.get("ocr", [])
            image_urls = [r.get("image_url", "") for r in ocr_data if r.get("image_url")]
            csv_rows.append({
                "search_term": t,
                "match_confidence": d.get("match_confidence"),
                "image_urls": " | ".join(image_urls),
                "images_analyzed": len(ocr_data),
            })
        out_path = os.path.join(config.DATA_DIR, "enrichment_results.csv")
        pd.DataFrame(csv_rows).to_csv(out_path, index=False)
        print(f"Also saved to {out_path}")


if __name__ == "__main__":
    main()
