"""Upload scraped listings file (CSV or XLSX) to BigQuery."""
import pandas as pd
from google.cloud import bigquery

import config


def upload():
    client = bigquery.Client(project=config.GCP_PROJECT_ID)
    table_ref = f"{config.GCP_PROJECT_ID}.{config.BQ_DATASET}.{config.BQ_TABLE_SCRAPED}"

    print(f"Reading {config.INPUT_FILE}...")
    path = config.INPUT_FILE
    if path.endswith(".xlsx") or path.endswith(".xls"):
        df = pd.read_excel(path, sheet_name=config.INPUT_FILE_SHEET)
    else:
        df = pd.read_csv(path)
    print(f"Loaded {len(df):,} rows")

    # Clean price columns — strip $ and commas, convert to float
    for col in [config.COL_PRICE, config.COL_ORIGINAL_PRICE]:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace("$", "", regex=False)
                .str.replace(",", "", regex=False)
                .str.strip()
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"Uploading to {table_ref}...")
    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE",
        autodetect=True,
    )
    job = client.load_table_from_dataframe(df, table_ref, job_config=job_config)
    job.result()

    table = client.get_table(table_ref)
    print(f"Uploaded {table.num_rows:,} rows to {table_ref}")


if __name__ == "__main__":
    upload()
