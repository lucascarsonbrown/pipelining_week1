"""One-time setup: creates the BigQuery dataset."""
from google.cloud import bigquery

import config


def create_bigquery_dataset():
    client = bigquery.Client(project=config.GCP_PROJECT_ID)
    dataset_ref = f"{config.GCP_PROJECT_ID}.{config.BQ_DATASET}"
    dataset = bigquery.Dataset(dataset_ref)
    dataset.location = "US"
    client.create_dataset(dataset, exists_ok=True)
    print(f"Dataset ready: {dataset_ref}")


if __name__ == "__main__":
    print("Setting up BigQuery...")
    create_bigquery_dataset()
    print("GCP setup complete!")
