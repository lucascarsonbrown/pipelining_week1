import json
import os
import time

import config


def ensure_dirs():
    os.makedirs(config.DATA_DIR, exist_ok=True)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)


def load_checkpoint(name):
    path = os.path.join(config.CHECKPOINT_DIR, f"{name}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"completed": [], "last_index": 0}


def save_checkpoint(name, data):
    path = os.path.join(config.CHECKPOINT_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(data, f)


def save_batch(step_name, batch_id, records):
    dir_path = os.path.join(config.DATA_DIR, step_name)
    os.makedirs(dir_path, exist_ok=True)
    path = os.path.join(dir_path, f"batch_{batch_id}.json")
    with open(path, "w") as f:
        json.dump(records, f, indent=2)


def load_all_batches(step_name):
    dir_path = os.path.join(config.DATA_DIR, step_name)
    all_records = []
    if not os.path.exists(dir_path):
        return all_records
    for filename in sorted(os.listdir(dir_path)):
        if filename.endswith(".json"):
            with open(os.path.join(dir_path, filename)) as f:
                all_records.extend(json.load(f))
    return all_records
