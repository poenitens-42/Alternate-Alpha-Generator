import os
import json
from datetime import datetime

def ensure_folder(path: str):
    os.makedirs(path, exist_ok=True)

def timestamped_filename(prefix: str, ext: str = "json"):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}.{ext}"

def save_json(data, path: str):
    ensure_folder(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

def load_json(path: str):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None
