import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

DATA_RAW = os.path.join(ROOT, "Results", "raw", "reddit")
DATA_PROCESSED = os.path.join(ROOT, "Results", "processed", "reddit")
DATA_AGGREGATED = "data/aggregated/reddit_daily.json"

DEFAULT_SUBREDDIT = "wallstreetbets"
DEFAULT_LIMIT = 200
