# src/twitter/config.py
import os
from dotenv import load_dotenv

load_dotenv()

TWITTER_CONFIG = {
    "bearer_token": os.getenv("TWITTER_BEARER_TOKEN"),
    "api_key": os.getenv("TWITTER_API_KEY"),
    "api_secret": os.getenv("TWITTER_API_SECRET"),
    "access_token": os.getenv("TWITTER_ACCESS_TOKEN"),
    "access_secret": os.getenv("TWITTER_ACCESS_SECRET"),
}

SEARCH_PARAMS = {
    "query": "(stocks OR trading OR SPY OR $SPY) lang:en -is:retweet",
    "max_results": 100,
    "tweet_fields": ["created_at", "public_metrics", "text"],
    "start_time": None,  # Will be set dynamically
    "end_time": None,
}