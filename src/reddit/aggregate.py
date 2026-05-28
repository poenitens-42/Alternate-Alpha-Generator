import json
from datetime import datetime
from collections import defaultdict
import numpy as np
import pytz

from src.utils.file_utils import save_json
from src.utils.logger import logger


def aggregate_daily_sentiment(input_path: str, output_path: str):
    """
    Aggregate Reddit sentiment into daily signals.
    Uses US Eastern Time for date alignment with market data.
    """

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    daily = defaultdict(list)

    # Define timezones
    utc_tz = pytz.UTC
    eastern_tz = pytz.timezone('US/Eastern')

    # Group by US Eastern date (market timezone)
    for d in data:
        ts = d.get("created_utc")
        if ts is None:
            continue

        try:
            # Convert UTC timestamp to US Eastern date
            utc_time = datetime.utcfromtimestamp(ts)
            utc_time = utc_tz.localize(utc_time)  # Make timezone aware
            eastern_time = utc_time.astimezone(eastern_tz)
            date = eastern_time.date().isoformat()  # US Eastern market date
            daily[date].append(d)
        except Exception as e:
            logger.warning(f"Failed to process timestamp {ts}: {e}")
            continue

    aggregated = []

    for date, posts in sorted(daily.items()):
        if len(posts) == 0:
            continue
            
        sentiments = np.array([p["sentiment_score"] for p in posts])
        weights = np.array([1 + p["score"] + p["comments"] for p in posts])

        # Handle cases with all zero weights
        if weights.sum() == 0:
            weights = np.ones_like(weights)

        aggregated.append({
            "date": date,
            "mean_sentiment": float(sentiments.mean()),
            "weighted_sentiment": float(
                np.average(sentiments, weights=weights)
            ),
            "sentiment_std": float(sentiments.std()),
            "post_count": len(posts),
        })

    save_json(aggregated, output_path)
    logger.info(f"Daily sentiment saved to {output_path}")
    
    # Log summary
    logger.info(f"Aggregated {len(data)} posts into {len(aggregated)} trading days")
    if aggregated:
        logger.info(f"Date range: {aggregated[0]['date']} to {aggregated[-1]['date']}")

    return aggregated