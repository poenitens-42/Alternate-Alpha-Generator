import json
import numpy as np
import pandas as pd 
from datetime import datetime

from src.utils.file_utils import save_json
from src.utils.logger import logger


def rolling_zscore(series, window):
    """Calculate rolling z-score with proper NaN handling"""
    z = [None] * len(series)
    for i in range(len(series)):
        if i < window:
            continue
        window_slice = series[i - window:i]
        mean = np.nanmean(window_slice)
        std = np.nanstd(window_slice)
        if std == 0 or np.isnan(std) or np.isnan(series[i]):
            z[i] = 0.0
        else:
            z[i] = (series[i] - mean) / std
    return z


def build_reddit_alpha(
    input_path: str,
    output_path: str,
    window: int = 7,
    min_posts: int = 5,
    lag: int = 1,
):
    """
    Build lagged rolling-Z Reddit sentiment alpha.
    Alpha on date T is designed to predict returns on date T+1.
    """

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Sort by date ascending
    data = sorted(data, key=lambda x: x["date"])

    sentiments = []
    dates = []

    for d in data:
        if d["post_count"] < min_posts:
            sentiments.append(None)
        else:
            sentiments.append(d["weighted_sentiment"])
        dates.append(d["date"])

    # Replace None with NaN for numpy
    sentiments_np = np.array(
        [np.nan if v is None else v for v in sentiments],
        dtype=float,
    )

    # Forward-fill NaNs for stability
    for i in range(1, len(sentiments_np)):
        if np.isnan(sentiments_np[i]):
            sentiments_np[i] = sentiments_np[i - 1]

    # Handle remaining NaNs at start
    if np.isnan(sentiments_np[0]):
        sentiments_np[0] = 0.0

    zscores = rolling_zscore(sentiments_np.tolist(), window)

    # Create alpha: shift FORWARD by lag days
    # alpha[T] should predict returns[T+lag]
    alpha = []
    for i in range(len(zscores)):
        # If we want alpha on date i to predict date i+lag
        # We need to shift alpha values backward in the series
        if i + lag < len(zscores):
            alpha.append(zscores[i])  # alpha[i] uses sentiment from date i
        else:
            alpha.append(None)  # No forward return available

    result = []
    for i in range(len(dates)):
        if alpha[i] is None or np.isnan(alpha[i]):
            continue
        result.append({
            "date": dates[i],
            "reddit_alpha": float(alpha[i]),
            "raw_sentiment": float(sentiments_np[i]),
            "z_score": float(zscores[i]) if zscores[i] is not None else None,
        })

    save_json(result, output_path)
    logger.info(f"Reddit alpha saved → {output_path}")
    logger.info(f"Alpha date range: {result[0]['date']} to {result[-1]['date']}")
    logger.info(f"Alpha values range: {min(r['reddit_alpha'] for r in result):.2f} to {max(r['reddit_alpha'] for r in result):.2f}")

    return result


def load_reddit_alpha(path="data/alpha/reddit_alpha.json"):
    """Load and prepare alpha for backtesting"""
    df = pd.read_json(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    
    # Alpha on date T predicts returns on date T+1
    # So we align alpha[T] with returns[T+1]
    df["alpha_date"] = df["date"]
    df["prediction_date"] = df["date"] + pd.Timedelta(days=1)
    
    return df