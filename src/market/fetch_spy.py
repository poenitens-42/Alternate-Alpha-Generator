# src/market/fetch_spy.py
import numpy as np
import yfinance as yf
import pandas as pd
from pathlib import Path
from src.market.returns import add_returns, save_market_data_with_nan_handling
from datetime import datetime
from src.utils.logger import logger

DATA_DIR = Path("data/market")
DATA_DIR.mkdir(parents=True, exist_ok=True)
SPY_PATH = DATA_DIR / "spy.csv"


def fetch_spy(start="2015-01-01", end=None, force=False):
    """
    Fetch SPY data using yfinance.
    Cached to data/market/spy.csv
    """
    # If forcing refresh or file doesn't exist, fetch new data
    if force or not SPY_PATH.exists():
        logger.info("Forcing refresh or no cache found, fetching fresh data...")
    elif SPY_PATH.exists():
        try:
            # Try to load cached data
            cached = pd.read_csv(SPY_PATH, parse_dates=["date"], index_col="date")
            
            # Convert empty strings back to NaN
            cached = cached.replace('', np.nan)
            
            # Check if cached data is recent enough
            latest_cached_date = cached.index[-1].date()
            today = datetime.now().date()
            
            # If cached data is from today or yesterday, it's fresh enough
            days_diff = (today - latest_cached_date).days
            if days_diff <= 1:
                logger.info(f"Using cached SPY data up to {latest_cached_date}")
                return cached
            else:
                logger.info(f"Cached data is {days_diff} days old, fetching fresh...")
        except Exception as e:
            logger.warning(f"Failed to load cached data: {e}. Fetching fresh...")

    # Set end date properly
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")
    
    logger.info(f"Fetching SPY data from {start} to {end}")
    
    spy = yf.download(
        "SPY",
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )

    if spy.empty:
        logger.error("No data received from yfinance")
        # Try to load cached data as fallback
        if SPY_PATH.exists():
            logger.info("Falling back to cached data")
            cached = pd.read_csv(SPY_PATH, parse_dates=["date"], index_col="date")
            return cached.replace('', np.nan)
        else:
            raise ValueError("No SPY data available")

    spy = spy[["Close"]].copy()
    spy.columns = ["close"]
    spy.index.name = "date"
    
    # Ensure proper datetime index
    spy.index = pd.to_datetime(spy.index)
    
    logger.info(f"Fetched {len(spy)} days of SPY data")
    logger.info(f"Date range: {spy.index[0].date()} to {spy.index[-1].date()}")

    return spy


def ensure_spy(start="2015-01-01", horizon=1, force=False):
    """
    Ensure SPY prices + returns exist on disk.
    Safe to call from pipeline.
    """
    # Fetch price data
    spy = fetch_spy(start=start, force=force)
    
    # Always add returns to ensure consistency
    logger.info("Calculating returns...")
    spy_with_returns = add_returns(spy, horizon=horizon)
    
    # Use the new function to save with proper NaN handling
    save_market_data_with_nan_handling(spy_with_returns, SPY_PATH)
    
    # Log summary
    fwd_col = f"fwd_ret_{horizon}d"
    valid_fwd = spy_with_returns[fwd_col].notna().sum()
    
    logger.info(f" SPY data saved to {SPY_PATH}")
    logger.info(f"   Date range: {spy_with_returns.index[0].date()} to {spy_with_returns.index[-1].date()}")
    logger.info(f"   Rows with forward returns: {valid_fwd}/{len(spy_with_returns)}")
    
    if valid_fwd > 0:
        last_valid = spy_with_returns[spy_with_returns[fwd_col].notna()].index[-1].date()
        logger.info(f"   Latest date with forward return: {last_valid}")
    
    return spy_with_returns