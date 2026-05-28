# src/market/returns.py

import pandas as pd
import numpy as np
from src.utils.logger import logger


def add_returns(df, horizon=1):
    """
    Adds daily returns and forward returns for prediction.
    
    CRITICAL FIX: Don't dropna() here - it removes the last row which has NaN forward returns.
    """
    df = df.copy()

    # Calculate daily returns
    df["ret_1d"] = df["close"].pct_change()
    
    # Calculate forward returns
    # fwd_ret_1d[t] = return from close[t] to close[t+1]
    df[f"fwd_ret_{horizon}d"] = df["ret_1d"].shift(-horizon)
    
    # Log statistics
    total_rows = len(df)
    valid_ret = df["ret_1d"].notna().sum()
    valid_fwd = df[f"fwd_ret_{horizon}d"].notna().sum()
    
    logger.info(f"Returns calculation:")
    logger.info(f"  - Total rows: {total_rows}")
    logger.info(f"  - Rows with valid daily returns: {valid_ret}")
    logger.info(f"  - Rows with valid forward returns: {valid_fwd}")
    logger.info(f"  - Latest date with forward return: {df[df[f'fwd_ret_{horizon}d'].notna()].index[-1].date() if valid_fwd > 0 else 'None'}")
    
    # IMPORTANT: DO NOT DROPNA HERE!
    # The plotting function will handle NaN dropping appropriately
    return df  # REMOVED: .dropna()


def get_available_forward_returns(df, horizon=1):
    """
    Get a subset of data where forward returns are available.
    Use this in your plotting/analysis functions.
    """
    col = f"fwd_ret_{horizon}d"
    if col not in df.columns:
        raise ValueError(f"Column {col} not found in DataFrame")
    
    # Return only rows with valid forward returns
    return df[df[col].notna()].copy()


def validate_market_data(df):
    """
    Validate that market data has proper structure.
    """
    required = ["close", "ret_1d", "fwd_ret_1d"]
    missing = [col for col in required if col not in df.columns]
    
    if missing:
        raise ValueError(f"Missing columns in market data: {missing}")
    
    # Check for reasonable values
    if df["close"].min() <= 0:
        logger.warning("Some close prices are <= 0")
    
    if df["ret_1d"].abs().max() > 0.5:  # 50% daily move is extreme
        logger.warning("Extreme daily returns detected")
    
    return True


def get_market_summary(df):
    """
    Get summary statistics for market data.
    """
    summary = {
        "date_range": f"{df.index[0].date()} to {df.index[-1].date()}",
        "total_days": len(df),
        "days_with_returns": df["ret_1d"].notna().sum(),
        "days_with_forward_returns": df["fwd_ret_1d"].notna().sum(),
        "mean_return": float(df["ret_1d"].mean()),
        "std_return": float(df["ret_1d"].std()),
        "latest_close": float(df["close"].iloc[-1]),
        "latest_date": df.index[-1].date().isoformat()
    }
    
    return summary
# Add this function to returns.py

def save_market_data_with_nan_handling(df, filepath):
    """
    Save market data to CSV with proper NaN handling.
    Converts NaN to empty strings for clean CSV output.
    """
    df_for_csv = df.copy()
    
    # Convert NaN to empty strings for clean CSV output
    for col in df_for_csv.columns:
        df_for_csv[col] = df_for_csv[col].replace({np.nan: '', pd.NaT: ''})
    
    # Save with index
    df_for_csv.to_csv(filepath, index=True, index_label='date')
    
    logger.info(f"Market data saved to {filepath}")
    logger.info(f"NaN values handled: {df.isna().sum().to_dict()}")
    
    return filepath