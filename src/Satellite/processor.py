# src/Satellite/processor.py
import numpy as np
import pandas as pd
from src.utils.logger import logger


def remove_outliers(df, col="occupancy_mean", z_thresh=3.0):
    mean = df[col].mean()
    std = df[col].std()
    if std == 0:
        return df
    mask = (df[col] - mean).abs() / std < z_thresh
    removed = (~mask).sum()
    if removed > 0:
        logger.info(f"Removed {removed} outliers (z > {z_thresh})")
    return df[mask].copy()


def adjust_day_of_week(df, col="occupancy_mean"):
    df = df.copy()
    df["dow"] = df["date"].dt.dayofweek
    dow_means = df.groupby("dow")[col].transform("mean")
    overall_mean = df[col].mean()
    df[f"{col}_dow_adj"] = df[col] - dow_means + overall_mean
    logger.info("Day-of-week adjustment applied")
    return df


def rolling_zscore(series, window=8):
    roll_mean = series.rolling(window=window, min_periods=window // 2).mean()
    roll_std = series.rolling(window=window, min_periods=window // 2).std()
    z = (series - roll_mean) / roll_std.replace(0, np.nan)
    return z.fillna(0.0)


def build_satellite_alpha(df, ticker, window=8, lag=5):
    """
    Build lagged satellite occupancy alpha.
    Alpha on date T predicts stock returns on date T+lag.
    Lag=5 days accounts for satellite revisit + processing + execution timing.
    """
    if df.empty:
        logger.warning(f"Empty dataframe for {ticker}")
        return pd.DataFrame()

    df = df.copy().sort_values("date").reset_index(drop=True)
    df = remove_outliers(df)
    df = adjust_day_of_week(df)

    adj_col = "occupancy_mean_dow_adj"
    df["z_score"] = rolling_zscore(df[adj_col], window=window)
    df["satellite_alpha"] = df["z_score"].shift(lag)
    df = df.dropna(subset=["satellite_alpha"])

    result = df[["date", "satellite_alpha", "occupancy_mean", "z_score"]].copy()
    result["ticker"] = ticker

    logger.info(f"{ticker}: {len(result)} alpha observations")
    return result


def merge_tickers(alpha_dfs):
    merged = None
    for ticker, df in alpha_dfs.items():
        if df.empty:
            continue
        df_pivot = df[["date", "satellite_alpha"]].rename(
            columns={"satellite_alpha": f"{ticker}_alpha"}
        )
        merged = df_pivot if merged is None else merged.merge(df_pivot, on="date", how="outer")

    if merged is None:
        return pd.DataFrame()

    merged = merged.sort_values("date").reset_index(drop=True)
    alpha_cols = [c for c in merged.columns if c.endswith("_alpha")]
    merged["composite_alpha"] = merged[alpha_cols].mean(axis=1)
    return merged