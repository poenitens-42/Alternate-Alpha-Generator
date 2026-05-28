# src/Satellite/pipeline.py
import json
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

from src.Satellite.client import fetch_all_locations, init_gee
from src.Satellite.locations import RETAIL_LOCATIONS
from src.Satellite.processor import build_satellite_alpha, merge_tickers
from src.utils.logger import logger

RESULTS_DIR = Path("Results/satellite")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
ALPHA_DIR = Path("data/alpha")
ALPHA_DIR.mkdir(parents=True, exist_ok=True)


def fetch_stock_returns(ticker, start_date, end_date):
    import yfinance as yf
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)
    if df.empty:
        return pd.DataFrame()
    df = df[["Close"]].copy()
    df.columns = ["close"]
    df.index.name = "date"
    df["ret_1d"] = df["close"].pct_change()
    df["fwd_ret_1d"] = df["ret_1d"].shift(-1)
    df["fwd_ret_5d"] = df["close"].pct_change(5).shift(-5)
    return df


def evaluate_alpha(alpha_df, returns_df, alpha_col="satellite_alpha"):
    if alpha_df.empty or returns_df.empty:
        return {"correlation_1d": None, "correlation_5d": None, "n": 0}

    alpha_df = alpha_df.copy()
    alpha_df["date"] = pd.to_datetime(alpha_df["date"])
    alpha_df = alpha_df.set_index("date")

    merged = alpha_df[[alpha_col]].join(
        returns_df[["fwd_ret_1d", "fwd_ret_5d"]], how="inner"
    ).dropna()

    n = len(merged)
    if n < 5:
        logger.warning(f"Only {n} aligned observations")
        return {"correlation_1d": None, "correlation_5d": None, "n": n}

    corr_1d = merged[alpha_col].corr(merged["fwd_ret_1d"])
    corr_5d = merged[alpha_col].corr(merged["fwd_ret_5d"])
    logger.info(f"N={n} | corr_1d={corr_1d:.4f} | corr_5d={corr_5d:.4f}")

    return {
        "correlation_1d": float(corr_1d),
        "correlation_5d": float(corr_5d),
        "n": n,
        "aligned_df": merged,
    }


def run_satellite_pipeline(tickers=["WMT", "TGT", "HD"], lookback_days=180, window=8, lag=5):
    logger.info("=" * 60)
    logger.info("Satellite Alpha Pipeline Started")
    logger.info("=" * 60)

    init_gee()

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    logger.info(f"Date range: {start_date} to {end_date}")

    alpha_dfs = {}
    results_summary = {}

    for ticker in tickers:
        if ticker not in RETAIL_LOCATIONS:
            logger.warning(f"No locations for {ticker}, skipping")
            continue

        logger.info(f"\n--- {ticker} ---")

        occupancy_df = fetch_all_locations(
            ticker=ticker,
            locations=RETAIL_LOCATIONS[ticker],
            start_date=start_date,
            end_date=end_date,
            save=True,
        )

        if occupancy_df.empty:
            continue

        alpha_df = build_satellite_alpha(df=occupancy_df, ticker=ticker, window=window, lag=lag)
        if alpha_df.empty:
            continue

        alpha_dfs[ticker] = alpha_df
        alpha_path = ALPHA_DIR / f"satellite_alpha_{ticker}.json"
        alpha_df.to_json(alpha_path, orient="records", date_format="iso")
        logger.info(f"Alpha saved: {alpha_path}")

        returns_df = fetch_stock_returns(ticker, start_date, end_date)
        eval_results = evaluate_alpha(alpha_df, returns_df)

        results_summary[ticker] = {
            "observations": len(alpha_df),
            "date_range": f"{alpha_df['date'].min().date()} to {alpha_df['date'].max().date()}",
            "correlation_1d_fwd": eval_results.get("correlation_1d"),
            "correlation_5d_fwd": eval_results.get("correlation_5d"),
            "aligned_n": eval_results.get("n"),
        }

    if len(alpha_dfs) > 1:
        composite_df = merge_tickers(alpha_dfs)
        composite_df.to_json(ALPHA_DIR / "satellite_alpha_composite.json", orient="records", date_format="iso")

    summary_path = RESULTS_DIR / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_path, "w") as f:
        json.dump(results_summary, f, indent=2)

    logger.info("\n=== RESULTS ===")
    for ticker, res in results_summary.items():
        logger.info(f"{ticker}: obs={res['observations']}, corr_1d={res['correlation_1d_fwd']}, corr_5d={res['correlation_5d_fwd']}")

    return results_summary


if __name__ == "__main__":
    run_satellite_pipeline()