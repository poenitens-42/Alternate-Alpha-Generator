import json
from src.reddit.client import RedditClient
from src.reddit.processor import process_reddit_data
from src.reddit.aggregate import aggregate_daily_sentiment
from src.reddit.config import (
    DEFAULT_SUBREDDIT,
    DEFAULT_LIMIT,
    DATA_RAW,
    DATA_PROCESSED,
    DATA_AGGREGATED,
)

from src.utils.logger import logger
from src.reddit.alpha import build_reddit_alpha
from src.reddit.plot_alpha import plot_alpha_vs_returns
from src.market.fetch_spy import ensure_spy
import pandas as pd
from datetime import datetime, timedelta


ENABLE_PLOTS = True
MARKET_DATA = "data/market/spy.csv"
ALPHA_PATH = "data/alpha/reddit_alpha.json"


def run_pipeline(use_historical_test=False):
    """
    Run the complete Reddit alpha pipeline.
    
    Args:
        use_historical_test: If True, uses a fixed historical period
                            for testing alignment
    """
    logger.info("Starting Reddit pipeline")


    # Ensure SPY market data exists

    logger.info("Updating market data...")
    market_df = ensure_spy(
        start="2024-01-01",
        horizon=1,
        force=True
    )
    
    # Get latest market date for alignment check
    latest_market_date = market_df.index[-1].date()
    logger.info(f"Latest market data date: {latest_market_date}")
    

    # For testing: Use historical period if requested
  
    if use_historical_test:
        logger.info("Using historical test period for alignment debugging")
        # Use period ending 2 days before latest market date
        # to ensure we have forward returns
        test_end_date = latest_market_date - timedelta(days=2)
        test_start_date = test_end_date - timedelta(days=30)
        logger.info(f"Test period: {test_start_date} to {test_end_date}")
        # Note: You'll need to modify client.fetch_posts to accept time range
        # or filter posts after fetching

  
    # Reddit data pipeline
    
    client = RedditClient()

    posts = client.fetch_posts(DEFAULT_SUBREDDIT, DEFAULT_LIMIT)
    if not posts:
        logger.warning("No posts fetched. Exiting pipeline.")
        return None

    raw_file = client.save_posts(posts, DATA_RAW, DEFAULT_SUBREDDIT)
    logger.info(f"Raw data saved → {raw_file}")

    processed_file = process_reddit_data(raw_file, DATA_PROCESSED)
    logger.info(f"Processed data saved → {processed_file}")

    aggregated_file = aggregate_daily_sentiment(
        input_path=processed_file,
        output_path=DATA_AGGREGATED
    )

   
    # Build alpha signal
  
    alpha_result = build_reddit_alpha(
        input_path=DATA_AGGREGATED,
        output_path=ALPHA_PATH,
        window=7,
        min_posts=5,
        lag=1,  # alpha on date T predicts returns on date T+1
    )
    
    # Log alpha statistics
    if alpha_result:
        alpha_dates = [r["date"] for r in alpha_result]
        logger.info(f"Generated alpha for {len(alpha_dates)} days")
        logger.info(f"Alpha date range: {alpha_dates[0]} to {alpha_dates[-1]}")
        
        # Check if we have market data for alpha dates + 1 day
        alpha_df = pd.DataFrame(alpha_result)
        alpha_df["date"] = pd.to_datetime(alpha_df["date"])
        alpha_df["next_date"] = alpha_df["date"] + pd.Timedelta(days=1)
        
        available_next_dates = [d.date() for d in alpha_df["next_date"] if d.date() <= latest_market_date]
        logger.info(f"Days with forward returns available: {len(available_next_dates)}/{len(alpha_df)}")


    # Evaluation and plotting
    
    if ENABLE_PLOTS and alpha_result:
        logger.info("Plotting Reddit alpha vs market returns")
        try:
            corr, aligned_df = plot_alpha_vs_returns(
                alpha_path=ALPHA_PATH,
                market_path=MARKET_DATA,
                show=True,
            )
            
            if corr is not None:
                logger.info(f"Alpha-market correlation: {corr:.4f}")
                logger.info(f"Aligned days: {len(aligned_df)}")
                
                # Save correlation results
                import os
                os.makedirs("Results/metrics", exist_ok=True)
                results_file = f"Results/metrics/correlation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(results_file, 'w') as f:
                    json.dump({
                        "correlation": corr,
                        "aligned_days": len(aligned_df),
                        "alpha_dates": alpha_dates,
                        "market_latest_date": latest_market_date.isoformat()
                    }, f, indent=2)
                logger.info(f"Results saved to {results_file}")
        except Exception as e:
            logger.error(f"Plotting failed: {e}")

    logger.info("Reddit pipeline completed successfully")
    return aggregated_file


if __name__ == "__main__":
    # Run with historical test mode for debugging
    run_pipeline(use_historical_test=False)