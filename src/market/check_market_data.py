# check_market_data.py
import pandas as pd
import numpy as np
from datetime import datetime

def diagnose_market_data():
    """Diagnose issues with market data"""
    
    print("=" * 60)
    print("MARKET DATA DIAGNOSIS")
    print("=" * 60)
    
    # Try to load the CSV
    try:
        df = pd.read_csv("data/market/spy.csv")
        print(f"✓ CSV loaded successfully")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {df.columns.tolist()}")
        print(f"  First few rows:")
        print(df.head(3).to_string())
    except Exception as e:
        print(f"✗ Failed to load CSV: {e}")
        return
    
    # Try with date parsing
    print("\n" + "-" * 60)
    print("Loading with date parsing:")
    try:
        df = pd.read_csv("data/market/spy.csv", parse_dates=["date"], index_col="date")
        print(f"✓ Loaded with date index")
        print(f"  Index name: {df.index.name}")
        print(f"  Index type: {type(df.index)}")
        print(f"  Date range: {df.index[0]} to {df.index[-1]}")
        
        # Check columns
        print(f"\nColumns present: {df.columns.tolist()}")
        
        # Check for required columns
        required = ["close", "ret_1d", "fwd_ret_1d"]
        for col in required:
            if col in df.columns:
                print(f"  ✓ {col}: {df[col].notna().sum()}/{len(df)} non-NaN values")
            else:
                print(f"  ✗ {col}: MISSING")
        
        # Check latest date
        latest = df.index[-1]
        today = datetime.now().date()
        latest_date = latest.date()
        
        print(f"\nLatest data date: {latest_date}")
        print(f"Today: {today}")
        print(f"Days since latest: {(today - latest_date).days}")
        
        if (today - latest_date).days > 2:
            print("  WARNING: Data is more than 2 days old!")
        
        # Check weekend data
        print(f"\nWeekend check:")
        weekend_dates = [d for d in df.index if d.weekday() >= 5]
        if weekend_dates:
            print(f"    Found {len(weekend_dates)} weekend dates (should be none)")
            print(f"  Example: {weekend_dates[0].date()}")
        else:
            print(f"  No weekend dates (good)")
            
    except Exception as e:
        print(f"✗ Failed to parse dates: {e}")
    
    print("\n" + "=" * 60)
    print("RECOMMENDED ACTION:")
    print("1. Run: python -c \"from src.market.fetch_spy import ensure_spy; ensure_spy(force=True)\"")
    print("2. Re-run your pipeline")
    print("=" * 60)

if __name__ == "__main__":
    diagnose_market_data()