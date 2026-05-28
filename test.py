# test_yfinance.py
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

print("Testing yfinance data availability...")

# Test 1: Recent data
spy = yf.download("SPY", period="5d", progress=False)
print(f"\nLast 5 days data:")
print(spy[['Close']].tail())

# Test 2: Specific date range
end_date = datetime.now().strftime("%Y-%m-%d")
start_date = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")
spy2 = yf.download("SPY", start=start_date, end=end_date, progress=False)
print(f"\nLast 10 days data ({start_date} to {end_date}):")
print(spy2[['Close']].tail())

# Check if we have Dec 19
dec_19 = pd.Timestamp("2025-12-19")
if dec_19 in spy2.index:
    print(f"\n✅ Dec 19 data found: {spy2.loc[dec_19, 'Close']}")
else:
    print(f"\n❌ Dec 19 NOT in data")
    print(f"Available dates: {spy2.index.date.tolist()}")