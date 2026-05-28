# src/reddit/plot_alpha.py

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from src.utils.logger import logger


def plot_alpha_vs_returns(
    alpha_path="data/alpha/reddit_alpha.json",
    market_path="data/market/spy.csv",
    show=True,
    save_path=None,
):
    """
    Plot alpha vs forward returns with comprehensive diagnostics.
    Handles single data point case properly.
    """
    logger.info(f"Loading alpha from {alpha_path}")
    
    # Load alpha data
    try:
        alpha_data = pd.read_json(alpha_path)
        if alpha_data.empty:
            logger.error("Alpha data is empty")
            return None, pd.DataFrame()
            
        alpha_data["date"] = pd.to_datetime(alpha_data["date"])
        alpha_data = alpha_data.set_index("date").sort_index()
        
        # Set the alpha column name
        alpha_col = "reddit_alpha"
        
        if alpha_col not in alpha_data.columns:
            logger.error(f"Alpha column '{alpha_col}' not found. Available: {alpha_data.columns.tolist()}")
            return None, pd.DataFrame()
            
        logger.info(f"Loaded alpha data: {len(alpha_data)} rows, dates: {alpha_data.index[0].date()} to {alpha_data.index[-1].date()}")
        
    except Exception as e:
        logger.error(f"Failed to load alpha data: {e}")
        return None, pd.DataFrame()

    # Load market data with NaN handling
    logger.info(f"Loading market data from {market_path}")
    try:
        # Try multiple loading strategies
        try:
            market = pd.read_csv(market_path, parse_dates=["date"], index_col="date")
        except:
            try:
                market = pd.read_csv(market_path, parse_dates=[0], index_col=0)
                market.columns = ["close", "ret_1d", "fwd_ret_1d"]
            except:
                market = pd.read_csv(
                    market_path, 
                    index_col=0,
                    parse_dates=True,
                    header=None,
                    names=["close", "ret_1d", "fwd_ret_1d"]
                )
        
        market = market.sort_index()
        
        # CRITICAL FIX: Convert empty strings to NaN
        market = market.replace('', np.nan)
        market = market.replace('nan', np.nan)
        market = market.replace('NaN', np.nan)
        market = market.replace('None', np.nan)
        
        # Ensure numeric columns
        for col in ["close", "ret_1d", "fwd_ret_1d"]:
            if col in market.columns:
                market[col] = pd.to_numeric(market[col], errors='coerce')
        
        # Verify we have the right column
        if "fwd_ret_1d" not in market.columns:
            if len(market.columns) >= 3:
                market.columns = ["close", "ret_1d", "fwd_ret_1d"][:len(market.columns)]
            else:
                logger.error(f"Market data missing columns. Found: {market.columns.tolist()}")
                return None, pd.DataFrame()
                
        logger.info(f"Loaded market data: {len(market)} rows, dates: {market.index[0].date()} to {market.index[-1].date()}")
        logger.info(f"Market data with valid fwd_ret_1d: {market['fwd_ret_1d'].notna().sum()}/{len(market)}")
        
    except Exception as e:
        logger.error(f"Failed to load market data: {e}")
        return None, pd.DataFrame()

    # Print diagnostic information
    print("\n" + "="*60)
    print("DATA DIAGNOSTICS")
    print("="*60)
    print(f"Alpha dates: {alpha_data.index[0].date()} to {alpha_data.index[-1].date()}")
    print(f"Market dates: {market.index[0].date()} to {market.index[-1].date()}")
    print(f"Alpha shape: {alpha_data.shape}")
    print(f"Market shape: {market.shape}")
    print(f"Market fwd_ret_1d notna: {market['fwd_ret_1d'].notna().sum()}")
    print("-"*60)

    # Align alpha with forward returns
    # Alpha on date T should predict returns on date T+1
    df = alpha_data.join(
        market[["fwd_ret_1d"]],
        how="inner",
    )

    # Drop rows where forward return is not observable
    df = df.dropna(subset=[alpha_col, "fwd_ret_1d"])

    if df.empty:
        print("\n NO OVERLAPPING DATA FOUND")
        print("\nDetailed analysis:")
        print(f"Alpha date range: {alpha_data.index[0].date()} to {alpha_data.index[-1].date()}")
        print(f"Market date range: {market.index[0].date()} to {market.index[-1].date()}")
        
        # Check date by date
        print("\nChecking individual dates:")
        overlap_count = 0
        for alpha_date in alpha_data.index:
            # Check if alpha_date + 1 day exists in market WITH valid forward return
            next_date = alpha_date + pd.Timedelta(days=1)
            if next_date in market.index and pd.notna(market.loc[next_date, "fwd_ret_1d"] if next_date in market.index else np.nan):
                overlap_count += 1
                print(f"✓ {alpha_date.date()} → {next_date.date()} (next day has forward return)")
            else:
                # Check why not
                if next_date not in market.index:
                    print(f"✗ {alpha_date.date()} → {next_date.date()} (next day NOT in market index)")
                elif next_date in market.index and pd.isna(market.loc[next_date, "fwd_ret_1d"]):
                    print(f"✗ {alpha_date.date()} → {next_date.date()} (next day in market but NO forward return)")
        
        print(f"\nPotential overlapping pairs: {overlap_count}/{len(alpha_data)}")
        
        # Try direct alignment (same day) for debugging
        print("\nTrying same-day alignment for debugging:")
        df_direct = alpha_data.join(market[["ret_1d"]], how="inner").dropna()
        if not df_direct.empty:
            print(f"Same-day alignment works: {len(df_direct)} overlapping days")
            print("This suggests lag=0 might be correct, or dates are misaligned")
        
        # Debug: Show market forward returns for relevant dates
        print("\nMarket forward returns for alpha dates + 1 day:")
        for alpha_date in alpha_data.index:
            next_date = alpha_date + pd.Timedelta(days=1)
            if next_date in market.index:
                fwd_ret = market.loc[next_date, "fwd_ret_1d"]
                print(f"  {alpha_date.date()} → {next_date.date()}: fwd_ret_1d = {fwd_ret}")
        
        stats = {
            'correlation': None,
            'aligned_days': 0,
            'status': 'NO_DATA',
            'alpha_dates': [d.date().isoformat() for d in alpha_data.index],
            'market_dates_count': len(market),
            'market_fwd_dates_count': market['fwd_ret_1d'].notna().sum()
        }
        
        return stats, df

    # Calculate correlation (handle single data point case)
    if len(df) < 2:
        print(f"\n  WARNING: Only {len(df)} aligned day(s) - correlation cannot be calculated")
        corr = np.nan
        correlation_status = "INSUFFICIENT_DATA"
    else:
        corr = df[alpha_col].corr(df["fwd_ret_1d"])
        correlation_status = "VALID"
    
    print(f"\n DATA ALIGNMENT SUCCESSFUL")
    print(f"Aligned days: {len(df)}")
    if len(df) >= 2:
        print(f"Correlation (alpha vs next-day return): {corr:.4f}")
    print(f"Alpha mean: {df[alpha_col].mean():.4f}")
    print(f"Returns mean: {df['fwd_ret_1d'].mean():.4f}")
    if len(df) >= 2:
        print(f"Alpha std: {df[alpha_col].std():.4f}")
        print(f"Returns std: {df['fwd_ret_1d'].std():.4f}")
    print("="*60)

    # Create plot based on number of data points
    if len(df) == 0:
        print("No data to plot")
        stats = {
            'correlation': None,
            'aligned_days': 0,
            'status': 'NO_DATA'
        }
        return stats, df
    
    elif len(df) == 1:
        print("Only 1 data point - creating simple plot")
        # Simple plot for single point
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot 1: Single point with context
        date_str = df.index[0].date().isoformat()
        alpha_val = df[alpha_col].iloc[0]
        return_val = df['fwd_ret_1d'].iloc[0]
        
        ax1.scatter([alpha_val], [return_val], s=200, color='blue', alpha=0.7, 
                   edgecolors='k', linewidth=2)
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        ax1.set_xlabel("Reddit Alpha", fontsize=12)
        ax1.set_ylabel("Next-day Market Return", fontsize=12)
        ax1.set_title(f"Single Aligned Data Point (Date: {date_str})", fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='both', which='major', labelsize=10)
        
        # Add value labels
        ax1.text(alpha_val, return_val, f"  Alpha={alpha_val:.3f}\n  Return={return_val:.4f}", 
                fontsize=10, verticalalignment='bottom')
        
        # Plot 2: Diagnostics information
        ax2.axis('off')
        info_text = f"""
         ALIGNED DATA POINT
        {'='*30}
        Date: {date_str}
        Alpha Value: {alpha_val:.4f}
        Next-day Return: {return_val:.4f}
        
         DIAGNOSTICS
        {'='*30}
        Alpha Date Range: {alpha_data.index[0].date()} to {alpha_data.index[-1].date()}
        Alpha Days: {len(alpha_data)}
        Market Date Range: {market.index[0].date()} to {market.index[-1].date()}
        Market Days: {len(market)}
        Market Days with Forward Returns: {market['fwd_ret_1d'].notna().sum()}
        
          RECOMMENDATION
        {'='*30}
        Need more data for meaningful correlation.
        Try fetching older Reddit posts or wait for more days.
        """
        ax2.text(0.05, 0.5, info_text, fontfamily='monospace', fontsize=10, 
                verticalalignment='center', linespacing=1.5)
        
        plt.tight_layout()
        
    else:
        # Original plotting code for multiple data points
        print(f"Creating comprehensive plot for {len(df)} data points")
        fig = plt.figure(figsize=(15, 10))
        
        # Plot 1: Time series
        ax1 = plt.subplot(3, 1, 1)
        ax1.plot(df.index, df[alpha_col], label="Reddit Alpha", color='blue', linewidth=2, alpha=0.8)
        ax1.set_ylabel("Alpha Value", color='blue', fontsize=12)
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')
        
        ax1b = ax1.twinx()
        ax1b.plot(df.index, df["fwd_ret_1d"], label="Next-day Return", color='red', linewidth=1, alpha=0.6)
        ax1b.set_ylabel("Return", color='red', fontsize=12)
        ax1b.tick_params(axis='y', labelcolor='red')
        ax1b.legend(loc='upper right')
        
        ax1.set_title(f"Reddit Alpha vs Next-day Market Returns (Correlation: {corr:.4f}, N={len(df)})", 
                     fontsize=14, fontweight='bold')
        
        # Plot 2: Scatter plot
        ax2 = plt.subplot(3, 1, 2)
        if len(df) > 1:
            colors = np.arange(len(df))  # Simple array for colormap
            scatter = ax2.scatter(df[alpha_col], df["fwd_ret_1d"], 
                                 c=colors, cmap='viridis', 
                                 alpha=0.6, edgecolors='k', linewidth=0.5)
        else:
            # Fallback for edge case
            scatter = ax2.scatter(df[alpha_col], df["fwd_ret_1d"], 
                                 color='blue', alpha=0.6, edgecolors='k', linewidth=0.5)
        
        ax2.set_xlabel("Reddit Alpha", fontsize=12)
        ax2.set_ylabel("Next-day Market Return", fontsize=12)
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        ax2.grid(True, alpha=0.3)
        
        # Add colorbar for time progression (only if we have colors array)
        if 'colors' in locals() and len(df) > 1:
            plt.colorbar(scatter, ax=ax2, label='Time progression')
        ax2.set_title("Scatter Plot", fontsize=13)
        
        # Plot 3: Rolling correlation
        ax3 = plt.subplot(3, 1, 3)
        window = min(20, len(df) // 2)
        if window > 1 and len(df) >= window:
            rolling_corr = df[alpha_col].rolling(window=window).corr(df["fwd_ret_1d"])
            ax3.plot(df.index, rolling_corr, color='green', linewidth=2)
            ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax3.fill_between(df.index, 0, rolling_corr, where=rolling_corr>=0, 
                             color='green', alpha=0.3)
            ax3.fill_between(df.index, 0, rolling_corr, where=rolling_corr<0, 
                             color='red', alpha=0.3)
            ax3.set_xlabel("Date", fontsize=12)
            ax3.set_ylabel(f"Rolling Correlation (window={window})", fontsize=12)
            ax3.grid(True, alpha=0.3)
            ax3.set_title(f"Rolling Correlation Over Time", fontsize=13)
        else:
            ax3.text(0.5, 0.5, f"Need at least {window} data points\nfor rolling correlation", 
                    ha='center', va='center', transform=ax3.transAxes, fontsize=12)
            ax3.set_title("Rolling Correlation (Insufficient Data)", fontsize=13)
            ax3.axis('off')
    
    plt.tight_layout()

    if show:
        plt.show()
    
    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
        plt.close()
    
    # Return statistics
    stats = {
        'correlation': float(corr) if not pd.isna(corr) else None,
        'aligned_days': len(df),
        'status': correlation_status if 'correlation_status' in locals() else ('INSUFFICIENT_DATA' if len(df) < 2 else 'VALID'),
        'alpha_mean': float(df[alpha_col].mean()) if len(df) > 0 else None,
        'returns_mean': float(df['fwd_ret_1d'].mean()) if len(df) > 0 else None,
        'start_date': df.index[0].date().isoformat() if len(df) > 0 else None,
        'end_date': df.index[-1].date().isoformat() if len(df) > 0 else None,
        'alpha_values': [float(x) for x in df[alpha_col].tolist()] if len(df) > 0 else [],
        'return_values': [float(x) for x in df['fwd_ret_1d'].tolist()] if len(df) > 0 else []
    }

    if len(df) >= 2:
        stats['alpha_std'] = float(df[alpha_col].std())
        stats['returns_std'] = float(df['fwd_ret_1d'].std())
    
    logger.info(f"Plotting complete: {stats['aligned_days']} aligned days, correlation: {stats['correlation']}")
    
    return stats, df


def debug_market_data(market_path="data/market/spy.csv"):
    """
    Debug function to examine market data issues.
    """
    print("\n" + "="*60)
    print("MARKET DATA DEBUG")
    print("="*60)
    
    try:
        market = pd.read_csv(market_path)
        print(f"Raw CSV shape: {market.shape}")
        print(f"Columns: {market.columns.tolist()}")
        print(f"\nFirst 3 rows:")
        print(market.head(3).to_string())
        print(f"\nLast 3 rows:")
        print(market.tail(3).to_string())
        
        # Check for empty values
        print(f"\nEmpty values in last row:")
        last_row = market.iloc[-1]
        for col in last_row.index:
            value = last_row[col]
            print(f"  {col}: '{value}' (type: {type(value).__name__})")
        
        # Check data types
        print(f"\nData types:")
        for col in market.columns:
            unique_types = market[col].apply(type).unique()
            print(f"  {col}: {[t.__name__ for t in unique_types[:3]]}")
        
    except Exception as e:
        print(f"Error: {e}")


def test_single_point_plot():
    """
    Test function to verify single point plotting works.
    """
    print("\n" + "="*60)
    print("TESTING SINGLE POINT PLOT")
    print("="*60)
    
    # Create test data
    test_alpha = pd.DataFrame({
        'date': ['2025-12-18'],
        'reddit_alpha': [1.3065]
    })
    test_alpha['date'] = pd.to_datetime(test_alpha['date'])
    test_alpha = test_alpha.set_index('date')
    
    test_market = pd.DataFrame({
        'date': ['2025-12-19'],
        'fwd_ret_1d': [0.0091]
    })
    test_market['date'] = pd.to_datetime(test_market['date'])
    test_market = test_market.set_index('date')
    
    # Test join
    df = test_alpha.join(test_market, how='inner')
    print(f"Test join result: {len(df)} rows")
    print(df)
    
    return len(df) == 1


if __name__ == "__main__":
    # For testing
    print("Testing plot_alpha.py functionality...")
    debug_market_data()
    
    if test_single_point_plot():
        print("\n Single point test passed")
    else:
        print("\n Single point test failed")
    
    # Run actual plot
    stats, df = plot_alpha_vs_returns(show=True)
    if stats:
        print(f"\nResults: {stats}")