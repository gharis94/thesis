# data_fetcher.py
import yfinance as yf
import pandas as pd
from datetime import datetime
from config import START_DATE, MERGER_MAPPING

def fetch_ticker_data(ticker, logger):
    """
    Fetches data for a ticker. Handles Mergers by trying NEW then OLD.
    """
    
    # 1. Check for Merger/Rename
    if ticker in MERGER_MAPPING:
        new_ticker = MERGER_MAPPING[ticker]
        logger.info(f"[{ticker}] Mapped to new ticker: {new_ticker}. Attempting fetch...")
        
        # Try fetching New Ticker
        df_new = _download_safe(new_ticker)
        
        if not df_new.empty:
            # If successful, we might still want old history if new history is short.
            # But usually, Yahoo merges history for renames. Let's check start date.
            first_date = df_new.index.min()
            
            # If data starts AFTER 2021, we might be missing the 'Old' ticker history.
            if first_date.year > 2021:
                logger.info(f"[{ticker}] New ticker history short. Fetching old ticker {ticker}.KA to append.")
                df_old = _download_safe(f"{ticker}.KA")
                
                if not df_old.empty:
                    # Combine and remove duplicates
                    df_combined = pd.concat([df_old, df_new])
                    df_combined = df_combined[~df_combined.index.duplicated(keep='last')]
                    df_combined.sort_index(inplace=True)
                    return df_combined
            
            return df_new
            
        else:
            logger.warning(f"[{ticker}] New ticker {new_ticker} failed. Falling back to {ticker}.KA")

    # 2. Standard Fetch (Or Fallback)
    yahoo_ticker = f"{ticker}.KA"
    return _download_safe(yahoo_ticker)

def _download_safe(yahoo_symbol):
    """
    Helper to download without raising exceptions on empty data.
    """
    try:
        # actions=True gets Dividends/Splits
        # auto_adjust=False gets RAW prices (Required for our Cleaner)
        df = yf.download(
            yahoo_symbol, 
            start=START_DATE, 
            end=datetime.today(), 
            auto_adjust=False, 
            actions=True, 
            progress=False
        )
        
        # Flatten MultiIndex columns (Fix for new yfinance)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        return df
    except Exception as e:
        return pd.DataFrame() # Return empty on error