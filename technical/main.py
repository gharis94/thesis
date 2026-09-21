# main.py
import os
import pandas as pd
from config import TICKERS, OUTPUT_FOLDER
from logger_setup import setup_logger
from data_fetcher import fetch_ticker_data
from data_cleaner import clean_psx_data

def main():
    # 1. Setup
    logger = setup_logger()
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    logger.info("========== STARTING PSX DATA FETCH ==========")
    logger.info(f"Target Folder: {OUTPUT_FOLDER}")
    
    stats = {"success": 0, "failed": 0, "empty": 0}

    # 2. Loop Tickers
    for ticker in TICKERS:
        try:
            logger.info(f"--- Processing: {ticker} ---")
            
            # A. Fetch
            raw_data = fetch_ticker_data(ticker, logger)
            
            if raw_data.empty:
                logger.warning(f"[{ticker}] No data found (Empty DataFrame).")
                stats["empty"] += 1
                continue
                
            # B. Clean
            # Note: This function prints detection details to console directly
            cleaned_data = clean_psx_data(raw_data, ticker)
            
            if cleaned_data.empty:
                logger.warning(f"[{ticker}] Data became empty after cleaning (All glitches?).")
                stats["empty"] += 1
                continue

            # C. Save
            file_path = os.path.join(OUTPUT_FOLDER, f"{ticker}.csv")
            cleaned_data.to_csv(file_path)
            
            logger.info(f"[{ticker}] Saved {len(cleaned_data)} rows.")
            stats["success"] += 1

        except Exception as e:
            logger.error(f"[{ticker}] CRITICAL ERROR: {e}", exc_info=True)
            stats["failed"] += 1

    # 3. Summary
    logger.info("========== FETCH COMPLETE ==========")
    logger.info(f"Success: {stats['success']} | Empty: {stats['empty']} | Failed: {stats['failed']}")

if __name__ == "__main__":
    main()