# config.py
from datetime import datetime
import os
import pandas as pd

# ---------------------------------------------------------
# GLOBAL SETTINGS
# ---------------------------------------------------------
START_DATE = "2020-01-01"
OUTPUT_FOLDER = "psx_data_cleaned"

# ---------------------------------------------------------
# TICKER LOADER FUNCTION
# ---------------------------------------------------------
def load_psx_tickers():
    """
    Load tickers from 'symbol.xlsx' if it exists, otherwise use default list.
    Expected Excel column name: 'Symbol'
    """
    file_path = "symbol.xlsx"
    
    if os.path.exists(file_path):
        try:
            # Load Excel
            df = pd.read_excel(file_path)
            
            # Ensure 'Symbol' column exists
            if 'Symbol' in df.columns:
                tickers = df['Symbol'].dropna().astype(str).tolist()
                # Clean up tickers (remove whitespace)
                tickers = [t.strip() for t in tickers]
                print(f"[Config] Loaded {len(tickers)} tickers from {file_path}")
                return tickers
            else:
                print(f"[Config] Warning: 'Symbol' column not found in {file_path}. Using defaults.")
        except Exception as e:
            print(f"[Config] Error reading {file_path}: {e}. Using defaults.")

    # FALLBACK SAMPLE LIST
    print("[Config] Using default sample tickers.")
    return [
        "HBL", "UBL", "OGDC", "PPL", "LUCK", "ENGRO", 
        "SYS", "DAWH", "SILK", "SMBL", "KAPCO", "HUBC"
    ]

# ---------------------------------------------------------
# EXECUTE LOADER
# ---------------------------------------------------------
TICKERS = load_psx_tickers()

# ---------------------------------------------------------
# MERGER MAPPING (Renames/Restructures ONLY)
# ---------------------------------------------------------
MERGER_MAPPING = {
    "DAWH": "ENGROH.KA",   # Dawood Hercules -> Engro Holdings
    "SMBL": "BML.KA",      # Summit Bank -> Bank Makramah
    "GSKCH": "HALEON.KA"   # GSK Consumer -> Haleon
}