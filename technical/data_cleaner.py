import pandas as pd
import numpy as np

def clean_psx_data(df, ticker_name):


    df = df.copy()

    # --- FIX STARTS HERE ---
    # 1. Force Numeric Conversion (Coerce errors to NaN)
    cols_to_fix = ['Stock Splits', 'Dividends', 'Close', 'Open', 'High', 'Low']
    for col in cols_to_fix:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 2. Standardize Event Columns (Fill NaNs with 0.0)
    if 'Stock Splits' not in df.columns: df['Stock Splits'] = 0.0
    if 'Dividends' not in df.columns: df['Dividends'] = 0.0
    
    df['Stock Splits'] = df['Stock Splits'].fillna(0.0)
    df['Dividends'] = df['Dividends'].fillna(0.0)

    """
    Cleans PSX data with strict validation:
    1. Verifies Yahoo Metadata (ignores splits that didn't happen).
    2. Removes Glitches (crashes that recover).
    3. Detects Real Splits (crashes that stay down), checking for Dividends.
    4. Calculates Adjusted Value for Verification.
    """
    # df = df.copy()

    # # Standardize Event Columns
    # if 'Stock Splits' not in df.columns: df['Stock Splits'] = 0.0
    # if 'Dividends' not in df.columns: df['Dividends'] = 0.0
    # df['Stock Splits'] = df['Stock Splits'].fillna(0.0)
    # df['Dividends'] = df['Dividends'].fillna(0.0)
    
    # Add Event Logging Column
    df['Event_Status'] = ""

    # Calculate Prev_Close
    df['Prev_Close'] = df['Close'].shift(1)

    # ---------------------------------------------------------
    # PHASE 0: VALIDATE YAHOO METADATA
    # ---------------------------------------------------------
    obs_ratio = df['Prev_Close'] / df['Close']
    yahoo_splits = df['Stock Splits'] > 0
    
    for date in df[yahoo_splits].index:
        ratio = obs_ratio.loc[date]
        reported_split = df.loc[date, 'Stock Splits']
        
        # If Price didn't drop (Ratio < 1.2 implies < 16% drop)
        if ratio < 1.2:
            df.at[date, 'Stock Splits'] = 0.0
            df.at[date, 'Event_Status'] = f"IGNORED YAHOO EX-DATE ({reported_split}:1)"
        else:
            df.at[date, 'Event_Status'] = f"CONFIRMED YAHOO EX-DATE ({reported_split}:1)"

    # ---------------------------------------------------------
    # PHASE 1: GLITCH REMOVAL
    # ---------------------------------------------------------
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=10)
    df['Max_Future_Price'] = df['Close'].rolling(window=indexer).max()
    
    is_crash = (df['Close'] < df['Prev_Close'] * 0.40)
    is_recovery = (df['Max_Future_Price'] > df['Close'] * 2.5)
    
    glitch_mask = (is_crash & is_recovery & (df['Stock Splits'] == 0))
    
    if glitch_mask.any():
        print(f"   -> [GLITCH] Removing {glitch_mask.sum()} glitch rows for {ticker_name}.")
        ma_30 = df['Close'].rolling(30).mean()
        bad_data = (df['Close'] < ma_30 * 0.4) & (df['Max_Future_Price'] > ma_30 * 0.8)
        df = df[~bad_data].copy()
        
        # Re-align
        df['Prev_Close'] = df['Close'].shift(1)

    # ---------------------------------------------------------
    # PHASE 2: REAL SPLIT DETECTION (With Dividend Fix)
    # ---------------------------------------------------------
    cond = df['Prev_Close'] > 10
    limit = np.where(cond, df['Prev_Close'] * 0.10, 1.0)
    lower_lock = df['Prev_Close'] - limit
    
    is_drop = (df['Close'] < lower_lock).fillna(False)
    potential_splits = df[is_drop]
    
    df['Split_Ratio'] = 1.0

    for date in potential_splits.index:
        curr = df.loc[date, 'Close']
        prev = df.loc[date, 'Prev_Close']
        div = df.loc[date, 'Dividends']
        
        if pd.isna(prev) or pd.isna(curr) or curr == 0: continue
        
        # --- DIVIDEND FIX: Check if Dividend explains the drop ---
        ex_div_price = prev - div
        adjusted_ratio = ex_div_price / curr
        
        # Only treat as Split if ratio is still high AFTER accounting for dividend
        if adjusted_ratio > 1.10 and df.loc[date, 'Stock Splits'] == 0:
            action = "SPLIT" if adjusted_ratio > 1.8 else "BONUS"
            print(f"   -> [DETECTED] Actual Ex-Date on {date.date()} (Ratio: {adjusted_ratio:.2f})")
            
            df.at[date, 'Split_Ratio'] = adjusted_ratio
            df.at[date, 'Event_Status'] = f"DETECTED ACTUAL EX-DATE ({action} ~{adjusted_ratio:.1f}:1)"

    # ---------------------------------------------------------
    # PHASE 3: CALCULATE RETURNS & ADJUSTED VALUE
    # ---------------------------------------------------------
    final_split = np.where(df['Split_Ratio'] > 1.0, df['Split_Ratio'], df['Stock Splits'].replace(0.0, 1.0))
    
    # --- ADJUSTED VALUE COLUMN ---
    # Shows the theoretical value held: (Price * Split) + Dividend
    df['Adjusted_Value'] = (df['Close'] * final_split) + df['Dividends']
    
    # Log Return
    df['Log_Return'] = np.log(df['Adjusted_Value'] / df['Prev_Close'])
    df['Log_Return'] = df['Log_Return'].replace([np.inf, -np.inf], np.nan)
    
    cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits', 'Event_Status', 'Adjusted_Value', 'Log_Return']
    return df[cols]