
import pandas as pd
from arch import arch_model
import numpy as np
import os
import logging

def forecast_returns(data_path, garch_params_path, setar_params_path, column='Close', forecast_steps=1, log_file='technical/logs/05_forecast.log'):
    """
    Forecasts returns using the best GARCH-SETAR model.

    Args:
        data_path (str): The path to the CSV file containing the time series data.
        garch_params_path (str): Path to the saved GARCH model parameters.
        setar_params_path (str): Path to the saved SETAR model parameters.
        column (str): The name of the column to be used.
        forecast_steps (int): The number of steps to forecast ahead.
        log_file (str): The path to the log file.

    Returns:
        None
    """
    # Configure logging
    log_dir = os.path.dirname(log_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logging.basicConfig(filename=log_file, level=logging.INFO, filemode='w',
                        format='%(asctime)s - %(levelname)s - %(message)s')

    # Load the data
    df = pd.read_csv(data_path, index_col='Date', parse_dates=True)

    # Calculate the returns
    returns = df[column].pct_change().dropna() * 100

    # Load the best GARCH model parameters
    try:
        garch_params = np.load(garch_params_path, allow_pickle=True).item()
        garch_p = garch_params['p']
        garch_q = garch_params['q']
    except FileNotFoundError:
        logging.error(f"Error: GARCH parameters file not found at {garch_params_path}")
        logging.error("Please run '03_find_best_garch.py' first.")
        return

    # Fit the best GARCH model to get the latest state
    garch = arch_model(returns, vol='Garch', p=garch_p, q=garch_q)
    garch_results = garch.fit(disp='off')

    # Forecast the conditional volatility
    garch_forecast = garch_results.forecast(horizon=forecast_steps)
    future_volatility = garch_forecast.variance.iloc[-1, 0]
    logging.info(f"Forecasted conditional volatility: {future_volatility}")

    # Load the best SETAR model parameters
    try:
        setar_params = np.load(setar_params_path, allow_pickle=True).item()
    except FileNotFoundError:
        logging.error(f"Error: SETAR parameters file not found at {setar_params_path}")
        logging.error("Please run '04_fit_setar.py' first.")
        return

    threshold = setar_params['threshold']
    ar_order = setar_params['ar_order']
    params = setar_params['params']

    # Get the last 'ar_order' returns to use for forecasting
    historical_returns = returns.iloc[-ar_order:].values
    
    # The SETAR model has two regimes. The parameters are for [regime1_const, regime1_ar..., regime2_const, regime2_ar...]
    params_regime1 = params[:ar_order + 1]
    params_regime2 = params[ar_order + 1:]

    # Determine the regime based on the forecasted volatility
    if future_volatility <= threshold:
        # Regime 1
        logging.info("Forecasting with SETAR Regime 1")
        forecasted_return = params_regime1[0] + np.dot(params_regime1[1:], historical_returns)
    else:
        # Regime 2
        logging.info("Forecasting with SETAR Regime 2")
        forecasted_return = params_regime2[0] + np.dot(params_regime2[1:], historical_returns)
        
    logging.info(f"\nForecasted return for the next step: {forecasted_return / 100}")
    
    return forecasted_return

if __name__ == '__main__':
    data_path = '../psx_data_cleaned/LUCK_Cleaned.csv'
    garch_params_path = 'technical/best_garch_params.npy'
    setar_params_path = 'technical/best_setar_params.npy'
    
    forecast_returns(data_path, garch_params_path, setar_params_path)
