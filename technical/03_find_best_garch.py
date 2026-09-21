
import pandas as pd
from arch import arch_model
import warnings
import os
import logging
import numpy as np

def find_best_garch(data_path, column='Close', p_range=range(1, 5), q_range=range(1, 5), log_file='technical/logs/03_garch.log'):
    """
    Finds the best GARCH(p,q) model based on AIC and BIC.

    Args:
        data_path (str): The path to the CSV file containing the time series data.
        column (str): The name of the column to be used.
        p_range (range): A range of values for the p parameter.
        q_range (range): A range of values for the q parameter.
        log_file (str): The path to the log file.

    Returns:
        tuple: A tuple containing the best model's results, best p, and best q.
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
    returns = df[column].pct_change().dropna() * 100 # Multiply by 100 for better convergence

    # Initialize variables to store the best model and its criteria
    best_aic = float('inf')
    best_bic = float('inf')
    best_p = 0
    best_q = 0
    best_model_results = None

    # Suppress convergence warnings
    warnings.filterwarnings("ignore", category=UserWarning, module='arch')

    # Iterate over all p and q values
    for p in p_range:
        for q in q_range:
            try:
                # Define and fit the GARCH model
                model = arch_model(returns, vol='Garch', p=p, q=q)
                model_results = model.fit(disp='off')

                # Check if this model is better
                if model_results.aic < best_aic:
                    best_aic = model_results.aic
                    best_bic = model_results.bic
                    best_p = p
                    best_q = q
                    best_model_results = model_results

            except Exception as e:
                logging.error(f"Error fitting GARCH({p},{q}): {e}")
                continue

    # Log the best model's summary
    if best_model_results:
        logging.info("Best GARCH Model Found:")
        logging.info(f"p = {best_p}, q = {best_q}")
        logging.info(f"AIC: {best_aic}")
        logging.info(f"BIC: {best_bic}")
        logging.info(best_model_results.summary())
        
        # Save conditional volatility
        conditional_volatility = best_model_results.conditional_volatility
        vol_df = pd.DataFrame(conditional_volatility)
        vol_path = 'technical/conditional_volatility.csv'
        vol_df.to_csv(vol_path)
        logging.info(f"\nConditional volatility saved to '{vol_path}'")

        # Save the best parameters
        best_params = {'p': best_p, 'q': best_q}
        params_path = 'technical/best_garch_params.npy'
        np.save(params_path, best_params)
        logging.info(f"Best GARCH parameters saved to '{params_path}'")

    return best_model_results, best_p, best_q

if __name__ == '__main__':
    # Use a file from the psx_data_cleaned directory
    data_path = '../psx_data_cleaned/LUCK_Cleaned.csv'
    find_best_garch(data_path)
