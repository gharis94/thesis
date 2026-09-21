import pandas as pd
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import os
import logging

def check_stationarity(data_path, column='Close', signif_level=0.05, log_file='logs/01_stationarity.log'):
    """
    Checks the stationarity of a time series using the Augmented Dickey-Fuller test.

    Args:
        data_path (str): The path to the CSV file containing the time series data.
        column (str): The name of the column to be tested for stationarity.
        signif_level (float): The significance level for the ADF test.
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
    returns = df[column].pct_change().dropna()

    # Perform the ADF test
    result = adfuller(returns)

    # Log the results
    logging.info('Augmented Dickey-Fuller Test:')
    logging.info(f'ADF Statistic: {result[0]}')
    logging.info(f'p-value: {result[1]}')
    logging.info('Critical Values:')
    for key, value in result[4].items():
        logging.info(f'\t{key}: {value}')

    # Interpret the results
    if result[1] <= signif_level:
        logging.info(f"\nResult: The p-value is less than {signif_level}, so we reject the null hypothesis.")
        logging.info("The time series is likely stationary.")
    else:
        logging.info(f"\nResult: The p-value is greater than {signif_level}, so we fail to reject the null hypothesis.")
        logging.info("The time series is likely non-stationary.")

    # Plot the returns and save it
    plt.figure(figsize=(10, 5))
    plt.plot(returns)
    plt.title('Stock Returns')
    plt.xlabel('Date')
    plt.ylabel('Returns')
    plt.grid(True)
    plot_path = os.path.join(log_dir, '01_stationarity_plot.png')
    plt.savefig(plot_path)
    logging.info(f"Plot saved to {plot_path}")
    plt.close()
    print(f"Stationarity check completed. Log saved to {log_file} and plot saved to {plot_path}.")

if __name__ == '__main__':
    # Use a file from the psx_data_cleaned directory
    data_path = '../psx_data_cleaned/LUCK_Cleaned.csv'
    check_stationarity(data_path)
