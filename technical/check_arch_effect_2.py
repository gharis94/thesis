
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
import matplotlib.pyplot as plt
import os
import logging

def check_arch_effect(data_path, column='Close', lags=10, signif_level=0.05, log_file='logs/02_arch_effect.log'):
    """
    Checks for the presence of ARCH effects in a time series.

    Args:
        data_path (str): The path to the CSV file containing the time series data.
        column (str): The name of the column to be tested.
        lags (int): The number of lags to include in the Ljung-Box test.
        signif_level (float): The significance level for the test.
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

    # Calculate the squared returns
    squared_returns = returns**2

    # Perform the Ljung-Box test on the squared returns
    lb_test = acorr_ljungbox(squared_returns, lags=[lags], return_df=True)
    p_value = lb_test.iloc[0]['lb_pvalue']

    # Log the results
    logging.info('Ljung-Box Test for ARCH Effect:')
    logging.info(f'Lags: {lags}')
    logging.info(f'p-value: {p_value}')

    # Interpret the results
    if p_value <= signif_level:
        logging.info(f"\nResult: The p-value is less than {signif_level}, so we reject the null hypothesis.")
        logging.info("There is evidence of ARCH effects in the data.")
    else:
        logging.info(f"\nResult: The p-value is greater than {signif_level}, so we fail to reject the null hypothesis.")
        logging.info("There is no evidence of ARCH effects in the data.")

    # Plot the squared returns and save it
    plt.figure(figsize=(10, 5))
    plt.plot(squared_returns)
    plt.title('Squared Stock Returns')
    plt.xlabel('Date')
    plt.ylabel('Squared Returns')
    plt.grid(True)
    plot_path = os.path.join(log_dir, '02_arch_effect_plot.png')
    plt.savefig(plot_path)
    logging.info(f"Plot saved to {plot_path}")
    plt.close()
    print(f"ARCH effect check completed. Log saved to {log_file} and plot saved to {plot_path}.")

if __name__ == '__main__':
    # Use a file from the psx_data_cleaned directory
    data_path = '../psx_data_cleaned/LUCK_Cleaned.csv'
    check_arch_effect(data_path)
