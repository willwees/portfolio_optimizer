import sys
import json
import logging
import numpy as np
from portfolio_optimization.mean_variance import optimize_mean_variance
from portfolio_optimization.risk_parity import optimize_risk_parity
from utils.data_fetch import fetch_adj_close_prices, fetch_risk_free_rate
from utils.risk_metrics import expected_return, standard_deviation
from utils.sharpe_ratio import sharpe_ratio

# Setup logging
logging.basicConfig(level=logging.INFO)

def main():
    try:
        # E.g., 'VOO,GLD,BND'
        tickers = sys.argv[1].split(',')
        # E.g., 'mean_variance' or 'risk_parity'
        method = sys.argv[2]
        
        adj_close_df = fetch_adj_close_prices(tickers)
        
        # Calculate log returns and covariance matrix
        log_returns = np.log(adj_close_df / adj_close_df.shift(1)).dropna()
        cov_matrix = log_returns.cov() * 252
        risk_free_rate = fetch_risk_free_rate()

        if method == 'mean_variance':
            optimal_weights = optimize_mean_variance(log_returns, cov_matrix, risk_free_rate, tickers)
        else:
            raise ValueError(f"Unknown optimization method: {method}")
        
        # Calculate performance metrics
        annual_return = expected_return(optimal_weights, log_returns)
        volatility = standard_deviation(optimal_weights, cov_matrix)
        sharpe = sharpe_ratio(optimal_weights, log_returns, cov_matrix, risk_free_rate)

        # Print the result
        result = {'optimal_weights': dict(zip(tickers, optimal_weights)),
                  'annual_return': annual_return,
                  'volatility': volatility,
                  'sharpe': sharpe}
        print(json.dumps(result))

    except Exception as e:
        logging.error(f"Error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
