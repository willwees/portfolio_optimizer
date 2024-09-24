import numpy as np
from scipy.optimize import minimize
from utils.sharpe_ratio import negative_sharpe_ratio

# Optimize the portfolio weights to maximize the Sharpe Ratio
# TODO: dynamic bounds
def optimize_mean_variance(log_returns, cov_matrix, risk_free_rate, tickers):
    # Define constraints and bounds
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(len(tickers)))

    # Set initial weights
    initial_weights = [1 / len(tickers) for _ in range(len(tickers))]

    # Perform optimization to maximize the Sharpe Ratio
    result = minimize(negative_sharpe_ratio, initial_weights, args=(log_returns, cov_matrix, risk_free_rate),
                      method='SLSQP', bounds=bounds, constraints=constraints)

    optimal_weights = result.x
    
    return optimal_weights, annual_return, volatility, sharpe
