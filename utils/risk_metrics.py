import numpy as np

# Calculate the portfolio variance
def portfolio_variance(weights, cov_matrix):
    return weights.T @ cov_matrix @ weights

# Calculate portfolio standard deviation (volatility)
def standard_deviation(weights, cov_matrix):
    return np.sqrt(portfolio_variance(weights, cov_matrix))

# Calculate expected return
# TODO: Find other ways to calculate expected return
def expected_return(weights, log_returns):
    return np.sum(log_returns.mean() * weights) * 252

# Calculate the marginal risk contribution of each asset
def marginal_risk_contribution(weights, cov_matrix):
    portfolio_std_dev = standard_deviation(weights, cov_matrix)
    return (cov_matrix @ weights) / portfolio_std_dev

# Calculate the risk contribution of each asset
def risk_contribution(weights, cov_matrix):
    mrc = marginal_risk_contribution(weights, cov_matrix)
    total_portfolio_variance = portfolio_variance(weights, cov_matrix)
    return weights * mrc / np.sqrt(total_portfolio_variance)
