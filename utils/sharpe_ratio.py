from utils.risk_metrics import standard_deviation, expected_return

# Calculate Sharpe Ratio
def sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate):
    ret = expected_return(weights, log_returns) - risk_free_rate
    std = standard_deviation(weights, cov_matrix)
    return ret / std

# Negative Sharpe Ratio to be used in optimization (since we want to maximize Sharpe)
def negative_sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate):
    return -sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate)
