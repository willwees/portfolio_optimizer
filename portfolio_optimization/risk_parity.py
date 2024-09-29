import numpy as np
from scipy.optimize import minimize
from utils.risk_metrics import risk_contribution

# Objective function to minimize: difference between actual and target risk contributions
def risk_parity_objective(weights, cov_matrix):
    actual_risk_contributions = risk_contribution(weights, cov_matrix)
    # We aim for equal risk contribution from each asset
    target_risk_contributions = np.mean(actual_risk_contributions)
    return np.sum((actual_risk_contributions - target_risk_contributions) ** 2)

# Risk Parity optimization function
# TODO: dynamic bounds
def optimize_risk_parity(cov_matrix, tickers):
    num_assets = len(tickers)

    # Constraints: sum of weights must equal 1
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

    # Bounds: asset weights between 0 and 1 (long-only portfolio)
    bounds = tuple((0, 1) for _ in range(num_assets))

    # Initial weights (equal allocation as starting point)
    initial_weights = [1 / num_assets for _ in range(num_assets)]

    # Optimize using SLSQP to minimize the objective function
    result = minimize(risk_parity_objective, initial_weights, args=(cov_matrix),
                      method='SLSQP', bounds=bounds, constraints=constraints)

    optimal_weights = result.x

    return optimal_weights

