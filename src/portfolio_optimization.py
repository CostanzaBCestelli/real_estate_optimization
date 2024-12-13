import numpy as np
from scipy.optimize import minimize

def michaud_resampling(data, num_samples=500, max_weight=0.2):
    num_assets = len(data)
    resampled_weights = []

    for _ in range(num_samples):
        # Simulate returns assuming normal distribution
        simulated_returns = np.random.normal(data['Return'], data['Risk'])

        def portfolio_variance(weights):
            return np.dot(weights.T, np.dot(np.diag(data['Risk']**2), weights))

        constraints = (
            {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
        )
        bounds = [(0, max_weight) for _ in range(num_assets)]

        initial_weights = np.array([1.0 / num_assets] * num_assets)
        result = minimize(portfolio_variance, initial_weights, bounds=bounds, constraints=constraints)

        if result.success:
            resampled_weights.append(result.x)

    avg_weights = np.mean(resampled_weights, axis=0)
    data['Weight'] = avg_weights
    return data
