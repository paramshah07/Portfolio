import numpy as np


def estimate_market_caps(returns):
    """
    Estimate market caps based on a single row of returns.
    """

    abs_returns = np.abs(returns)

    inv_abs_returns = np.where(abs_returns != 0, 1 / abs_returns, 1)

    scaled_metric = (inv_abs_returns - np.min(inv_abs_returns)) / \
                    (np.max(inv_abs_returns) - np.min(inv_abs_returns))
    estimated_market_caps = 1e6 + scaled_metric * (1e9 - 1e6)

    return estimated_market_caps


def black_litterman_optimization(returns, risk_free_rate=0.02/252, delta=2.5, num_stocks=75, regularization=1e-5):

    if returns.ndim == 1:
        returns = returns.reshape(1, -1)

    market_caps = estimate_market_caps(returns.iloc[0])

    n = returns.shape[1]

    sigma = np.outer(returns.iloc[0], returns.iloc[0]
                     ) + np.eye(n) * regularization

    mkt_weights = market_caps / np.sum(market_caps)

    pi = delta * sigma.dot(mkt_weights)
    post_pi = pi
    post_sigma = sigma

    a = np.ones(n)
    b = np.dot(np.linalg.inv(delta * post_sigma), post_pi)
    c = np.dot(a.T, np.dot(np.linalg.inv(delta * post_sigma), a))
    d = np.dot(a.T, b)
    e = np.dot(post_pi.T, np.dot(np.linalg.inv(delta * post_sigma), post_pi))
    lam = (e * c - d * d) / (c * (e + risk_free_rate) - d * d)
    optimal_weights = (1/delta) * \
        np.dot(np.linalg.inv(post_sigma), post_pi - lam * a)

    sorted_indices = np.argsort(np.abs(optimal_weights))[::-1]
    selected_indices = sorted_indices[:num_stocks]
    selected_weights = optimal_weights[selected_indices]

    long_weights = selected_weights[selected_weights > 0]
    short_weights = selected_weights[selected_weights < 0]

    if len(long_weights) > 0:
        long_weights /= np.sum(np.abs(long_weights))
    if len(short_weights) > 0:
        short_weights /= np.sum(np.abs(short_weights))

    selected_weights = np.concatenate([long_weights, short_weights])

    return selected_indices, selected_weights
