import numpy as np
import pandas as pd


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


def black_litterman_optimization(returns, risk_free_rate=0.02/252, tau=0.05, delta=2.5, num_stocks=75, regularization=1e-5):

    if returns.ndim == 1:
        returns = returns.reshape(1, -1)

    market_caps = estimate_market_caps(returns.iloc[0])

    n = returns.shape[1]

    Sigma = np.outer(returns.iloc[0], returns.iloc[0]
                     ) + np.eye(n) * regularization

    mkt_weights = market_caps / np.sum(market_caps)

    Pi = delta * Sigma.dot(mkt_weights)
    post_Pi = Pi
    post_Sigma = Sigma

    A = np.ones(n)
    B = np.dot(np.linalg.inv(delta * post_Sigma), post_Pi)
    C = np.dot(A.T, np.dot(np.linalg.inv(delta * post_Sigma), A))
    D = np.dot(A.T, B)
    E = np.dot(post_Pi.T, np.dot(np.linalg.inv(delta * post_Sigma), post_Pi))
    lam = (E * C - D * D) / (C * (E + risk_free_rate) - D * D)
    optimal_weights = (1/delta) * \
        np.dot(np.linalg.inv(post_Sigma), post_Pi - lam * A)

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
