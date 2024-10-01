import numpy as np
import pandas as pd


def estimate_market_caps(returns):
    """
    Estimate market caps based on return volatility and average returns.
    """
    volatility = np.std(returns, axis=0)
    avg_returns = np.mean(returns, axis=0)

    inv_volatility = 1 / volatility
    combined_metric = inv_volatility * (1 + avg_returns)

    scaled_metric = (combined_metric - np.min(combined_metric)) / \
        (np.max(combined_metric) - np.min(combined_metric))
    estimated_market_caps = 1e6 + scaled_metric * (1e9 - 1e6)

    return estimated_market_caps


def black_litterman_optimization(returns, risk_free_rate=0.02/252, tau=0.05, delta=2.5, views=None, num_stocks=75):
    # Estimate market caps
    market_caps = estimate_market_caps(returns)

    # Calculate inputs
    n = returns.shape[1]
    Sigma = np.cov(returns.T)

    # Calculate market weights
    mkt_weights = market_caps / np.sum(market_caps)

    # Calculate implied excess returns (Pi)
    Pi = delta * Sigma.dot(mkt_weights)

    # Incorporate views if provided
    if views is not None:
        P, Q = views
        omega = np.diag(tau * np.dot(P, np.dot(Sigma, P.T)))
        post_Pi = np.linalg.inv(np.linalg.inv(tau * Sigma) + P.T.dot(np.linalg.inv(omega)).dot(P)).dot(
            np.linalg.inv(tau * Sigma).dot(Pi) + P.T.dot(np.linalg.inv(omega)).dot(Q))
    else:
        post_Pi = Pi

    # Calculate posterior covariance
    if views is not None:
        post_Sigma = np.linalg.inv(np.linalg.inv(
            Sigma) + (1/tau) * P.T.dot(np.linalg.inv(omega)).dot(P))
    else:
        post_Sigma = Sigma

    # Calculate optimal weights
    A = np.ones(n)
    B = np.dot(np.linalg.inv(delta * post_Sigma), post_Pi)
    C = np.dot(A.T, np.dot(np.linalg.inv(delta * post_Sigma), A))
    D = np.dot(A.T, B)
    E = np.dot(post_Pi.T, np.dot(np.linalg.inv(delta * post_Sigma), post_Pi))

    lam = (E * C - D * D) / (C * (E + risk_free_rate) - D * D)
    optimal_weights = (1/delta) * \
        np.dot(np.linalg.inv(post_Sigma), post_Pi - lam * A)

    # Select top stocks
    sorted_indices = np.argsort(np.abs(optimal_weights))[::-1]
    selected_indices = sorted_indices[:num_stocks]
    selected_weights = optimal_weights[selected_indices]

    # Normalize weights to ensure they sum to 1 (or -1 for short positions)
    long_weights = selected_weights[selected_weights > 0]
    short_weights = selected_weights[selected_weights < 0]

    if len(long_weights) > 0:
        long_weights /= np.sum(np.abs(long_weights))
    if len(short_weights) > 0:
        short_weights /= np.sum(np.abs(short_weights))

    selected_weights = np.concatenate([long_weights, short_weights])

    return selected_indices, selected_weights
