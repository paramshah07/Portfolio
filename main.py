"""
TODO
"""

import os.path

import pandas as pd
import matplotlib.pyplot as plt

from ai_algorithm.ml.price_prediction import price_prediction_algorithm
from ai_algorithm.rl.stock_rl import ppo_portfolio_algorithm
from common.config import DATA_DIR
from common.data_collection import setup_data
from finance_algorithm.black_litterman import black_litterman_optimization
from finance_algorithm.fama_french import fama_french_5_algorithm
from finance_algorithm.evaluate_portfolio import backtest_portfolio


def test_black_litterman():
    """
    TODO
    """

    data_file = os.path.join(DATA_DIR, "stock_prices.parquet")
    data = pd.read_parquet(data_file)
    returns = data.copy().pct_change()
    returns = returns.fillna(0)

    index_to_keep = returns.std() != 0

    data = data.loc[:, index_to_keep]
    returns = returns.loc[:, index_to_keep]
    stocks = data.keys()

    selected_indices, selected_weights = black_litterman_optimization(returns)

    stock_selected = [stocks[i] for i in selected_indices]

    plt.plot(stock_selected, selected_weights)
    plt.show()


def test_fama_french_5(ticker):
    """
    TODO
    """

    coefficients = fama_french_5_algorithm(ticker)
    print(coefficients)


def test_ppo_portfolio_algorithm(steps=10_000, device='mps', ticker="AAPL"):
    """
    TODO
    """

    ppo_portfolio_algorithm(total_timestamps=steps,
                            device=device, checker_ticker=ticker)


def test_portfolio():
    """
    TODO
    """

    data_from_file = os.path.join(DATA_DIR, 'test_hackathon_data_with_adjusted_splits.parquet')
    performance_file = os.path.join(DATA_DIR, 'portfolio_performance.csv')
    composition_file = os.path.join(DATA_DIR, 'portfolio_composition.csv')

    data = pd.read_parquet(data_from_file)
    performance, composition = backtest_portfolio(data)
    performance.to_csv(performance_file)
    composition.to_csv(composition_file)


def test_price_prediction():
    """
    TODO
    """

    observed, expected = price_prediction_algorithm()

    plt.plot(observed, expected)


if __name__ == "__main__":
    setup_data()
