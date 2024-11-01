import pandas as pd
import matplotlib.pyplot as plt
from data.data_collection import setup_data
from ai_algorithm.stock_rl import ppo_portfolio_algorithm
from finance_algorithm.black_litterman import black_litterman_optimization
from finance_algorithm.fama_french import fama_french_5_algorithm
from finance_algorithm.evaluate_portfolio import backtest_portfolio


def test_black_litterman():
    data = pd.read_parquet('stock_prices.parquet')
    returns = data.pct_change()
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
    coefficients = fama_french_5_algorithm(ticker)
    print(coefficients)


def test_ppo_portfolio_algorithm(steps=10_000, device='mps', ticker="AAPL"):
    ppo_portfolio_algorithm(total_timestamps=steps,
                            device=device, checker_ticker=ticker)


def test_portfolio():
    data = pd.read_parquet('data/test_hackathon_data_with_adjusted_splits.parquet')
    performance, composition = backtest_portfolio(data)
    performance.to_csv('portfolio_performance.csv')
    composition.to_csv('portfolio_composition.csv')


if __name__ == "__main__":
    setup_data()
    test_portfolio()
