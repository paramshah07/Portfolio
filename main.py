import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_collection import setup_data
from black_litterman import black_litterman_optimization
from fama_french import fama_french_5_algorithm
from stock_rl import ppo_porfolio_algorithm


def test_black_litterman():
    data = pd.read_parquet('stock_prices.parquet')
    returns = data.pct_change()
    returns = returns.fillna(0)

    index_to_keep = returns.std() != 0

    data = data.loc[:, index_to_keep]
    returns = returns.loc[:, index_to_keep]
    stocks = data.keys()

    # Run optimization
    selected_indices, selected_weights = black_litterman_optimization(returns)

    stock_selected = [stocks[i] for i in selected_indices]

    plt.plot(stock_selected, selected_weights)
    plt.show()


def test_fama_french_5(ticker):
    fama_french_5_algorithm(ticker)


def test_ppo_porfolio_algorithm(steps=10_000, device='mps'):
    ppo_porfolio_algorithm(total_timesteps=steps, device=device)


if __name__ == "__main__":
    setup_data()
    test_ppo_porfolio_algorithm()
