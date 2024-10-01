import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_collection import setup_data
from black_litterman import black_litterman_optimization


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


if __name__ == "__main__":
    setup_data()