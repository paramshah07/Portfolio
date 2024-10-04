import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from personal_env import PersonalStockEnv, personal_process_data
import os.path
from stock_rl import ppo_porfolio_algorithm
from config import indicators


def select_stock_portfolio(data, num_stocks=75, window_size=1, device='mps'):
    ticker_index = data.columns.get_loc('stock_ticker')
    price_index = data.columns.get_loc('prc')
    indicator_indices = [data.columns.get_loc(col) for col in indicators]

    results = []
    model_path = 'trading_bot.zip'

    if not os.path.isfile(model_path):
        ppo_porfolio_algorithm()

    model = PPO.load(model_path)

    for _, stock_data in data.iterrows():
        ticker = stock_data['stock_ticker']
        obs_data = stock_data[indicators].values
        obs_data = obs_data.reshape(1, -1).astype(np.float32)
        obs_tensor = torch.from_numpy(obs_data)

        with torch.no_grad():
            action, _ = model.predict(obs_tensor, deterministic=True)
            action_tensor = torch.from_numpy(action)
            obs_tensor = model.policy.obs_to_tensor(obs_tensor)[0]
            value, _, _ = model.policy.evaluate_actions(
                obs_tensor, action_tensor)

        expected_reward = value.item()
        position = "Long" if action > 0 else "Short"

        results.append({
            'ticker': ticker,
            'position': position,
            'expected_reward': expected_reward
        })

    results.sort(key=lambda x: x['expected_reward'], reverse=True)
    selected_stocks = results[:min(num_stocks, len(results))]

    return selected_stocks


def backtest_portfolio(data, num_stocks=75, window_size=1, device='mps'):
    # Sort the data by date
    data = data.sort_values('date')

    # Get unique dates
    dates = data['date'].unique()

    portfolio_performance = []
    current_portfolio = None

    for i, current_date in enumerate(dates):
        print(f"Processing date: {current_date}")

        # Get data for the current date
        current_data = data[data['date'] == current_date]

        # Select stocks for the current date
        selected_stocks = select_stock_portfolio(
            current_data, num_stocks, window_size, device)

        # If we have a previous portfolio, calculate returns
        if current_portfolio is not None and i > 0:
            returns = []
            for stock in current_portfolio:
                ticker = stock['ticker']
                position = stock['position']

                prev_price_data = data[(
                    data['date'] == dates[i-1]) & (data['stock_ticker'] == ticker)]['prc']
                curr_price_data = current_data[current_data['stock_ticker']
                                               == ticker]['prc']

                if not prev_price_data.empty and not curr_price_data.empty:
                    prev_price = prev_price_data.values[0]
                    curr_price = curr_price_data.values[0]

                    if position == "Long":
                        returns.append((curr_price - prev_price) / prev_price)
                    else:
                        returns.append((prev_price - curr_price) / prev_price)
            if returns:
                portfolio_return = np.mean(returns)
                portfolio_performance.append({
                    'date': current_date,
                    'return': portfolio_return
                })
            else:
                print(
                    f"Warning: No valid returns calculated for date {current_date}")

        current_portfolio = selected_stocks

    performance_df = pd.DataFrame(portfolio_performance)

    performance_df['cumulative_return'] = (
        1 + performance_df['return']).cumprod() - 1

    total_return = performance_df['cumulative_return'].iloc[-1]
    sharpe_ratio = performance_df['return'].mean(
    ) / performance_df['return'].std() * np.sqrt(252)  # Assuming 252 trading days in a year
    max_drawdown = (performance_df['cumulative_return'] /
                    performance_df['cumulative_return'].cummax() - 1).min()

    print(f"Total Return: {total_return:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Max Drawdown: {max_drawdown:.2%}")

    return performance_df
