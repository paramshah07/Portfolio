import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from personal_env import PersonalStockEnv, personal_process_data
import os.path
from stock_rl import ppo_porfolio_algorithm
from config import indicators
from black_litterman import black_litterman_optimization


def select_stock_portfolio(data, num_stocks=75, window_size=1, device='mps'):
    ticker_index = data.columns.get_loc('stock_ticker')
    price_index = data.columns.get_loc('prc')
    indicator_indices = [data.columns.get_loc(col) for col in indicators]

    results_list = []
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

        model_weight = value.item()

        price = stock_data['prc']

        composite_score = (
            0.4 * model_weight +
            0.6 * (1 / price)
        )
        results_list.append(
            {'ticker': ticker, 'composite_score': composite_score})

    results = pd.DataFrame(results_list)
    results = results.pivot_table(
        index=None, columns='ticker', values='composite_score')
    results.reset_index(drop=True, inplace=True)

    selected_indices, selected_weights = black_litterman_optimization(results)
    selected_stocks = [data.stock_ticker.tolist()[i] for i in selected_indices]

    return selected_stocks, selected_weights


def backtest_portfolio(data, num_stocks=75, window_size=1, device='mps'):

    data = data.sort_values('date')

    dates = data['date'].unique()

    portfolio_performance = []
    portfolio_compositions = []
    current_portfolio = None

    for i, current_date in enumerate(dates):
        print(f"Processing date: {current_date}")

        current_data = data[data['date'] == current_date]

        selected_stocks, selected_weights = select_stock_portfolio(
            current_data, num_stocks, window_size, device)

        blackLittermanReturn = pd.DataFrame(
            [selected_weights], columns=selected_stocks)

        sorted_blackLittermanReturn = blackLittermanReturn.sort_values(
            by=0, axis=1, ascending=True)

        if current_portfolio is not None and i > 0:
            portfolio_return = 0
            for stock, weight in zip(sorted_blackLittermanReturn.columns, sorted_blackLittermanReturn.iloc[0]):
                prev_price_data = data[(
                    data['date'] == dates[i-1]) & (data['stock_ticker'] == stock)]['prc']
                curr_price_data = current_data[current_data['stock_ticker']
                                               == stock]['prc']

                if not prev_price_data.empty and not curr_price_data.empty:
                    prev_price = prev_price_data.values[0]
                    curr_price = curr_price_data.values[0]

                    stock_return = (curr_price - prev_price) / prev_price
                    portfolio_return += stock_return * weight

            portfolio_performance.append({
                'date': current_date,
                'return': portfolio_return
            })
        else:
            print(f"Initializing portfolio on date {current_date}")

        portfolio_compositions.append({
            'date': current_date,
            'stocks': sorted_blackLittermanReturn.columns.tolist(),
            'weights': sorted_blackLittermanReturn.iloc[0].tolist()
        })

        current_portfolio = [
            {'ticker': stock, 'weight': weight}
            for stock, weight in zip(sorted_blackLittermanReturn.columns, sorted_blackLittermanReturn.iloc[0])
        ]

    performance_df = pd.DataFrame(portfolio_performance)
    compositions_df = pd.DataFrame(portfolio_compositions)

    if not performance_df.empty:
        performance_df['cumulative_return'] = (
            1 + performance_df['return']).cumprod() - 1

        total_return = performance_df['cumulative_return'].iloc[-1]
        sharpe_ratio = performance_df['return'].mean(
        ) / performance_df['return'].std() * np.sqrt(252)
        max_drawdown = (performance_df['cumulative_return'] /
                        performance_df['cumulative_return'].cummax() - 1).min()

        print(f"Total Return: {total_return:.2%}")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Max Drawdown: {max_drawdown:.2%}")
    else:
        print("No performance data generated. Check if the portfolio is being properly initialized and updated.")

    return performance_df, compositions_df
