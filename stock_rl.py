import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os.path

from stable_baselines3 import PPO

from personal_env import PersonalStockEnv, personal_process_data
from data_collection import setup_data_for_stock_rl


def setup_model(data, stockTickers, device):
    print('[logs] setting up the model')
    prices, signal_features = personal_process_data(
        df=data, window_size=30, stockTickers=stockTickers, frame_bound=(30, len(data)))
    env = PersonalStockEnv(prices, signal_features, df=data,
                           window_size=30, frame_bound=(30, len(data)))
    model = PPO("MlpPolicy", env, device=device,
                tensorboard_log='./logs/saved_models/', verbose=1)

    return model


def check_ppo_portfolio_algorithm(data, stockTickers, ticker="AAPL"):
    print('[logs] checking the trained model')
    prices, signal_features = personal_process_data(
        df=data[data['stock_ticker'] == ticker], window_size=30, stockTickers=stockTickers, frame_bound=(30, len(data[data['stock_ticker'] == ticker])))
    env = PersonalStockEnv(prices, signal_features, df=data[data['stock_ticker'] == ticker], window_size=30, frame_bound=(
        30, len(data[data['stock_ticker'] == ticker])))
    model = PPO.load("trading_bot")

    obs, _ = env.reset()

    while True:
        obs = obs[np.newaxis, ...]
        action, _ = model.predict(obs)
        obs, rewards, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if done:
            print("info", info)
            break

    plt.figure(figsize=(15, 6))
    plt.cla()
    env.render_all()
    plt.show()


def ppo_porfolio_algorithm(total_timesteps=10_000, device='mps', tickerToCheck='AAPL'):
    bot_name = 'trading_bot.zip'

    data, stockTickers = setup_data_for_stock_rl()

    if not os.path.isfile(bot_name):
        model = setup_model(data, stockTickers, device)

        print('[logs] starting the training process')
        model.learn(total_timesteps=total_timesteps)
        model.save(bot_name)

    check_ppo_portfolio_algorithm(data, stockTickers, tickerToCheck)
