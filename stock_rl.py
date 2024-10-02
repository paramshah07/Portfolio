from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3 import PPO
from config import indicators

from gym_anytrading.envs import StocksEnv

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os.path


def personal_process_data(df, window_size, stockTickers, frame_bound):
    start = frame_bound[0] - window_size
    end = frame_bound[1]

    prices = df.loc[:, 'prc'].to_numpy()[start:end]
    signal_features = df.loc[:, indicators].to_numpy()[start:end]

    return prices, signal_features


class PersonalStockEnv(StocksEnv):
    def __init__(self, prices, signal_features, **kwargs):
        self.prices = prices
        self.signal_features = signal_features
        return super(PersonalStockEnv, self).__init__(**kwargs)

    def _process_data(self):
        return self.prices, self.signal_features


def setup_data():
    print('[logs] starting the algorithm')
    data = pd.read_parquet('hackathon_sample_v2.parquet')
    data = data.fillna(0)
    stockTickers = data['stock_ticker'].unique().tolist()

    return data, stockTickers


def setup_model(data, stockTickers, device):
    print('[logs] setting up the model')
    checkpoint_callback = CheckpointCallback(
        save_freq=1e4, save_path='./model_checkpoints/')
    prices, signal_features = personal_process_data(
        df=data, window_size=30, stockTickers=stockTickers, frame_bound=(30, len(data)))
    env = PersonalStockEnv(prices, signal_features, df=data,
                           window_size=30, frame_bound=(30, len(data)))
    model = PPO("MlpPolicy", env, device=device,
                tensorboard_log='./logs/saved_models/', verbose=1)

    return model


def check_ppo_portfolio_algorithm(data, stockTickers):
    print('[logs] checking the trained model')
    prices, signal_features = personal_process_data(
        df=data[data['stock_ticker'] == 'AAPL'], window_size=30, stockTickers=stockTickers, frame_bound=(30, len(data[data['stock_ticker'] == 'AAPL'])))
    env = PersonalStockEnv(prices, signal_features, df=data[data['stock_ticker'] == 'AAPL'], window_size=30, frame_bound=(
        30, len(data[data['stock_ticker'] == 'AAPL'])))
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


def ppo_porfolio_algorithm(total_timesteps=10_000, device='mpu'):
    bot_name = 'trading_bot.zip'

    data, stockTickers = setup_data()

    if not os.path.isfile(bot_name):
        model = setup_model(data, stockTickers, device)

        print('[logs] starting the training process')
        model.learn(total_timesteps=total_timesteps)
        model.save(bot_name)

    check_ppo_portfolio_algorithm(data, stockTickers)


if __name__ == "__main__":
    # Use device="mps" for apple silicon macs
    ppo_porfolio_algorithm(device="mps")
