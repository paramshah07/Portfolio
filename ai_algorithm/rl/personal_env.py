"""
TODO
"""

from gym_anytrading.envs import StocksEnv

from common.config import INDICATORS


def personal_process_data(df, window_size, frame_bound):
    """
    TODO
    """

    start = frame_bound[0] - window_size
    end = frame_bound[1]

    prices = df.loc[:, 'prc'].to_numpy()[start:end]
    signal_features = df.loc[:, INDICATORS].to_numpy()[start:end]

    return prices, signal_features


class PersonalStockEnv(StocksEnv):
    """
    TODO
    """

    def __init__(self, prices, signal_features, **kwargs):
        self.prices = prices
        self.signal_features = signal_features
        super().__init__(**kwargs)

    def _process_data(self):
        """
        TODO
        """

        return self.prices, self.signal_features
