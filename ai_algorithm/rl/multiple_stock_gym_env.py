"""
Gymnasium Environment for Multi Stock Trading
"""
from typing import Optional, Any, SupportsFloat, Union

from gymnasium import Env
from gymnasium.core import ActType, ObsType, RenderFrame
from pandas import DataFrame

from common.config import INDICATORS, PRICE_COLUMN


def personal_process_data(stock_data: DataFrame, window_size: int, frame_bound: tuple[int, int]):
    """
    Processing the data such that it can be used by the reinforcement learning gymnasium environment
    to get rewards and observations.

    :param stock_data: DataFrame with stock data.
    :param window_size: How back in the past should we take into consideration?
    :param frame_bound: What's the allowable range of data that we are considering?
    :returns: [prices, signal_features]. Prices is a list of all the stock prices in that time 
    frame, while signal_features is a list of all the values that define all the stock's data.
    """

    start = frame_bound[0] - window_size
    end = frame_bound[1]

    prices = stock_data.loc[:, PRICE_COLUMN].to_numpy()[start:end]
    signal_features = stock_data.loc[:, INDICATORS].to_numpy()[start:end]

    return prices, signal_features


class MultiStockEnv(Env):
    """
    Building off the Gymnasium's Env, we want a stock environment such that we can trade
    multiple stocks at once, and get the collective learning experience.
    """

    def __init__(self, prices, signal_features, **kwargs):
        self.prices = prices
        self.signal_features = signal_features

        super().__init__(**kwargs)

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        TODO
        """

        return super().step(action)

    def reset(self, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None) \
            -> tuple[ObsType, dict[str, Any]]:
        """
        TODO
        """

        return super().reset()

    def render(self) -> Union[RenderFrame, list[RenderFrame], None]:
        """
        TODO
        """

        return super().render()

    def close(self) -> None:
        """
        TODO
        """

        super().close()

    def _process_data(self):
        """
        TODO
        """

        return self.prices, self.signal_features

    def render_all(self):
        """
        TODO
        """

        pass
