## Portfolio Optimization using AI and Financial Algorithms

### Project Structure

- `ai_algorithm`
  - `RL`
    - `personal_env.py`: Custom `gymnasium` environment for PPO to use and get rewards.
    - `stock_rl.py`: Using PPO RL algorithm to find the best future prediction.
  - `ML`
    - `price_prediction.py`: LSTM model to predict future behavior of a stock.


- `data`: Where data collection happens, and where all the necessary data is stored.
  - `data_collection.py`: sets up all the necessary data files and models


- `financial_algorithm`
  - `black_litterman.py`: Takes market caps, price, investor beliefs and calculates optimal weights for the portfolio.
  - `evaluate_portfolio.py`: Backtesting.
  - `fama_french.py`: Implements Fama French with 5 parameters.


- `config.py`: Global constants.


- `main.py`: Run this file for desired results.