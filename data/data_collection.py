import datetime
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from config import indicators, predictors, dataDir, dataFile


def setup_data():
    data_file = dataFile
    processed_file = os.path.join(dataDir, 'hackathon_sample_v2.parquet')

    if not os.path.isfile(processed_file):
        data = pd.read_csv(data_file)
        data.to_parquet(processed_file)

    setup_stock_prices()
    setup_tomorrow()
    setup_data_for_prediction()
    setup_data_for_stock_rl()
    setup_testing_data()


def setup_stock_prices():
    data_file = os.path.join(dataDir, 'hackathon_sample_v2.parquet')
    processed_file = os.path.join(dataDir, 'stock_prices.parquet')

    if not os.path.isfile(processed_file):
        table = pd.read_parquet(data_file)
        table = table.loc[:, ['date', 'stock_ticker', 'prc']]
        stocks = table['stock_ticker'].unique().tolist()
        dates = table['date'].unique().tolist()

        new_table = pd.DataFrame(columns=stocks, index=dates)

        for row in table.values:
            date = row[0]
            ticker = row[1]
            price = row[2]
            new_table.loc[date, ticker] = price

        new_table.to_parquet(processed_file)


def setup_tomorrow():
    data_file = os.path.join(dataDir, 'hackathon_sample_v2.parquet')
    processed_file = os.path.join(dataDir, 'stocks_with_tomorrow_prc.parquet')

    if not os.path.isfile(processed_file):
        df = pd.read_parquet(data_file)
        df_list = list()
        stock_tickers = df.stock_ticker.unique().tolist()
        for ticker in stock_tickers:
            new_df = df[df['stock_ticker'] == ticker]
            new_df.loc[:, 'Tomorrow'] = new_df.loc[:, 'prc'].shift(-1).copy()
            df_list.append(new_df)
        df = pd.concat(df_list)
        df = df[indicators + predictors]
        df.to_parquet(processed_file)


def create_sequences(data, seq_length=10):
    x = []
    y = []
    for i in range(len(data) - seq_length):
        x_append = data[i:i + seq_length, :len(indicators)]
        y_append = data[i:i + seq_length, len(indicators):]
        x.append(x_append)
        y.append(y_append)
    return np.array(x), np.array(y)


def setup_data_for_prediction():
    data_file = os.path.join(dataDir, 'stocks_with_tomorrow_prc.parquet')
    processed_file = os.path.join(dataDir, 'data_for_price_prediction.parquet')

    if not os.path.isfile(processed_file):
        data = pd.read_parquet(data_file)
        data = data.loc[:, indicators + predictors]
        data = data.fillna(0)
        data.to_parquet(processed_file)

        # Split into train and test sets
        train_size = int(len(data) * 0.8)
        train_data, test_data = data[:train_size], data[train_size:]

        # Normalize data
        scaler = MinMaxScaler()
        train_data = scaler.fit_transform(train_data)
        test_data = scaler.transform(test_data)

        # Create sequences for training set
        x_train, y_train = create_sequences(train_data)

        # Create sequences for testing set
        x_test, y_test = create_sequences(test_data)

        all_data_points = [x_train, y_train, x_train, y_test]

        with open(data_file, "wb") as f:
            pickle.dump(all_data_points, f)


def setup_data_for_fama_french(ticker):
    data_file = os.path.join(dataDir, 'hackathon_sample_v2.parquet')

    df = pd.read_parquet(data_file)
    stock_data = df[df['stock_ticker'] == ticker]
    stock_data.loc[:, 'Adj Close'] = stock_data.loc[:, 'prc'].copy()
    stock_data.loc[:, 'date'] = stock_data.loc[:, 'date'].apply(
        lambda date: datetime.datetime.strptime(str(date), '%Y%m%d').strftime('%Y-%m'))

    ticker_monthly = stock_data[['date', 'Adj Close']]
    ticker_monthly['date'] = pd.PeriodIndex(ticker_monthly['date'], freq="M")
    ticker_monthly.set_index('date', inplace=True)
    ticker_monthly['Return'] = ticker_monthly['Adj Close'].pct_change() * 100
    ticker_monthly = ticker_monthly.fillna(0)

    return ticker_monthly


def detect_and_adjust_splits_for_all_stocks(data):
    # Sort by stock_ticker and date to ensure proper comparison for each stock
    data = data.sort_values(by=['stock_ticker', 'date']).copy()

    # Iterate through each stock group identified by 'stock_ticker'
    for stock_ticker, stock_data in data.groupby('stock_ticker'):
        stock_data = stock_data.sort_values(by='date')

        # Iterate over the rows for the current stock to check for splits
        for i in range(1, len(stock_data)):
            prev_price = stock_data.iloc[i - 1]['prc']
            current_price = stock_data.iloc[i]['prc']
            prev_market_equity = stock_data.iloc[i - 1]['market_equity']
            current_market_equity = stock_data.iloc[i]['market_equity']

            # Detect a significant drop in price with a stable market equity
            if (prev_price > current_price * 2 and
                    (abs(prev_market_equity - current_market_equity) < 0.1 * prev_market_equity)):
                split_ratio = round(prev_price / current_price)
                print(
                    f"Stock split detected for {stock_ticker} on "
                    f"{stock_data.iloc[i]['date']} with a ratio of {split_ratio}-for-1")

                # Adjust all previous prices for this stock according to the split ratio
                data.loc[(data['stock_ticker'] == stock_ticker) & (
                        data['date'] <= stock_data.iloc[i]['date']), 'prc'] /= split_ratio

    data['stock_ticker'] = data['stock_ticker'].astype(str)
    return data


def setup_data_for_stock_rl():
    print('[logs] starting the algorithm')
    data_file = os.path.join(dataDir, 'hackathon_sample_v2.parquet')
    processed_file = os.path.join(dataDir, 'test_hackathon_data_with_adjusted_splits.parquet')

    if not os.path.isfile(processed_file):
        data = pd.read_parquet(data_file)
        data = detect_and_adjust_splits_for_all_stocks(data)
        data = data.fillna(0)
        data.to_parquet(processed_file)

    data = pd.read_parquet(processed_file)
    stock_tickers = data['stock_ticker'].unique().tolist()

    return data, stock_tickers


def setup_testing_data():
    data_file = os.path.join(dataDir, 'hackathon_sample_v2.parquet')
    processed_file = os.path.join(dataDir, 'test_hackathon_data_with_adjusted_splits.parquet')

    if not os.path.isfile(processed_file):
        data = pd.read_parquet(data_file)
        start_id = data.index[data.date == 20100129].tolist()[0]
        data = data[start_id:]
        data = detect_and_adjust_splits_for_all_stocks(data)
        data = data.fillna(0)
        data.to_parquet(processed_file)
