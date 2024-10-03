import pandas as pd
import numpy as np
import os
import datetime
from sklearn.preprocessing import MinMaxScaler
from config import indicators, predictors
import pickle


def setup_data():
    if not os.path.isfile('hackathon_sample_v2.parquet'):
        data = pd.read_csv('hackathon_sample_v2.csv')
        data.to_parquet('hackathon_sample_v2.parquet')

    setup_stock_prices()
    setup_tomorrow()
    setup_data_for_prediction()
    setup_data_for_stock_rl()


def setup_stock_prices():
    if not os.path.isfile('stock_prices.parquet'):
        table = pd.read_parquet('hackathon_sample_v2.parquet')
        table = table.loc[:, ['date', 'stock_ticker', 'prc']]
        stocks = table['stock_ticker'].unique().tolist()
        dates = table['date'].unique().tolist()

        newTable = pd.DataFrame(columns=stocks, index=dates)

        for row in table.values:
            date = row[0]
            ticker = row[1]
            price = row[2]
            newTable.loc[date, ticker] = price

        newTable.to_parquet('stock_prices.parquet')


def setup_tomorrow():
    if not os.path.isfile('stocks_with_tomorrow_prc.parquet'):
        df = pd.read_parquet("hackathon_sample_v2.parquet")
        dfList = list()
        stockTickers = df.stock_ticker.unique().tolist()
        for ticker in stockTickers:
            newDf = df[df['stock_ticker'] == ticker]
            newDf.loc[:, 'Tomorrow'] = newDf.loc[:, 'prc'].shift(-1).copy()
            dfList.append(newDf)
        df = pd.concat(dfList)
        df = df[indicators + predictors]
        df.to_parquet('stocks_with_tomorrow_prc.parquet')


def create_sequences(data, seq_length=10):
    X = []
    y = []
    for i in range(len(data) - seq_length):
        X_append = data[i:i + seq_length, :len(indicators)]
        y_append = data[i:i + seq_length, len(indicators):]
        X.append(X_append)
        y.append(y_append)
    return np.array(X), np.array(y)


def setup_data_for_prediction():
    data_file = 'data_for_price_prediction.data'

    if not os.path.isfile(data_file):
        data = pd.read_parquet('stocks_with_tomorrow_prc.parquet')
        data = data.loc[:, indicators + predictors]
        data = data.fillna(0)
        data.to_parquet(data_file)

        # Split into train and test sets
        train_size = int(len(data) * 0.8)
        train_data, test_data = data[:train_size], data[train_size:]

        # Normalize data
        scaler = MinMaxScaler()
        train_data = scaler.fit_transform(train_data)
        test_data = scaler.transform(test_data)

        # Create sequences for training set
        X_train, y_train = create_sequences(train_data)

        # Create sequences for testing set
        X_test, y_test = create_sequences(test_data)

        all_data_points = [X_train, y_train, X_train, y_test]

        with open(data_file, "wb") as f:
            pickle.dump(all_data_points, f)


def setup_data_for_fama_french(ticker):
    df = pd.read_parquet('hackathon_sample_v2.parquet')
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

# Function to detect and adjust stock splits
# Function to detect and adjust stock splits for all stocks in the dataset


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
            if prev_price > current_price * 2 and abs(prev_market_equity - current_market_equity) < 0.1 * prev_market_equity:
                split_ratio = round(prev_price / current_price)
                print(
                    f"Stock split detected for {stock_ticker} on {stock_data.iloc[i]['date']} with a ratio of {split_ratio}-for-1")

                # Adjust all previous prices for this stock according to the split ratio
                data.loc[(data['stock_ticker'] == stock_ticker) & (
                    data['date'] <= stock_data.iloc[i]['date']), 'prc'] /= split_ratio

    data['stock_ticker'] = data['stock_ticker'].astype(str)
    return data


def setup_data_for_stock_rl():
    print('[logs] starting the algorithm')

    data_file = 'hackathon_data_with_adjusted_splits.parquet'

    if not os.path.isfile(data_file):
        data = pd.read_parquet('hackathon_sample_v2.parquet')
        data = detect_and_adjust_splits_for_all_stocks(data)
        data = data.fillna(0)
        data.to_parquet(data_file)

    data = pd.read_parquet(data_file)
    stockTickers = data['stock_ticker'].unique().tolist()

    return data, stockTickers
