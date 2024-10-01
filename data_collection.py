import pandas as pd
import numpy as np
import os

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