import pandas as pd
import numpy as np
import os
from config import indicators, predictors


def setup_data():
    if not os.path.isfile('hackathon_sample_v2.parquet'):
        data = pd.read_csv('hackathon_sample_v2.csv')
        data.to_parquet('hackathon_sample_v2.parquet')

    setup_stock_prices()
    setup_tomorrow()


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
