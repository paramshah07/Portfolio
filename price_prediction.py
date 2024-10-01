import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os.path

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

from sklearn.preprocessing import MinMaxScaler
from config import indicators, predictors


def create_sequences(data, seq_length=10):
    X = []
    y = []
    for i in range(len(data) - seq_length):
        X_append = data[i:i + seq_length, :len(indicators)]
        y_append = data[i:i + seq_length, len(indicators):]
        X.append(X_append)
        y.append(y_append)
        print(X_append)
        print(y_append)
    return np.array(X), np.array(y)


def setup_data():
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

    with open(data_file, 'rb') as f:
        X_train, y_train, X_test, y_test = pickle.load(f)

    return X_train, y_train, X_test, y_test


def setup_model(X_train):
    model = Sequential()
    model.add(LSTM(units=256, input_shape=(
        X_train.shape[1], X_train.shape[2])))
    model.add(Dense(units=128))
    model.add(Dense(units=32))
    model.add(Dense(units=1))
    model.compile(loss='mean_squared_error',
                  optimizer=Adam(0.001),
                  metrics=['accuracy'])

    if os.path.isfile('models/price_prediction/model_weights.weights.h5'):
        model.load_weights('models/price_prediction/model_weights.weights.h5')

    return model


def price_prediction_algorithm():
    keras_file = 'price_prediction.keras'
    X_train, y_train, X_test, y_test = setup_data()

    if not os.path.isfile(keras_file):
        cp_path = "models/price_prediction/model_weight.weights.h5"
        log_path = "models/price_prediction/logs"

        cp_callback = ModelCheckpoint(
            filepath=cp_path, save_weights_only=True, verbose=1)
        tensorboard_callback = TensorBoard(log_dir=log_path, histogram_freq=1)

        model = setup_model(X_train)

        history = model.fit(
            X_train, y_train,
            epochs=10,
            batch_size=16,
            validation_split=0.1,
            verbose=1,
            shuffle=False,
            callbacks=[cp_callback, tensorboard_callback]
        )

        model.save(keras_file)
    else:
        model = tf.keras.models.load_model(keras_file)


if __name__ == "__main__":
    price_prediction_algorithm()
