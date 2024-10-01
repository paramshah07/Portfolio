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


def setup():
    data_file = 'data_for_price_prediction.data'

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
    X_train, y_train, X_test, y_test = setup()

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
