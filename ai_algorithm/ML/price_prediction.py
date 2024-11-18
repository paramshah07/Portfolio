import os.path
import pickle

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from config import DATA_DIR, MODELS_DIR

# Global Constants
weight_file = os.path.join(MODELS_DIR, 'price_prediction/model_weights.weights.h5')


def setup():
    data_file = os.path.join(DATA_DIR, 'data_for_price_prediction.data')

    with open(data_file, 'rb') as f:
        x_train, y_train, x_test, y_test = pickle.load(f)

    return x_train, y_train, x_test, y_test


def setup_model(x_train):
    model = Sequential()

    # Defining the model
    model.add(Input(shape=(x_train.shape[1], x_train.shape[2])))
    model.add(LSTM(units=256))
    model.add(Dropout(0.5))
    model.add(Dense(units=128))
    model.add(Dropout(0.5))
    model.add(Dense(units=32))
    model.add(Dropout(0.5))
    model.add(Dense(units=1))

    model.compile(loss='mean_squared_error',
                  optimizer=Adam(0.001),
                  metrics=['accuracy'])

    if os.path.isfile(weight_file):
        model.load_weights(weight_file)

    return model


def price_prediction_algorithm():
    keras_file = os.path.join(DATA_DIR, 'price_prediction.keras')
    x_train, y_train, x_test, y_test = setup()

    if not os.path.isfile(keras_file):
        log_path = os.path.join(MODELS_DIR, 'price_prediction/logs/')

        cp_callback = ModelCheckpoint(
            filepath=weight_file, save_weights_only=True, verbose=1)
        tensorboard_callback = TensorBoard(log_dir=log_path, histogram_freq=1)

        model = setup_model(x_train)

        model.fit(
            x_train, y_train,
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

    predicted_price = model.predict(x_test)
    return predicted_price, y_test