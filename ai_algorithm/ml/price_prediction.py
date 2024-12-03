"""
AI Price Prediction Model.
"""

import os.path
import pickle

from keras.api.callbacks import ModelCheckpoint, TensorBoard
from keras.api.layers import Dense, Input, Dropout
from keras.api.models import Sequential, load_model
from keras.api.optimizers import Adam

from common.config import DATA_DIR, MODELS_DIR


def define_model(x_train) -> Sequential:
    model = Sequential()

    model.add(Input(shape=(x_train.shape[1],)))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=256))
    model.add(Dropout(0.5))
    model.add(Dense(units=128))
    model.add(Dropout(0.5))
    model.add(Dense(units=32))
    model.add(Dropout(0.5))
    model.add(Dense(units=1))

    model.compile(loss='mean_squared_error',
                  optimizer=Adam(0.001),
                  metrics=['accuracy'])

    return model


class PricePredictionModel:
    """
    Instance of AI price prediction model. 

    If previous iteration data is available, we load that directly.
    """

    def __init__(self,
                 keras_file=os.path.join(DATA_DIR, 'price_prediction.keras'),
                 x_train=None,
                 y_train=None,
                 x_test=None,
                 y_test=None):
        self.weight_file: str = os.path.join(MODELS_DIR,
                                             'price_prediction/model_weights.weights.h5')
        self.data_file: str = os.path.join(DATA_DIR, 'data_for_price_prediction.data')

        self.keras_file = keras_file
        self.log_path = os.path.join(MODELS_DIR, 'price_prediction/logs/')

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        # Getting the data
        if self.x_train is None:
            with open(self.data_file, 'rb') as f:
                self.x_train, self.y_train, self.x_test, self.y_test = pickle.load(f)

        self.model = define_model(self.x_train)

        if os.path.isfile(self.weight_file):
            self.model.load_weights(self.weight_file)

    def train(self, epochs=10, batch_size=16, validation_split=0.1, verbose=1, shuffle=False):
        """
        Train the price prediction model.
        """

        if not os.path.isfile(self.keras_file):
            cp_callback = ModelCheckpoint(
                filepath=self.weight_file, save_weights_only=True, verbose=1)
            tensorboard_callback = TensorBoard(log_dir=self.log_path, histogram_freq=1)

            # The training model
            self.model.fit(
                self.x_train, self.y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                verbose=verbose,
                shuffle=shuffle,
                callbacks=[cp_callback, tensorboard_callback]
            )

            self.model.save(self.keras_file)
        else:
            self.model = load_model(self.keras_file)

    def prediction(self, data):
        """
        Give back a prediction given the model and an observed data
        """

        predicted_price = self.model.predict(data)
        return predicted_price
