"""
Testing our PricePredictionModel class.
"""

import os
import unittest

import numpy as np

from keras import Sequential, Input
from keras.src.layers import Dropout, Dense
from keras.api.optimizers import Adam

from ai_algorithm.ml.price_prediction import PricePredictionModel
from common.config import DATA_DIR


class TestPricePrediction(unittest.TestCase):
    """
    Run a model of price prediction, and see how it plays out
    """

    def __set_up_test_model__(self):
        """
        TODO
        """

        model = Sequential()

        model.add(Input(shape=(self.x_train.shape[1],)))
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

    def setUp(self):
        self.test_dir = os.path.join(DATA_DIR, 'test')
        self.test_keras_file = os.path.join(DATA_DIR, 'test/price_prediction.keras')

        if not os.path.exists(self.test_dir):
            os.mkdir(self.test_dir)

        self.x_train = np.array([[1, 2, 10, 15],
                                 [4, 5, -1, 1000],
                                 [6, 7, 5, -20]])
        self.y_train = np.array([[10],
                                 [11],
                                 [5]])
        self.x_test = np.array([[0, 3, 5, 2],
                                [10, -1, 4, 2],])

        self.price_predictor = PricePredictionModel(self.test_keras_file,
                                                    self.x_train,
                                                    self.y_train,
                                                    self.x_test)

    def tearDown(self):
        if os.path.exists(self.test_keras_file):
            os.remove(self.test_keras_file)

        os.rmdir(self.test_dir)

    def test_train_if_model_saved(self):
        """
        If the model is already saved, we should just import the saved values

        Should load our test's saved model in price predictor's model.
        """

        model = self.__set_up_test_model__()
        model.save(self.test_keras_file)

        self.price_predictor.train()

        number_layers = len(model.layers)

        for i in range(number_layers):
            our_layer = model.layers[i]
            test_layer = self.price_predictor.model.layers[i]
            self.assertEqual(our_layer.get_config(), test_layer.get_config())

    def test_train_if_model_not_saved(self):
        """
        If the model is not saved, we should be running a new training routine.

        Should run one epoch of the training section just for checking.
        """

        model = self.__set_up_test_model__()
        self.price_predictor.train(epochs=1)

        number_layers = len(model.layers)

        for i in range(number_layers):
            our_layer = model.layers[i]
            test_layer = self.price_predictor.model.layers[i]
            self.assertNotEqual(our_layer.get_config(), test_layer.get_config())

    def test_different_indicators_gives_different_predictions(self):
        """
        Different indicator values should give us different price predictions.

        We will train it for one epoch
        """

        self.price_predictor.train(epochs=1)

        first_instance = self.x_test[:1]
        second_instance = self.x_test[1:]

        self.assertFalse(np.array_equal(first_instance, second_instance))

        first_prediction = self.price_predictor.prediction(first_instance)
        second_prediction = self.price_predictor.prediction(second_instance)

        self.assertFalse(np.array_equal(first_prediction, second_prediction))

    def test_same_indicators_gives_same_predictions(self):
        """
        The same indicator values should give the same price predictions.

        We will train it for one epoch
        """

        self.price_predictor.train(epochs=1)

        first_instance = self.x_test[:1]
        second_instance = self.x_test[:1]

        self.assertTrue(np.array_equal(first_instance, second_instance))

        first_prediction = self.price_predictor.prediction(first_instance)
        second_prediction = self.price_predictor.prediction(second_instance)

        self.assertTrue(np.array_equal(first_prediction, second_prediction))
