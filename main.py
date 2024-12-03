"""
Run some cross-directory code.
"""
from ai_algorithm.ml.price_prediction import PricePredictionModel
from common.data_collection import setup_data

if __name__ == "__main__":
    setup_data()

    price_prediction_model = PricePredictionModel()
    price_prediction_model.train()
