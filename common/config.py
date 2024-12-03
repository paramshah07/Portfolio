"""
TODO
"""

import os.path

PRICE_COLUMN = 'prc'

INDICATORS = []  # TODO

PREDICTORS = []  # TODO

COMMON_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_DIR = os.path.dirname(COMMON_DIR)

DATA_DIR = os.path.join(PROJECT_DIR, "data")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
LOGS_DIR = os.path.join(PROJECT_DIR, "logs")

DATA_FILE = os.path.join(DATA_DIR, 'data.csv')
