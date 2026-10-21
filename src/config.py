"""Shared settings for the training and prediction scripts."""

from pathlib import Path

# Project root = parent of the src/ folder that contains this file
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "model" / "imdb_sentiment_analyzer.joblib"

KAGGLE_DATASET = "lakshmi25npathi/imdb-dataset-of-50k-movie-reviews"

# Train/test split
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Best hyperparameters found with GridSearchCV in the notebook
NGRAM_RANGE = (1, 2)
MIN_DF = 5
C = 30
MAX_ITER = 1000
