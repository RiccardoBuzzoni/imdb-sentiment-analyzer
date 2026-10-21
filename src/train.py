"""Train the IMDB sentiment classifier and save it to disk.

Usage:
    python src/train.py                  # downloads the dataset with kagglehub
    python src/train.py --csv reviews.csv  # uses a local CSV instead
"""

import argparse

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from config import C, MAX_ITER, MIN_DF, MODEL_PATH, NGRAM_RANGE, RANDOM_STATE, TEST_SIZE
from data import load_raw_dataset, prepare_dataset


def build_pipeline() -> Pipeline:
    """TF-IDF (unigrams + bigrams) followed by Logistic Regression."""
    return Pipeline([
        ("vectorizer", TfidfVectorizer(ngram_range=NGRAM_RANGE, min_df=MIN_DF)),
        ("classifier", LogisticRegression(C=C, max_iter=MAX_ITER)),
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the IMDB sentiment classifier.")
    parser.add_argument(
        "--csv",
        help="path to a local CSV with 'review' and 'sentiment' columns "
             "(default: download the dataset from Kaggle)",
    )
    args = parser.parse_args()

    print("Loading dataset...")
    raw = load_raw_dataset(args.csv)
    df = prepare_dataset(raw)
    print(f"{len(raw)} reviews loaded, {len(df)} left after removing duplicates")

    # The split happens before vectorization to avoid data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        df["review"],
        df["label"],
        test_size=TEST_SIZE,
        stratify=df["label"],
        random_state=RANDOM_STATE,
    )

    print(f"Training on {len(X_train)} reviews...")
    model = build_pipeline()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"\nTest accuracy ({len(X_test)} reviews): {accuracy_score(y_test, y_pred):.4f}\n")
    print(classification_report(y_test, y_pred, target_names=["negative", "positive"]))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
