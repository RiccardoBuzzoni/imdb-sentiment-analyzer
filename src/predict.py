"""Classify the sentiment of a movie review with the trained model.

Usage:
    python src/predict.py                       # interactive mode
    python src/predict.py "A wonderful movie!"  # single sentence
    python src/predict.py --explain "..."       # also show the words behind the prediction
"""

import argparse
import sys

import joblib
import numpy as np

from config import MODEL_PATH
from data import clean_text

TOP_N = 5


def load_model():
    if not MODEL_PATH.exists():
        sys.exit(f"Model not found at {MODEL_PATH}\nTrain it first with: python src/train.py")
    return joblib.load(MODEL_PATH)


def predict_proba_positive(model, text: str) -> float:
    """Probability that the review is positive."""
    return float(model.predict_proba([clean_text(text)])[0, 1])


def top_contributions(model, text: str, top_n: int = TOP_N):
    """Features that pushed the prediction the most in each direction.

    For a linear model the score is the sum of (TF-IDF value x coefficient)
    over the words found in the text, plus the intercept.
    """
    vectorizer = model.named_steps["vectorizer"]
    classifier = model.named_steps["classifier"]

    x = vectorizer.transform([clean_text(text)]).tocoo()
    names = vectorizer.get_feature_names_out()[x.col]
    contrib = x.data * classifier.coef_[0][x.col]

    order = np.argsort(contrib)
    negative = [(names[i], contrib[i]) for i in order[:top_n] if contrib[i] < 0]
    positive = [(names[i], contrib[i]) for i in order[::-1][:top_n] if contrib[i] > 0]
    return positive, negative


def format_contributions(items) -> str:
    return ", ".join(f"{name} ({value:+.2f})" for name, value in items)


def show(model, text: str, explain: bool) -> None:
    proba = predict_proba_positive(model, text)
    label = "POSITIVE" if proba >= 0.5 else "NEGATIVE"
    print(f"  Sentiment: {label}  (P(positive) = {proba:.2f})")

    if explain:
        positive, negative = top_contributions(model, text)
        if not positive and not negative:
            print("  No known words in the text: the score comes from the intercept only.")
            return
        if positive:
            print(f"  Towards positive: {format_contributions(positive)}")
        if negative:
            print(f"  Towards negative: {format_contributions(negative)}")


def run_interactive(model, explain: bool) -> None:
    print("Sentiment analyzer for movie reviews (in English).")
    print("Type a review and press Enter. Type 'q' to quit.")
    while True:
        try:
            text = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if text.lower() in {"q", "quit", "exit"}:
            break
        if not text:
            continue
        show(model, text, explain)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict the sentiment of a movie review.")
    parser.add_argument(
        "text", nargs="*", help="review to classify, in quotes (omit for interactive mode)"
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="show the words and bigrams that contributed most to the prediction",
    )
    args = parser.parse_args()

    model = load_model()
    if args.text:
        show(model, " ".join(args.text), args.explain)
    else:
        run_interactive(model, args.explain)


if __name__ == "__main__":
    main()
