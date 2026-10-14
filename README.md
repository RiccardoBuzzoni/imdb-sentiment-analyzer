# IMDB Sentiment Analyzer

A binary sentiment classifier for movie reviews: given the text of a review, it predicts whether the opinion is **positive** or **negative**.

The project uses a classical machine learning approach with [scikit-learn](https://scikit-learn.org/) (TF-IDF features + Logistic Regression) rather than deep learning, with the goal of building a fast, reproducible and interpretable pipeline.

## Results

| Model | CV accuracy (train) | Test accuracy |
|---|---|---|
| Baseline: `CountVectorizer` + Multinomial Naive Bayes | 0.8456 | not evaluated |
| TF-IDF (1-2 grams) + Logistic Regression, default settings | 0.9001 | not evaluated |
| TF-IDF (1-2 grams) + Logistic Regression, tuned (`C=30`) | 0.9105 | **0.9142** |

- Cross-validation: 5 folds on the training set. The test set was used only once, on the final model.
- Precision, recall and F1-score are balanced across the two classes (F1 of 0.91 for negative and 0.92 for positive reviews).
- The most influential features are strongly polarized words (*worst, awful, boring, waste* vs. *great, excellent, perfect, amazing*) and a few bigrams (*the worst, must see, well worth*).

## Dataset

[IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) from Kaggle, downloaded programmatically with `kagglehub` (cached in `~/.cache/kagglehub`, so nothing is stored in the repository).

- 50,000 English reviews, with a `review` (raw text) and a `sentiment` (`positive` / `negative`) column
- Perfectly balanced classes (25,000 each); 418 duplicated reviews are removed during cleaning, leaving 49,582 reviews

## Approach

1. **Exploratory analysis**: size, missing values, duplicates, class balance, review length, HTML noise
2. **Cleaning**: remove duplicates and HTML tags, encode labels as 0/1
3. **Split**: 80/20 stratified train/test split, before any vectorization to avoid data leakage
4. **Baseline**: bag-of-words + Multinomial Naive Bayes
5. **Main model**: TF-IDF with unigrams and bigrams + Logistic Regression, in a scikit-learn `Pipeline`
6. **Tuning**: `GridSearchCV` on `ngram_range` and `C`, with cross-validation on the training set only
7. **Final evaluation**: accuracy, classification report and confusion matrix on the test set
8. **Interpretation**: words and bigrams with the largest positive and negative coefficients
9. **Custom sentences**: predictions on hand-written examples, including negation and sarcasm

## Getting started

The libraries require **Python 3.11 or newer**.

```bash
# Create and activate an environment (conda example)
conda create -n imdb-sentiment python=3.12
conda activate imdb-sentiment

# Install the dependencies
pip install -r requirements.txt
```

Then open `imdb-sentiment-analyzer.ipynb` in Jupyter or VS Code and run all cells. Jupyter is not listed in `requirements.txt`: install it separately (e.g. `pip install jupyterlab`) if your editor does not provide it.

The last cells of the notebook save the trained pipeline to `model/imdb_sentiment_analyzer.joblib`. To reuse it without retraining:

```python
import joblib

model = joblib.load("model/imdb_sentiment_analyzer.joblib")
model.predict_proba(["A wonderful movie, I loved it."])[:, 1]  # P(positive)
```

The saved model must be loaded with the same scikit-learn version used to train it.

## Project structure

```
imdb-sentiment-analyzer/
├── imdb-sentiment-analyzer.ipynb   # analysis, training and evaluation
├── requirements.txt                # Python dependencies
├── model/                          # trained model (not versioned)
└── README.md
```

## Limitations

- **Bag-of-words**: word order and sentence structure are ignored, so negation, contrast and sarcasm are handled only through the individual words and bigrams involved.
- **No neutral class**: the model has only seen clearly positive or negative reviews, so a neutral sentence still gets a confident score.
- **Overconfident probabilities**: with weak regularization (`C=30`), `predict_proba` tends to return extreme values.
- **Domain and language**: trained on English movie reviews; it should not be expected to transfer to other domains or languages.
- **Spurious correlations**: some features carry weight for reasons unrelated to sentiment (e.g. *today* appears in 73.7% of positive reviews that contain it).

## Possible improvements

- Compare other linear models (`LinearSVC`, `SGDClassifier`) and widen the hyperparameter search (`min_df`, `max_features`, trigrams)
- Evaluate stop-word removal and other preprocessing choices
- Calibrate predicted probabilities with `CalibratedClassifierCV`
- Fine-tune a pre-trained transformer (e.g. DistilBERT) for higher accuracy and better handling of context