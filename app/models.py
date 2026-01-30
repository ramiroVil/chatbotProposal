# app/models.py
import json
import os
import pickle
from typing import Tuple, List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin

ART_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
INTENT_MODEL = os.path.join(ART_DIR, "intent_logreg.pkl")
TFIDF_PATH = os.path.join(ART_DIR, "tfidf_union.pkl")


def load_intent_data() -> Tuple[List[str], List[str]]:
    path = os.path.join(DATA_DIR, "intents.json")
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    X = [r["text"] for r in rows]
    y = [r["intent"] for r in rows]
    return X, y


def _identity(X):
    return X


def _build_vectorizer() -> FeatureUnion:
    """
    Mezcla de features:
      - Word n-grams (1,2) para semántica básica.
      - Char n-grams (3,5) para robustez ante typos/variantes.
    """
    word_tfidf = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
    )
    char_tfidf = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
    )
    # FeatureUnion suma matrices dispersas
    return FeatureUnion([
        ("word", word_tfidf),
        ("char", char_tfidf),
    ])


def train_intent_model(force: bool = False) -> None:
    os.makedirs(ART_DIR, exist_ok=True)
    if os.path.exists(INTENT_MODEL) and os.path.exists(TFIDF_PATH) and not force:
        return

    X, y = load_intent_data()
    vectorizer = _build_vectorizer()
    X_vec = vectorizer.fit_transform(X)

    clf = LogisticRegression(
        max_iter=1000,
        n_jobs=None,
        multi_class="auto",
        random_state=42,
    )
    clf.fit(X_vec, y)

    with open(INTENT_MODEL, "wb") as f:
        pickle.dump(clf, f)
    with open(TFIDF_PATH, "wb") as f:
        pickle.dump(vectorizer, f)


def load_intent_pipeline():
    with open(INTENT_MODEL, "rb") as f:
        clf = pickle.load(f)
    with open(TFIDF_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    return vectorizer, clf


def predict_intent(vectorizer, clf, text: str) -> str:
    X = vectorizer.transform([text])
    return clf.predict(X)[0]