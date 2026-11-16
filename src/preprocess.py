"""Text preprocessing: lowercasing, punctuation removal, stop-word filtering.

``TextPreprocessor`` is a scikit-learn compatible transformer so it can live
inside a ``Pipeline`` together with the TF-IDF vectorizer. That is what keeps
the preprocessing free of data leakage: the vectorizer statistics (IDF, vocab)
are learned from the training fold only, because ``Pipeline.fit`` is only ever
called on training data.
"""

from __future__ import annotations

import re

import nltk
from sklearn.base import BaseEstimator, TransformerMixin

# Negations must survive stop-word filtering, otherwise "not good" collapses to
# "good" and the sentiment flips. Everything else in the stop-word list goes.
_NEGATIONS_RAW = frozenset(
    {
        "ain", "ain't", "aren", "aren't", "can't", "cannot", "couldn", "couldn't",
        "didn", "didn't", "doesn", "doesn't", "don", "don't", "hadn", "hadn't",
        "hasn", "hasn't", "haven", "haven't", "isn", "isn't", "mightn",
        "mightn't", "mustn", "mustn't", "needn", "needn't", "never", "no",
        "nor", "not", "shan", "shan't", "shouldn", "shouldn't", "wasn",
        "wasn't", "weren", "weren't", "won", "won't", "wouldn", "wouldn't",
    }
)

_APOSTROPHES = re.compile(r"['’‘]")
_HTML_TAG = re.compile(r"<[^>]+>")
_NON_WORD = re.compile(r"[^\w\s]")


def _strip_apostrophes(text: str) -> str:
    return _APOSTROPHES.sub("", text)


def _load_stopword_list(language: str) -> list[str]:
    from nltk.corpus import stopwords as _stopwords

    try:
        return _stopwords.words(language)
    except LookupError:
        nltk.download("stopwords", quiet=True)
        return _stopwords.words(language)


class TextPreprocessor(BaseEstimator, TransformerMixin):
    """Lowercase -> strip HTML tags -> remove punctuation -> drop stop words."""

    def __init__(self, language: str = "english", keep_negations: bool = True):
        self.language = language
        self.keep_negations = keep_negations

    def fit(self, X, y=None):
        stops = {_strip_apostrophes(w) for w in _load_stopword_list(self.language)}
        if self.keep_negations:
            stops -= {_strip_apostrophes(w) for w in _NEGATIONS_RAW}
        stops.discard("")
        self.stop_words_ = stops
        return self

    def transform(self, X):
        if not hasattr(self, "stop_words_"):
            self.fit(X)
        return [self._clean(text) for text in X]

    def _clean(self, text: str) -> str:
        text = str(text).lower()
        text = _strip_apostrophes(text)   # "don't" -> "dont"
        text = _HTML_TAG.sub(" ", text)   # "<br /><br />" -> " "
        text = _NON_WORD.sub(" ", text)   # punctuation -> space
        tokens = text.split()
        return " ".join(tok for tok in tokens if tok not in self.stop_words_)
