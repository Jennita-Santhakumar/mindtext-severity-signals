"""
Feature extraction for readability and sentiment analysis.

Readability metrics (via the `textstat` library):
    - Flesch Reading Ease
    - Flesch-Kincaid Grade Level
    - SMOG Index
    - Gunning Fog Index
    - Automated Readability Index
    - Word count / sentence count (structural features)

Sentiment metrics (via VADER, `vaderSentiment`, a lightweight rule-based
lexicon/sentiment-intensity model well suited to short informal social-media
text - no GPU or large model download required):
    - compound, positive, neutral, negative scores
"""
from __future__ import annotations
import textstat
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd

_analyzer = SentimentIntensityAnalyzer()

READABILITY_COLUMNS = [
    "flesch_reading_ease",
    "flesch_kincaid_grade",
    "smog_index",
    "gunning_fog",
    "automated_readability_index",
    "word_count",
    "sentence_count",
]

SENTIMENT_COLUMNS = [
    "vader_compound",
    "vader_pos",
    "vader_neu",
    "vader_neg",
]


def extract_readability_features(text: str) -> dict:
    """Compute readability metrics for a single piece of text.

    Falls back to 0.0 for metrics that raise on degenerate input
    (e.g. empty string, single word) so the pipeline never crashes on
    edge-case posts.
    """
    text = text or ""
    if not text.strip():
        return {col: 0.0 for col in READABILITY_COLUMNS}

    def safe(fn, *args):
        try:
            val = fn(*args)
            return float(val)
        except Exception:
            return 0.0

    return {
        "flesch_reading_ease": safe(textstat.flesch_reading_ease, text),
        "flesch_kincaid_grade": safe(textstat.flesch_kincaid_grade, text),
        "smog_index": safe(textstat.smog_index, text),
        "gunning_fog": safe(textstat.gunning_fog, text),
        "automated_readability_index": safe(textstat.automated_readability_index, text),
        "word_count": safe(textstat.lexicon_count, text, True),
        "sentence_count": safe(textstat.sentence_count, text),
    }


def extract_sentiment_features(text: str) -> dict:
    """Compute VADER sentiment intensity scores for a single piece of text."""
    text = text or ""
    if not text.strip():
        return {col: 0.0 for col in SENTIMENT_COLUMNS}
    scores = _analyzer.polarity_scores(text)
    return {
        "vader_compound": float(scores["compound"]),
        "vader_pos": float(scores["pos"]),
        "vader_neu": float(scores["neu"]),
        "vader_neg": float(scores["neg"]),
    }


def build_feature_frame(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Given a DataFrame with a text column, return a new DataFrame with all
    readability + sentiment feature columns appended, preserving original
    columns (e.g. `severity`)."""
    readability_rows = df[text_col].apply(extract_readability_features).apply(pd.Series)
    sentiment_rows = df[text_col].apply(extract_sentiment_features).apply(pd.Series)
    out = pd.concat([df.reset_index(drop=True), readability_rows, sentiment_rows], axis=1)
    return out
