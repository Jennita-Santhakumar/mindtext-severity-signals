import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
from features import (
    extract_readability_features,
    extract_sentiment_features,
    build_feature_frame,
    READABILITY_COLUMNS,
    SENTIMENT_COLUMNS,
)
from generate_synthetic_data import generate_dataset


def test_readability_features_keys():
    feats = extract_readability_features("This is a simple sentence. It has two parts.")
    assert set(feats.keys()) == set(READABILITY_COLUMNS)
    assert feats["word_count"] > 0
    assert feats["sentence_count"] >= 1


def test_readability_features_empty_string():
    feats = extract_readability_features("")
    assert all(v == 0.0 for v in feats.values())


def test_readability_simple_vs_complex_text():
    simple = "I am sad. I feel bad."
    complex_text = (
        "The multifaceted socioeconomic ramifications precipitated by "
        "systemic institutional negligence engender profound psychological "
        "deterioration among marginalized demographic cohorts."
    )
    simple_feats = extract_readability_features(simple)
    complex_feats = extract_readability_features(complex_text)
    # simple text should read easier (higher flesch reading ease) than the
    # deliberately dense/complex sentence
    assert simple_feats["flesch_reading_ease"] > complex_feats["flesch_reading_ease"]
    assert simple_feats["flesch_kincaid_grade"] < complex_feats["flesch_kincaid_grade"]


def test_sentiment_features_keys():
    feats = extract_sentiment_features("I love this, it's wonderful and great!")
    assert set(feats.keys()) == set(SENTIMENT_COLUMNS)
    assert -1.0 <= feats["vader_compound"] <= 1.0


def test_sentiment_features_empty_string():
    feats = extract_sentiment_features("")
    assert all(v == 0.0 for v in feats.values())


def test_sentiment_positive_vs_negative():
    positive = "I am so happy and grateful, today was wonderful and amazing!"
    negative = "I feel hopeless and empty, everything hurts and nothing matters."
    pos_feats = extract_sentiment_features(positive)
    neg_feats = extract_sentiment_features(negative)
    assert pos_feats["vader_compound"] > neg_feats["vader_compound"]


def test_build_feature_frame_shapes():
    df = pd.DataFrame({"text": ["hello world", "this is bad"], "severity": [0, 1]})
    out = build_feature_frame(df)
    for col in READABILITY_COLUMNS + SENTIMENT_COLUMNS:
        assert col in out.columns
    assert len(out) == 2
    assert "severity" in out.columns


def test_generate_dataset_balanced_and_labeled():
    df = generate_dataset(n_per_class=10)
    assert len(df) == 30
    assert set(df["severity"].unique()) == {0, 1, 2}
    counts = df["severity"].value_counts()
    assert all(c == 10 for c in counts)
    assert df["text"].apply(lambda t: isinstance(t, str) and len(t) > 0).all()
