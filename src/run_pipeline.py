"""
End-to-end pipeline: comparing readability vs. sentiment features for
predicting Reddit post mental-health severity.

Usage:
    python src/run_pipeline.py

Steps:
    1. Load data: prefers data/real_dataset.csv (columns [text, severity]) if
       present, otherwise generates/loads the synthetic dataset.
    2. Extract readability + sentiment features per post.
    3. Train three RandomForest classifiers on the same train/test split:
         (a) readability features only
         (b) sentiment features only
         (c) combined (readability + sentiment)
    4. Report accuracy, macro-F1, and per-class metrics for all three, plus
       macro-averaged ROC-AUC (one-vs-rest) where possible.
    5. Write results/comparison_results.json and results/comparison_report.md
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import build_feature_frame, READABILITY_COLUMNS, SENTIMENT_COLUMNS  # noqa: E402
from generate_synthetic_data import generate_dataset  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

REAL_DATA_PATH = DATA_DIR / "real_dataset.csv"
SYNTHETIC_DATA_PATH = DATA_DIR / "synthetic_reddit_posts.csv"


def load_data() -> tuple[pd.DataFrame, str]:
    if REAL_DATA_PATH.exists():
        df = pd.read_csv(REAL_DATA_PATH)
        return df, "real_dataset.csv (user-supplied)"
    if not SYNTHETIC_DATA_PATH.exists():
        df = generate_dataset(n_per_class=250)
        df.to_csv(SYNTHETIC_DATA_PATH, index=False)
    else:
        df = pd.read_csv(SYNTHETIC_DATA_PATH)
    return df, "synthetic_reddit_posts.csv (generated; see src/generate_synthetic_data.py)"


def train_and_eval(X_train, X_test, y_train, y_test, label: str) -> dict:
    clf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42, class_weight="balanced")
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)

    acc = accuracy_score(y_test, preds)
    f1_macro = f1_score(y_test, preds, average="macro")
    report = classification_report(y_test, preds, output_dict=True, zero_division=0)

    try:
        classes = sorted(y_test.unique())
        y_test_bin = label_binarize(y_test, classes=classes)
        auc_macro = roc_auc_score(y_test_bin, probs, average="macro", multi_class="ovr")
    except Exception:
        auc_macro = None

    print(f"\n=== {label} ===")
    print(f"Accuracy: {acc:.4f}  Macro-F1: {f1_macro:.4f}  Macro-AUC(OvR): {auc_macro}")

    return {
        "feature_set": label,
        "accuracy": acc,
        "macro_f1": f1_macro,
        "macro_auc_ovr": auc_macro,
        "per_class_report": report,
        "n_features": X_train.shape[1],
    }


def main():
    df, source = load_data()
    print(f"Data source: {source}")
    print(f"Loaded {len(df)} rows. Severity distribution:\n{df['severity'].value_counts()}")

    feat_df = build_feature_frame(df, text_col="text")

    X_readability = feat_df[READABILITY_COLUMNS]
    X_sentiment = feat_df[SENTIMENT_COLUMNS]
    X_combined = feat_df[READABILITY_COLUMNS + SENTIMENT_COLUMNS]
    y = feat_df["severity"]

    (Xr_train, Xr_test,
     Xs_train, Xs_test,
     Xc_train, Xc_test,
     y_train, y_test) = train_test_split(
        X_readability, X_sentiment, X_combined, y,
        test_size=0.25, random_state=42, stratify=y,
    )

    results = []
    results.append(train_and_eval(Xr_train, Xr_test, y_train, y_test, "readability_only"))
    results.append(train_and_eval(Xs_train, Xs_test, y_train, y_test, "sentiment_only"))
    results.append(train_and_eval(Xc_train, Xc_test, y_train, y_test, "combined"))

    summary = {
        "data_source": source,
        "n_rows": len(df),
        "severity_distribution": df["severity"].value_counts().to_dict(),
        "results": results,
    }

    with open(RESULTS_DIR / "comparison_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Markdown summary report
    lines = ["# Readability vs Sentiment Comparison Results\n",
             f"Data source: **{source}**\n",
             f"Rows: {len(df)}\n",
             "| Feature Set | Accuracy | Macro-F1 | Macro-AUC (OvR) | # Features |",
             "|---|---|---|---|---|"]
    for r in results:
        auc_str = f"{r['macro_auc_ovr']:.4f}" if r["macro_auc_ovr"] is not None else "N/A"
        lines.append(f"| {r['feature_set']} | {r['accuracy']:.4f} | {r['macro_f1']:.4f} | {auc_str} | {r['n_features']} |")

    best = max(results, key=lambda r: r["macro_f1"])
    lines.append(f"\n**Best performing feature set (by macro-F1): `{best['feature_set']}`**\n")

    with open(RESULTS_DIR / "comparison_report.md", "w") as f:
        f.write("\n".join(lines))

    print("\nWrote results/comparison_results.json and results/comparison_report.md")


if __name__ == "__main__":
    main()
