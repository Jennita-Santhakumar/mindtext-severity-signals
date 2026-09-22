# Readability vs Sentiment Comparison Results

Data source: **synthetic_reddit_posts.csv (generated; see src/generate_synthetic_data.py)**

Rows: 750

| Feature Set | Accuracy | Macro-F1 | Macro-AUC (OvR) | # Features |
|---|---|---|---|---|
| readability_only | 0.7128 | 0.7122 | 0.8714 | 7 |
| sentiment_only | 0.6649 | 0.6591 | 0.8565 | 4 |
| combined | 0.8245 | 0.8239 | 0.9365 | 11 |

**Best performing feature set (by macro-F1): `combined`**
