
# 📊 Comparing Readability and Sentiment Analysis to Predict Mental Health Severity on Reddit Data

> **Maintained by [Jennita S](https://github.com/Jennita-Santhakumar)** · [LinkedIn](https://linkedin.com/in/jennitas) · jennitasanthakumar0@gmail.com
**Author:** A. Kabilesh Rajaselvan
**Reg. No:** 21MIA1132
**Institution:** VIT Chennai – SCOPE School

---

## 🧠 Objective

This project has two phases:

1. **Phase 1 - EDA (R):** exploratory data analysis (`eda.Rmd`) — data cleaning,
   descriptive statistics, missing-value/outlier checks, and correlation
   visualizations.
2. **Phase 2 - Modeling (Python, `src/`):** the actual comparison the title
   promises — extracting **readability** metrics and **sentiment** scores
   from Reddit-style posts, then training and comparing classifiers to see
   which signal (or their combination) better predicts a post's
   **mental-health severity band**.

Phase 2 was added because the original repo only contained the EDA notebook
and did not yet implement the readability/sentiment comparison or any
severity prediction model — this update makes the codebase match the title.

---

## ⚠️ Honest note on data source

This environment could not reliably reach Kaggle or the Hugging Face Hub to
download a real labeled dataset (e.g. **dreaddit**, **SWMH**, or a Reddit
"mental health" Kaggle corpus) — several of those require account
credentials or API tokens not available here. Rather than fake a data
source, the pipeline defaults to a **documented synthetic dataset**
(`src/generate_synthetic_data.py`) of 750 Reddit-style posts across three
severity bands (low / moderate / high), with realistic informal register and
deliberately injected noise (register-swaps, filler, rambling) so the task
isn't trivially separable.

**If you have access to a real labeled dataset**, drop a CSV with columns
`text,severity` at `data/real_dataset.csv` — `run_pipeline.py` automatically
prefers it over the synthetic data, and everything downstream (feature
extraction, model training, evaluation) is unchanged. The modeling
methodology below is real and applies identically to either data source;
only the input rows would change.

**Interview-honesty checklist for this project:**
- The feature extraction (readability + sentiment) and modeling/comparison
  code are fully real and functional — verified by running the pipeline
  end-to-end (see Results below).
- The dataset used to produce the numbers below is synthetic, built to bake
  in a realistic but noisy relationship between text style and severity. The
  measured metrics reflect how well the *method* recovers a known-noisy
  signal, not real-world predictive validity on genuine Reddit posts.
- Swapping in a real dataset (dreaddit, SWMH, etc.) is a one-file drop-in
  and would be the natural next step before drawing real-world conclusions.

---

## 🛠️ Tools & Libraries Used

**Phase 1 (EDA, R):** R, RStudio, tidyverse, ggplot2, dplyr, corrplot, skimr, readr, ggcorrplot, knitr

**Phase 2 (Modeling, Python)** — chosen deliberately over R for this phase
because Python's NLP/readability/sentiment tooling (`textstat`,
`vaderSentiment`, `scikit-learn`) is more mature, lighter-weight, and easier
to test than R equivalents:
- **textstat** — Flesch Reading Ease, Flesch-Kincaid Grade Level, SMOG Index,
  Gunning Fog Index, Automated Readability Index, word/sentence counts
- **vaderSentiment** — VADER sentiment intensity (compound/pos/neu/neg),
  a lexicon/rule-based analyzer well suited to short informal social-media
  text, no GPU or model download required
- **scikit-learn** — RandomForestClassifier, train/test split, metrics
- **pandas / numpy** — data handling
- **pytest** — unit tests for feature extraction

---

## 📁 File Structure

```
📦 repo root
 ┣ 📜 eda.Rmd                          # Phase 1: R EDA notebook
 ┣ 📜 README.md                        # This file
 ┣ 📜 requirements.txt                 # Python deps for Phase 2
 ┣ 📂 src/
 ┃ ┣ 📜 generate_synthetic_data.py     # Synthetic Reddit-post generator (documented)
 ┃ ┣ 📜 features.py                    # Readability + sentiment feature extraction
 ┃ ┗ 📜 run_pipeline.py                # End-to-end training/comparison pipeline
 ┣ 📂 tests/
 ┃ ┗ 📜 test_features.py               # pytest coverage for feature extraction
 ┣ 📂 data/
 ┃ ┗ 📜 synthetic_reddit_posts.csv     # Generated dataset (or drop real_dataset.csv here)
 ┗ 📂 results/
   ┣ 📜 comparison_results.json        # Full metrics (machine-readable)
   ┗ 📜 comparison_report.md           # Human-readable results summary
```

---

## 🚀 How to Run

### Phase 1 — EDA (R)
```R
install.packages(c("tidyverse", "ggplot2", "dplyr", "readr", "corrplot", "skimr", "ggcorrplot", "knitr"))
```
Open `eda.Rmd` in RStudio and Knit, or run chunk-by-chunk.

### Phase 2 — Readability vs Sentiment Modeling (Python)
```bash
pip install -r requirements.txt
python -m pytest tests/ -q          # run unit tests
python src/run_pipeline.py          # generate data (if absent), extract features, train, compare
```
Outputs land in `results/comparison_results.json` and
`results/comparison_report.md`.

---

## 🔍 Methodology (Phase 2)

1. **Data:** 750 synthetic Reddit-style posts, 250 per severity band
   (0 = low, 1 = moderate, 2 = high). See "Honest note on data source" above.
2. **Readability features (7):** Flesch Reading Ease, Flesch-Kincaid Grade,
   SMOG Index, Gunning Fog, Automated Readability Index, word count,
   sentence count.
3. **Sentiment features (4):** VADER compound, positive, neutral, negative
   scores.
4. **Models:** three `RandomForestClassifier` models (300 trees, max depth 8,
   balanced class weights, same 75/25 stratified train/test split and random
   seed across all three) trained on:
   - (a) readability features only
   - (b) sentiment features only
   - (c) combined (readability + sentiment)
5. **Metrics reported:** accuracy, macro-F1, macro-averaged one-vs-rest
   ROC-AUC, and full per-class precision/recall/F1 (in the JSON output).

---

## 📈 Results (measured, from `results/comparison_report.md`)

Data source: `synthetic_reddit_posts.csv` (750 rows, balanced across 3 classes)

| Feature Set | Accuracy | Macro-F1 | Macro-AUC (OvR) | # Features |
|---|---|---|---|---|
| readability_only | 0.7128 | 0.7122 | 0.8714 | 7 |
| sentiment_only | 0.6649 | 0.6591 | 0.8565 | 4 |
| combined | 0.8245 | 0.8239 | 0.9365 | 11 |

**Interpretation:** on this dataset, **readability features alone were more
predictive of severity than sentiment features alone** (macro-F1 0.712 vs
0.659, roughly a 5-point gap; AUC gap smaller, ~0.015). This is plausible
because the synthetic generator ties severity to both structural complexity
(sentence/word patterns baked into each severity band's fragment bank) and
emotional tone, but the readability metrics also implicitly pick up on
post *length and structure* differences between bands that VADER's
per-post lexicon scoring does not capture as strongly once bands overlap in
emotional register (the 15% "register-noise" injection in the generator
adds positive-sounding sentences into high-severity posts, diluting VADER's
signal more than it dilutes the structural/readability signal). **Combining
both feature families outperformed either alone by a wide margin**
(macro-F1 0.824, +11 points over readability alone), showing the two
signals are complementary rather than redundant — consistent with the
premise of the original title that *comparing* them (rather than picking
one) is the more useful framing.

**Caveat to state explicitly in an interview:** these numbers describe how
well the modeling method recovers a synthetic, hand-authored signal. They
should not be quoted as real-world predictive performance on genuine Reddit
mental-health text. The pipeline, feature extraction, and evaluation code
are production-quality and would be applied unchanged to a real dataset
(e.g. dreaddit) — that substitution is the clearly identified next step.

---

## 🧪 Tests

`tests/test_features.py` covers:
- readability/sentiment feature extraction key/shape correctness
- graceful handling of empty-string input
- directional sanity checks (simple text reads easier than dense text;
  positive text scores higher than negative text)
- `build_feature_frame` output shape
- synthetic dataset generator balance and labeling

Run with `python -m pytest tests/ -q`.

---

## 💡 Applications

- Comparative feature-engineering methodology for text-based mental-health
  risk signals
- Template for swapping in a real labeled dataset (dreaddit, SWMH, Kaggle
  Reddit mental-health corpora) with no pipeline changes
- Demonstrates why comparing multiple linguistic signal families (not just
  picking one) tends to outperform any single family alone

---

## 📝 Notes

- Ensure R packages are installed before running `eda.Rmd`; ensure Python
  packages from `requirements.txt` are installed before running `src/`.
- Random seeds are fixed (42) throughout for reproducibility.
- To use a real dataset, add `data/real_dataset.csv` with columns
  `text,severity` — no code changes required.

## Credits

Developed by Kabilesh Rajaselvan with contributions from Jennita S.
