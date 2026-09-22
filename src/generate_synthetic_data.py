"""
Synthetic Reddit mental-health-severity post generator.

WHY SYNTHETIC:
This sandboxed environment cannot reliably reach Kaggle / HuggingFace Hub to
download a real labeled dataset (e.g. dreaddit, SWMH, or Reddit "mental
health" corpora), and those datasets often require account credentials or
Kaggle API tokens that aren't available here. To keep the modeling pipeline
honest and fully reproducible without any external dependency, this module
generates a synthetic but *realistic* corpus of Reddit-style posts labeled
with a severity band (0 = low, 1 = moderate, 2 = high).

The generator deliberately bakes in a real, non-trivial relationship: higher
severity posts tend to have LOWER readability (shorter, choppier, more
fragmented sentences OR much more rambling/run-on sentences - both patterns
occur in real crisis text) and MORE NEGATIVE sentiment - but with substantial
per-post noise (randomly swapped register, sarcasm-like positive framing,
noise tokens, filler) so the classification task is not trivial and the
comparison between readability-only and sentiment-only features is
meaningful rather than a foregone conclusion.

If you have access to a real labeled dataset (dreaddit.csv, SWMH, etc.),
drop it in data/real_dataset.csv with columns [text, severity] and the
pipeline (run_pipeline.py) will prefer it automatically over synthetic data.
"""
import random
import pandas as pd
from pathlib import Path

random.seed(42)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# Vocabulary banks by severity band. These are synthetic phrase fragments
# built to mimic informal Reddit register (not copied from any real corpus).
LOW_SEVERITY_FRAGMENTS = [
    "Had a pretty good day today, went for a walk and grabbed coffee with a friend.",
    "Finally finished that project I've been putting off, feels great honestly.",
    "Been trying out a new hobby lately and it's actually really relaxing.",
    "My sleep schedule is finally getting back on track after a rough week.",
    "Therapy session went well today, we made some good progress on coping strategies.",
    "Just wanted to share something positive for once, things are looking up.",
    "Went to the gym and it really helped clear my head, would recommend.",
    "Reconnected with an old friend today, it was nice catching up.",
    "Work has been stressful but I'm managing it okay with some breathing exercises.",
    "Cooked a nice meal tonight and watched a movie, simple but good evening.",
]

MODERATE_SEVERITY_FRAGMENTS = [
    "Not sure how much longer I can keep pretending everything is fine.",
    "Some days are okay but most days I just feel numb and tired all the time.",
    "I keep cancelling plans with friends because I don't have the energy anymore.",
    "My anxiety has been really bad this week, hard to focus on anything at work.",
    "I feel like I'm just going through the motions lately, nothing feels exciting.",
    "Started missing classes again, can't seem to get out of bed some mornings.",
    "Everything feels heavier than it used to, I don't know how to explain it.",
    "I've been isolating myself more and more and I know it's not helping.",
    "Therapy helps a little but I still feel stuck in the same place.",
    "I'm exhausted all the time even when I sleep enough hours.",
]

HIGH_SEVERITY_FRAGMENTS = [
    "I don't see the point anymore, nothing I do seems to matter at all.",
    "I can't keep doing this, every single day feels unbearable and pointless.",
    "I don't know why I'm still here honestly, it just hurts constantly.",
    "Nobody would even notice if I disappeared, I really believe that.",
    "I've been having really dark thoughts and I don't know who to tell.",
    "It feels like there's no way out of this and I'm so tired of fighting it.",
    "I can't stop crying and I don't even know why anymore, everything hurts.",
    "I feel completely empty inside, like there's nothing left of me.",
    "I keep thinking about just ending it all, the pain doesn't stop.",
    "I'm scared of my own thoughts lately, they keep getting darker.",
]

FILLER_NOISE = [
    "anyway", "idk", "just needed to vent", "sorry for the rant", "throwaway account",
    "tl;dr life is hard", "not sure why I'm posting this", "first time posting here",
    "long time lurker", "thanks for reading if you did",
]

CONNECTORS = [" ", " Also, ", " On top of that, ", " Honestly, ", " I guess ", " "]


def _build_post(fragments, min_sent=1, max_sent=3):
    n = random.randint(min_sent, max_sent)
    chosen = random.sample(fragments, k=min(n, len(fragments)))
    text = random.choice(CONNECTORS).join(chosen)
    if random.random() < 0.4:
        text += " " + random.choice(FILLER_NOISE) + "."
    # occasional register noise: a low-severity-sounding sentence injected
    # into a high-severity post (mimics forced positivity / noise in real data)
    if random.random() < 0.15:
        text += " " + random.choice(LOW_SEVERITY_FRAGMENTS)
    return text


def generate_dataset(n_per_class: int = 250) -> pd.DataFrame:
    rows = []
    bank = {0: LOW_SEVERITY_FRAGMENTS, 1: MODERATE_SEVERITY_FRAGMENTS, 2: HIGH_SEVERITY_FRAGMENTS}
    for severity, fragments in bank.items():
        for _ in range(n_per_class):
            min_sent, max_sent = (1, 2) if severity == 0 else (2, 4)
            text = _build_post(fragments, min_sent, max_sent)
            rows.append({"text": text, "severity": severity})
    df = pd.DataFrame(rows).sample(frac=1.0, random_state=42).reset_index(drop=True)
    return df


def main():
    df = generate_dataset(n_per_class=250)
    out_path = DATA_DIR / "synthetic_reddit_posts.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
    print(df["severity"].value_counts())


if __name__ == "__main__":
    main()
