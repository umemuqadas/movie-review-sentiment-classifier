"""Audit the project against every requirement and acceptance criterion.

Run:  python verify_project.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

from preprocess import TextPreprocessor

RESULTS: list[tuple[str, bool, str]] = []


def check(section: str, name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((section, name, bool(ok), detail))


# ---------------------------------------------------------------- core ----
pre = joblib.load(ROOT / "models" / "text_preprocessor.joblib").fit(["warm up"])

# 1. lowercasing
out = pre.transform(["This Movie Was GREAT"])[0]
check("Core", "Lowercasing", "GREAT" not in out and "great" in out, f"-> {out!r}")

# 2. punctuation removal
out = pre.transform(["amazing!! worth every penny, 10/10."])[0]
check("Core", "Punctuation removal", not re.search(r"[^\w\s]", out), f"-> {out!r}")

# 3. stop-word filtering
out = pre.transform(["the movie was really very good"])[0]
check(
    "Core",
    "Stop-word filtering",
    "the" not in out.split() and "was" not in out.split() and "good" in out,
    f"-> {out!r}",
)

# 4. TF-IDF vectorization inside the pipeline
best = joblib.load(ROOT / "models" / "best_pipeline.joblib")
vec = joblib.load(ROOT / "models" / "tfidf_vectorizer.joblib")
check(
    "Core",
    "TF-IDF vectorization",
    "tfidf" in best.named_steps and hasattr(vec, "idf_"),
    f"{len(vec.vocabulary_):,} features, {vec.ngram_range} ngrams",
)

# 5. both algorithms trained + compared
import pandas as pd

metrics = pd.read_csv(ROOT / "outputs" / "metrics.csv")
names = set(metrics["Model"])
check(
    "Core",
    "Model comparison (Logistic Regression vs Multinomial NB)",
    {"Logistic Regression", "Multinomial Naive Bayes"} <= names,
    " / ".join(sorted(names)),
)

# 6. all four metrics reported
needed = {"Accuracy", "Precision (macro)", "Recall (macro)", "F1-Score (macro)"}
check(
    "Core",
    "Accuracy / Precision / Recall / F1 reported",
    needed <= set(metrics.columns),
    ", ".join(c for c in metrics.columns if c != "Model"),
)

# 7. confusion matrix visualisations
cms = sorted((ROOT / "outputs").glob("confusion_matrix_*.png"))
check(
    "Core",
    "Confusion Matrix visualisation",
    len(cms) >= 2 and all(p.stat().st_size > 5_000 for p in cms),
    ", ".join(p.name for p in cms),
)

# 8. interactive prediction function
import predict as predict_mod

check(
    "Core",
    "Interactive prediction function",
    callable(getattr(predict_mod, "predict_sentiment", None))
    and callable(getattr(predict_mod, "interactive_loop", None)),
    "predict_sentiment() + interactive_loop()",
)

# ---------------------------------------------------------- deliverables --
scripts = sorted((ROOT / "src").glob("*.py"))
check("Deliverable", "Complete model training script", (ROOT / "src" / "train.py").is_file(),
      ", ".join(p.name for p in scripts))

arts = sorted((ROOT / "models").glob("*.joblib"))
check(
    "Deliverable",
    "Saved vectorizer + model pipeline (joblib)",
    (ROOT / "models" / "best_pipeline.joblib").is_file()
    and (ROOT / "models" / "tfidf_vectorizer.joblib").is_file()
    and any("model" in p.name for p in arts),
    ", ".join(p.name for p in arts),
)

readme = (ROOT / "README.md").read_text(encoding="utf-8")
has_table = "Precision" in readme and "F1-Score" in readme and "0.894" in readme
has_log = "Sentiment : POSITIVE" in readme and "Sentiment : NEGATIVE" in readme
check("Deliverable", "README: classification metrics table", has_table)
check("Deliverable", "README: sample prediction logs", has_log)

# ---------------------------------------------------------- acceptance ----
# balanced precision / recall recomputed on the held-out test set
from dataset import load_imdb
from sklearn.metrics import precision_score, recall_score

_, test = load_imdb(shuffle=False)
pred = best.predict(test["text"])
y_true = test["label"]
p_neg = precision_score(y_true, pred, pos_label=0)
r_neg = recall_score(y_true, pred, pos_label=0)
p_pos = precision_score(y_true, pred, pos_label=1)
r_pos = recall_score(y_true, pred, pos_label=1)
spread = max(abs(p_neg - p_pos), abs(r_neg - r_pos)) * 100
check(
    "Acceptance",
    "Balanced precision & recall on held-out test set",
    spread < 2.0,
    f"neg P={p_neg:.4f} R={r_neg:.4f} | pos P={p_pos:.4f} R={r_pos:.4f} "
    f"| max gap {spread:.2f} pp (n={len(test):,})",
)

# no preprocessing data leakage: saved vocabulary must equal a vectorizer fit
# on the TRAIN split only, and must differ from one fit on the TEST split.
train, _ = load_imdb(shuffle=False)
fitted_pre = TextPreprocessor().fit(train["text"])
fresh_train = TfidfVectorizer(
    max_features=50_000, ngram_range=(1, 2), min_df=2, sublinear_tf=True
).fit(fitted_pre.transform(train["text"]))

fitted_pre_test = TextPreprocessor().fit(test["text"])
fresh_test = TfidfVectorizer(
    max_features=50_000, ngram_range=(1, 2), min_df=2, sublinear_tf=True
).fit(fitted_pre_test.transform(test["text"]))

same_as_train = fresh_train.vocabulary_ == vec.vocabulary_
differs_from_test = fresh_test.vocabulary_ != vec.vocabulary_
check(
    "Acceptance",
    "No preprocessing data leakage (TF-IDF fit on train only)",
    same_as_train and differs_from_test,
    f"matches train-only fit={same_as_train}, differs from test-only fit={differs_from_test}",
)

# real-time inference on brand-new inputs
samples = {
    "Absolutely loved it, would watch again!": "positive",
    "Awful, a complete waste of time and money.": "negative",
}
ok = True
log = []
for text, expected in samples.items():
    r = predict_mod.predict_sentiment(text, best)
    ok &= r["label"] == expected
    log.append(f"{r['label']} ({r['confidence']:.0%}) <- {text[:46]}...")
check(
    "Acceptance",
    "Real-time inference on new user inputs",
    ok,
    " | ".join(log),
)

# ----------------------------------------------------------------- report -
width = max(len(n) for _, n, _, _ in RESULTS)
section = None
print()
for sec, name, ok, detail in RESULTS:
    if sec != section:
        section = sec
        print(f"\n{section.upper()}")
        print("-" * 78)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:<{width}}  {detail}")

failed = [n for _, n, ok, _ in RESULTS if not ok]
print("\n" + "=" * 78)
print(
    f"{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed"
    + ("" if not failed else f"  -> FAILED: {', '.join(failed)}")
)
print("=" * 78)
raise SystemExit(1 if failed else 0)
