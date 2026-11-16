"""Train, compare and evaluate sentiment classifiers.

Compares Logistic Regression vs Multinomial Naive Bayes over a TF-IDF feature
space built by the preprocessing pipeline.

Usage:
    python src/train.py            # full training + evaluation
    python src/train.py --quick    # small subset, fast smoke test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from dataset import LABELS, RANDOM_STATE, download_dataset, load_imdb
from preprocess import TextPreprocessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
TARGET_NAMES = ["negative", "positive"]


def slugify(name: str) -> str:
    return name.lower().replace(" ", "_")


def build_pipeline(estimator) -> Pipeline:
    """preprocess -> TF-IDF -> classifier, fit end-to-end on training data only."""
    return Pipeline(
        [
            ("preprocess", TextPreprocessor()),
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=50_000,
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            ("clf", estimator),
        ]
    )


def model_registry() -> dict:
    return {
        "Logistic Regression": lambda: build_pipeline(
            LogisticRegression(
                max_iter=1_000, C=1.0, solver="lbfgs", random_state=RANDOM_STATE
            )
        ),
        "Multinomial Naive Bayes": lambda: build_pipeline(MultinomialNB(alpha=1.0)),
    }


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision (macro)": precision_score(y_true, y_pred, average="macro"),
        "Recall (macro)": recall_score(y_true, y_pred, average="macro"),
        "F1-Score (macro)": f1_score(y_true, y_pred, average="macro"),
        "Precision (neg)": precision_score(y_true, y_pred, pos_label=0),
        "Recall (neg)": recall_score(y_true, y_pred, pos_label=0),
        "Precision (pos)": precision_score(y_true, y_pred, pos_label=1),
        "Recall (pos)": recall_score(y_true, y_pred, pos_label=1),
    }


def plot_confusion_matrix(y_true, y_pred, title: str, path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
        annot_kws={"size": 15},
        ax=ax,
    )
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label", fontsize=12)
    ax.set_title(f"Confusion Matrix \u2014 {title}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train sentiment classifiers.")
    parser.add_argument("--quick", type=int, default=0, metavar="N",
                        help="use N reviews per class for a fast smoke test")
    parser.add_argument("--skip-download", action="store_true",
                        help="assume the IMDB corpus is already in data/")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    sns.set_theme(style="whitegrid")

    if not args.skip_download:
        download_dataset()
    train_df, test_df = load_imdb()

    if args.quick:
        n = args.quick
        train_df = pd.concat(
            [train_df[train_df.label == 0].head(n), train_df[train_df.label == 1].head(n)]
        ).sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
        test_df = pd.concat(
            [test_df[test_df.label == 0].head(max(n // 4, 25)),
             test_df[test_df.label == 1].head(max(n // 4, 25))]
        ).sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

    X_train, y_train = train_df["text"], train_df["label"]
    X_test, y_test = test_df["text"], test_df["label"]

    print(f"Training reviews : {len(X_train):,}  {y_train.value_counts().to_dict()}")
    print(f"Test reviews     : {len(X_test):,}  {y_test.value_counts().to_dict()}")

    # Validation split carved out of the TRAINING set only - used for model
    # selection. The official test split stays untouched until final scoring.
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=RANDOM_STATE, stratify=y_train
    )
    print(f"Fit / validation : {len(X_fit):,} / {len(X_val):,}\n")

    # ---- Stage 1: fit on the training fold, compare on validation ----------
    val_rows, fitted = [], {}
    for name, factory in model_registry().items():
        started = time.perf_counter()
        pipe = factory()
        pipe.fit(X_fit, y_fit)  # TF-IDF statistics learned from X_fit only
        metrics = compute_metrics(y_val, pipe.predict(X_val))
        elapsed = time.perf_counter() - started
        fitted[name] = pipe
        val_rows.append({"Model": name, **metrics, "Train time (s)": round(elapsed, 1)})
        print(f"[val] {name:<26} F1={metrics['F1-Score (macro)']:.4f} "
              f"Acc={metrics['Accuracy']:.4f}  ({elapsed:.1f}s)")

    val_table = pd.DataFrame(val_rows).set_index("Model")
    best_name = val_table["F1-Score (macro)"].idxmax()
    print(f"\nSelected on validation F1: {best_name}")

    # ---- Stage 2: refit on the FULL official train set, score ONCE on test -
    MODELS_DIR.mkdir(exist_ok=True)
    OUTPUTS_DIR.mkdir(exist_ok=True)

    test_rows, final = [], {}
    for name, factory in model_registry().items():
        pipe = factory()
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        metrics = compute_metrics(y_test, y_pred)
        report = classification_report(
            y_test, y_pred, target_names=TARGET_NAMES, digits=4
        )
        final[name] = {"pipeline": pipe, "pred": y_pred, "metrics": metrics,
                       "report": report}
        test_rows.append({"Model": name, **metrics})

        slug = slugify(name)
        (OUTPUTS_DIR / f"classification_report_{slug}.txt").write_text(
            f"{name}\n{'=' * len(name)}\n\n{report}", encoding="utf-8"
        )
        plot_confusion_matrix(
            y_test, y_pred, name, OUTPUTS_DIR / f"confusion_matrix_{slug}.png"
        )
        joblib.dump(pipe.named_steps["clf"], MODELS_DIR / f"{slug}_model.joblib")
        print(f"[test] {name:<26} F1={metrics['F1-Score (macro)']:.4f} "
              f"Acc={metrics['Accuracy']:.4f}")

    # ---- Persist artifacts -------------------------------------------------
    best_pipeline = final[best_name]["pipeline"]
    joblib.dump(best_pipeline.named_steps["preprocess"],
                MODELS_DIR / "text_preprocessor.joblib")
    joblib.dump(best_pipeline.named_steps["tfidf"],
                MODELS_DIR / "tfidf_vectorizer.joblib")
    joblib.dump(best_pipeline, MODELS_DIR / "best_pipeline.joblib")

    test_table = pd.DataFrame(test_rows).set_index("Model")
    test_table.round(4).to_csv(OUTPUTS_DIR / "metrics.csv")

    metadata = {
        "best_model": best_name,
        "selection_split": "validation (10% of official train set)",
        "evaluation_split": "official IMDB test split (held out)",
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "metrics": {k: {m: round(v, 4) for m, v in v_["metrics"].items()}
                    for k, v_ in final.items()},
        "artifacts": {
            "pipeline": "models/best_pipeline.joblib",
            "vectorizer": "models/tfidf_vectorizer.joblib",
            "preprocessor": "models/text_preprocessor.joblib",
            "models": {slugify(n): f"models/{slugify(n)}_model.joblib"
                       for n in model_registry()},
        },
    }
    (MODELS_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 78)
    print("HELD-OUT TEST SET RESULTS")
    print("=" * 78)
    print(test_table.round(4).to_string())
    print(f"\nArtifacts saved to {MODELS_DIR}")
    print(f"Reports/plots saved to {OUTPUTS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
