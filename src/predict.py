"""Interactive sentiment prediction on custom review text.

Usage:
    python src/predict.py                  # interactive prompt
    python src/predict.py --text "..."     # one-shot prediction
    python src/predict.py --demo           # run the built-in sample reviews
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib

from dataset import LABELS, PROJECT_ROOT

MODELS_DIR = PROJECT_ROOT / "models"

DEMO_REVIEWS = [
    "An absolute masterpiece. The acting was superb and the story kept me hooked till the end!",
    "Terrible waste of two hours. Boring plot, wooden acting and the dialogue was cringe.",
    "It was okay, nothing special. Some good scenes but overall pretty average.",
    "I loved every minute of it - funny, smart and beautifully shot. Highly recommend!",
    "Not what I expected at all. The product broke after a single week of use. Very disappointed.",
    "Great value for money, works exactly as described and arrived a day early.",
]

_pipeline = None


def load_pipeline(refresh: bool = False):
    """Load the saved best pipeline once and cache it."""
    global _pipeline
    if _pipeline is None or refresh:
        path = MODELS_DIR / "best_pipeline.joblib"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run 'python src/train.py' first."
            )
        _pipeline = joblib.load(path)
    return _pipeline


def predict_sentiment(text: str, pipeline=None) -> dict:
    """Classify a single review. Returns label, confidence and probabilities."""
    pipe = pipeline or load_pipeline()
    proba = pipe.predict_proba([text])[0]
    classes = list(pipe.classes_)
    best = int(max(range(len(classes)), key=lambda i: proba[i]))
    label_id = int(classes[best])
    return {
        "text": text,
        "label": LABELS[label_id],
        "label_id": label_id,
        "confidence": float(proba[best]),
        "probabilities": {
            LABELS[int(c)]: float(p) for c, p in zip(classes, proba)
        },
    }


def format_result(result: dict) -> str:
    p = result["probabilities"]
    return (
        f"  Sentiment : {result['label'].upper()}  "
        f"(confidence {result['confidence'] * 100:.1f}%)\n"
        f"  P(negative) = {p['negative'] * 100:.1f}%   "
        f"P(positive) = {p['positive'] * 100:.1f}%"
    )


def run_demo(pipeline=None) -> None:
    pipe = pipeline or load_pipeline()
    print("\nSample prediction log")
    print("-" * 78)
    for review in DEMO_REVIEWS:
        result = predict_sentiment(review, pipe)
        print(f"  > {review}")
        print(format_result(result))
        print("-" * 78)


def interactive_loop(pipeline=None) -> None:
    pipe = pipeline or load_pipeline()
    print("\nSentiment classifier ready. Type a review, or 'q' to quit.")
    while True:
        try:
            text = input("\nReview> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not text:
            continue
        if text.lower() in {"q", "quit", "exit"}:
            print("Bye.")
            break
        started = time.perf_counter()
        result = predict_sentiment(text, pipe)
        elapsed = (time.perf_counter() - started) * 1000
        print(format_result(result))
        print(f"  (inference {elapsed:.1f} ms)")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sentiment prediction CLI.")
    parser.add_argument("--text", type=str, default=None,
                        help="classify this review and exit")
    parser.add_argument("--demo", action="store_true",
                        help="classify the built-in sample reviews")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    pipe = load_pipeline()

    if args.text:
        print(format_result(predict_sentiment(args.text, pipe)))
        return 0
    if args.demo:
        run_demo(pipe)
        return 0

    interactive_loop(pipe)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
