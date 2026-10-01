"""Command-line sentiment prediction.

Usage:
    python predict.py "This movie was great"
    python predict.py            # interactive mode
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

from model import clean_text

HERE = Path(__file__).resolve().parent


def load_artifacts(model_path: Path, vectorizer_path: Path):
    if not model_path.is_file() or not vectorizer_path.is_file():
        raise SystemExit(
            "Model artifacts not found. Train them first:\n    python train.py"
        )
    with model_path.open("rb") as fh:
        model = pickle.load(fh)
    with vectorizer_path.open("rb") as fh:
        vectorizer = pickle.load(fh)
    return model, vectorizer


def predict(model, vectorizer, text: str) -> tuple[str, float]:
    """Return (label, confidence) using the model's own class ordering."""
    features = vectorizer.transform([clean_text(text)])
    label = model.predict(features)[0]
    classes = list(model.classes_)
    probabilities = model.predict_proba(features)[0]
    return label, float(probabilities[classes.index(label)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", nargs="*", help="Text to analyse")
    args = parser.parse_args()

    model, vectorizer = load_artifacts(HERE / "model.pkl", HERE / "vectorizer.pkl")

    if args.text:
        text = " ".join(args.text)
        label, confidence = predict(model, vectorizer, text)
        print(f"{label} ({confidence:.1%})")
        return

    print("Enter text to analyse. Type 'exit' or Ctrl-C to quit.")
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text.lower() in {"exit", "quit"}:
            break
        if not text.strip():
            continue
        label, confidence = predict(model, vectorizer, text)
        print(f"  {label} ({confidence:.1%})")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
