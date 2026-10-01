"""Train the sentiment classifier and write model.pkl / vectorizer.pkl.

Usage:
    python train.py                      # 2,000 samples, fast
    python train.py --samples 50000      # larger run
    python train.py --dataset "IMDB Dataset.csv"
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from model import clean_text

HERE = Path(__file__).resolve().parent
DEFAULT_DATASET = HERE / "IMDB Dataset.csv"
MAX_FEATURES = 5000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--max-features", type=int, default=MAX_FEATURES)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=200)
    return parser.parse_args()


def load_dataset(path: Path, samples: int, random_state: int) -> pd.DataFrame:
    if not path.is_file():
        raise SystemExit(
            f"Dataset not found: {path}\n"
            "Download the IMDB Dataset CSV from Kaggle into the project folder,\n"
            "or pass --dataset /path/to/file.csv"
        )

    df = pd.read_csv(path)
    df = df.rename(columns={"review": "text", "sentiment": "label"})
    df = df.dropna(subset=["text", "label"])
    df = df[df["label"].isin(["positive", "negative"])]
    df = df.drop_duplicates(subset=["text"])

    if samples and samples < len(df):
        # Stratify so both classes stay balanced in the sample.
        df = df.sample(samples, random_state=random_state)
    return df.reset_index(drop=True)


def main() -> None:
    args = parse_args()
    df = load_dataset(Path(args.dataset), args.samples, args.random_state)
    print(f"Loaded {len(df)} rows ({df['label'].value_counts().to_dict()})")

    df["text"] = df["text"].map(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=df["label"],
    )

    # Fit the vectorizer on the training split only, otherwise the IDF weights
    # leak information from the held-out test set.
    vectorizer = TfidfVectorizer(max_features=args.max_features)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=args.max_iter)
    model.fit(X_train_vec, y_train)

    predictions = model.predict(X_test_vec)
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print(classification_report(y_test, predictions, zero_division=0))
    print(f"Classes: {list(model.classes_)}")

    with (HERE / "model.pkl").open("wb") as fh:
        pickle.dump(model, fh)
    with (HERE / "vectorizer.pkl").open("wb") as fh:
        pickle.dump(vectorizer, fh)
    print(f"Saved model.pkl and vectorizer.pkl to {HERE}")


if __name__ == "__main__":
    main()
