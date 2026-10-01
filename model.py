"""Text preprocessing shared by training and inference.

The cleaning behaviour here must stay byte-for-byte consistent with the version
used to fit `vectorizer.pkl`, otherwise predictions from the committed model
silently degrade. Retrain the model if you change any of it.
"""

from __future__ import annotations

import re
from functools import lru_cache

_NON_ALPHA = re.compile(r"[^a-zA-Z]")


@lru_cache(maxsize=1)
def _stop_words() -> frozenset[str]:
    """English stopwords, fetched on first use.

    Importing this module must never raise, so a missing NLTK corpus degrades to
    "no stopword removal" instead of crashing the app at import time.
    """
    try:
        import nltk

        nltk.data.find("corpora/stopwords")
    except LookupError:
        try:
            import nltk

            nltk.download("stopwords", quiet=True)
            nltk.data.find("corpora/stopwords")
        except Exception:
            return frozenset()
    except ImportError:
        return frozenset()

    try:
        from nltk.corpus import stopwords

        return frozenset(stopwords.words("english"))
    except Exception:
        return frozenset()


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    text = text.lower()
    text = _NON_ALPHA.sub(" ", text)
    words = [w for w in text.split() if w not in _stop_words()]
    return " ".join(words)
