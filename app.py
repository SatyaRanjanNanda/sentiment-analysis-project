"""Streamlit UI for the sentiment analysis tool."""

from __future__ import annotations

import pickle
from pathlib import Path

import streamlit as st

from model import clean_text

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "model.pkl"
VECTORIZER_PATH = HERE / "vectorizer.pkl"

EXAMPLES = [
    "This movie is absolutely brilliant, I loved every minute of it.",
    "A complete waste of time. The plot made no sense at all.",
    "The acting was fine but the pacing dragged badly.",
    "I would watch this again. One of the best films this year.",
]

st.set_page_config(page_title="Sentiment Analyzer", page_icon="💬", layout="centered")


@st.cache_resource(show_spinner="Loading model...")
def load_artifacts():
    """Load the pickles once per server process."""
    with MODEL_PATH.open("rb") as fh:
        model = pickle.load(fh)
    with VECTORIZER_PATH.open("rb") as fh:
        vectorizer = pickle.load(fh)
    return model, vectorizer


def predict(model, vectorizer, text: str) -> dict[str, float]:
    """Confidence per label, keyed by label name rather than column position."""
    features = vectorizer.transform([clean_text(text)])
    classes = list(model.classes_)
    probabilities = model.predict_proba(features)[0]
    return dict(zip(classes, (float(p) for p in probabilities)))


def main() -> None:
    st.markdown(
        "<h1 style='text-align: center;'>💬 Sentiment Analysis Tool</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align: center; color: gray;'>"
        "Classify text as positive or negative using a TF-IDF + "
        "Logistic Regression model trained on IMDB movie reviews.</p>",
        unsafe_allow_html=True,
    )

    if not (MODEL_PATH.is_file() and VECTORIZER_PATH.is_file()):
        st.error(
            "Model artifacts are missing. Run `python train.py` to generate "
            "`model.pkl` and `vectorizer.pkl`, then restart the app."
        )
        st.stop()

    try:
        model, vectorizer = load_artifacts()
    except Exception as exc:
        st.error(f"Could not load the model: {exc}")
        st.stop()

    st.write("---")
    text = st.text_area("✍️ Enter your text here:", height=150, key="input_text")

    with st.expander("Try an example"):
        for example in EXAMPLES:
            if st.button(example, key=f"ex_{example[:20]}"):
                st.session_state["input_text"] = example
                st.rerun()

    if st.button("🔍 Analyze Sentiment", type="primary"):
        if not text.strip():
            st.warning("⚠️ Please enter some text.")
        else:
            with st.spinner("Analyzing..."):
                try:
                    scores = predict(model, vectorizer, text)
                except Exception as exc:
                    st.error(f"Error during prediction: {exc}")
                    st.stop()

            label = max(scores, key=scores.get)
            confidence = scores[label]

            if label == "positive":
                st.success(f"😊 **Positive** — {confidence:.1%} confidence")
            else:
                st.error(f"😠 **Negative** — {confidence:.1%} confidence")

            if confidence < 0.65:
                st.info(
                    "Low confidence — the model is close to calling this one. "
                    "Try rephrasing or adding more context."
                )

            st.subheader("📊 Sentiment Confidence")
            st.bar_chart(
                {"Sentiment": list(scores), "Probability": list(scores.values())},
                x="Sentiment",
                y="Probability",
            )

    st.write("---")
    st.caption(
        "Binary classifier trained on IMDB movie reviews. It is tuned for "
        "movie-review language and is not tuned for sarcasm, mixed opinions, "
        "or non-English text."
    )


if __name__ == "__main__":
    main()
