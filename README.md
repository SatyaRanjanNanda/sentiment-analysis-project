# Sentiment Analysis Tool

A machine learning web app that classifies text as **positive** or **negative**
using TF-IDF features and a Logistic Regression classifier trained on the IMDB
movie reviews dataset.

---

## Features

- 🔍 Real-time sentiment prediction with a confidence score
- 📊 Interactive probability bar chart
- 🧹 NLTK stopword-based text cleaning
- 💬 Chat-style Streamlit interface, plus a CLI
- 🧪 One-command retraining with a held-out test split
- ⚠️ Warns when the model is unsure, instead of overstating the result

## Tech stack

Python · scikit-learn · NLTK · pandas · NumPy · Streamlit

## Project layout

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit UI |
| `train.py` | Trains and evaluates the model, writes the pickles |
| `predict.py` | CLI prediction, single-shot or interactive |
| `model.py` | Shared text-cleaning function |
| `model.pkl` / `vectorizer.pkl` | Trained artifacts (committed so the app runs without retraining) |
| `runtime.txt` | Python version for Streamlit Cloud |

## Setup

```bash
git clone https://github.com/SatyaRanjanNanda/sentiment-analysis-project.git
cd sentiment-analysis-project

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

The first run downloads the NLTK `stopwords` corpus automatically.

## Run the app

```bash
streamlit run app.py
```

Open http://localhost:8501.

## Use the CLI

```bash
python predict.py "This movie was absolutely brilliant"
# positive (92.4%)

python predict.py     # interactive; type 'exit' to quit
```

## Retraining

The dataset is **not** committed (it is ~66 MB). Download
[IMDB Dataset.csv](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-movie-reviews-50k)
from Kaggle and place it in the project root, then:

```bash
python train.py                      # 2,000 samples, trains in seconds
python train.py --samples 50000      # larger run
```

The script prints accuracy and a classification report, then overwrites
`model.pkl` and `vectorizer.pkl`. Commit them if you change the model.

> **Note:** the TF-IDF vectorizer is fitted on the training split only. An
> earlier version of this script fitted it before splitting, which leaked
> test-set information into the feature weights.

## How it works

1. Text is lowercased, stripped of non-alphabetic characters, and filtered
   against NLTK English stopwords.
2. `TfidfVectorizer` (5,000 features) converts it to a sparse vector.
3. `LogisticRegression` outputs a class label and probabilities.
4. The UI shows the winning label, its confidence, and the full distribution.

## Limitations

- Binary only — there is no neutral class, so mixed or factual text is forced
  into one bucket.
- Trained on English movie reviews, so it transfers poorly to product reviews,
  social media, sarcasm, or other languages.
- The cleaning in `model.py` must stay in sync with whatever version trained
  `vectorizer.pkl`. Change it and retrain.

## Deploying to Streamlit Cloud

1. Push the repo to GitHub.
2. In [share.streamlit.io](https://share.streamlit.io), point the app at this
   repo and `app.py`.
3. No secrets are required — the model ships with the repo.

## Credits

Built and maintained by **SatyaRanjanNanda**.

## License

**No license file is present in this repository.** Default copyright applies —
no one may legally reuse, modify, or redistribute this code without permission.
