# Emotion Classification from Text 😊

A machine learning web app that detects the emotion behind any sentence. Built with Python, scikit-learn, and Streamlit.

🔗 **Live Demo**: https://sentiment-classification-004.streamlit.app/

---

## What it does

You type a sentence, and the model predicts one of 6 emotions:

| Emotion | Emoji |
|---------|-------|
| Joy | 😊 |
| Sadness | 😢 |
| Anger | 😠 |
| Fear | 😨 |
| Love | ❤️ |
| Surprise | 😲 |

It also shows a **confidence score** and a **bar chart** of probabilities across all 6 emotions.

---

## How it works

1. **Text Preprocessing** — lowercasing, punctuation removal, digit removal, stopword removal (NLTK)
2. **Vectorization** — TF-IDF with bigrams (`max_features=5000`, `ngram_range=(1,2)`)
3. **Model** — Logistic Regression with `class_weight='balanced'` to handle class imbalance
4. **Accuracy** — ~82% on the test set

---

## Dataset

- 16,000 labeled sentences from `train.txt`
- Format: `sentence;emotion`
- 6 emotion classes: joy, sadness, anger, fear, love, surprise

---

## Project Structure

```
SentimentClassification_NLP/
├── app.py              # Streamlit web app
├── model.ipynb         # Training notebook
├── model.pkl           # Trained Logistic Regression model
├── vectorizer.pkl      # Fitted TF-IDF vectorizer
├── label_map.pkl       # Emotion label mapping
├── train.txt           # Training dataset
└── requirements.txt    # Dependencies
```

---

## Run Locally

```bash
# Clone the repo
git clone https://github.com/Harshit997/Sentiment-Classification.git
cd your-repo-name

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## Future Improvements

- Replace TF-IDF + Logistic Regression with a pretrained transformer (BERT/DistilBERT) for higher accuracy
- Add more training data for underrepresented emotions (surprise, love)
- Support multi-label emotion detection

---

## Tech Stack

- Python
- scikit-learn
- NLTK
- Streamlit
- Pandas
