import streamlit as st
import pickle
import string
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data
nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

# ── Load model artifacts ───────────────────────────────────────────────────────
model      = pickle.load(open("model.pkl",      "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
label_map  = pickle.load(open("label_map.pkl",  "rb"))

# ── Text preprocessing ─────────────────────────────────────────────────────────
stop_words = set(stopwords.words('english'))

def preprocess(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = "".join(ch for ch in text if not ch.isdigit())
    words = word_tokenize(text)
    return " ".join([w for w in words if w not in stop_words])

# ── Emoji mapping ──────────────────────────────────────────────────────────────
EMOJI = {
    "joy":      "😊",
    "sadness":  "😢",
    "anger":    "😠",
    "fear":     "😨",
    "love":     "❤️",
    "surprise": "😲",
}

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Emotion Classifier", page_icon="😊")
st.title("Emotion Classification from Text")
st.write("Enter a sentence and the model will predict the emotion.")

# ── Input ──────────────────────────────────────────────────────────────────────
user_input = st.text_area("Enter your text here:")

if st.button("Predict Emotion"):
    if user_input.strip():
        # Preprocess → vectorize → predict
        cleaned   = preprocess(user_input)
        input_vec = vectorizer.transform([cleaned])

        # Predicted emotion
        prediction = model.predict(input_vec)[0]
        emotion    = label_map.get(prediction, "unknown")
        emoji      = EMOJI.get(emotion, "")

        # Confidence score
        proba      = model.predict_proba(input_vec)[0]
        confidence = proba.max() * 100

        # Show result
        st.success(f"Predicted Emotion: **{emotion.capitalize()}** {emoji}  —  {confidence:.1f}% confident")

        # Bar chart of all emotion probabilities
        st.subheader("Emotion Probabilities")
        prob_df = pd.DataFrame({
            "Emotion": [f"{label_map[i].capitalize()} {EMOJI.get(label_map[i], '')}" for i in range(len(label_map))],
            "Probability (%)": [round(p * 100, 1) for p in proba]
        }).set_index("Emotion")

        st.bar_chart(prob_df)

    else:
        st.warning("Please enter some text.")