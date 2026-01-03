# ======================== IMPORT PACKAGES =========================
import streamlit as st
import numpy as np
import re
from nltk.stem import PorterStemmer
import pickle
import nltk

# ======================== NLTK SETUP ===============================
nltk.download('stopwords')
stopwords = set(nltk.corpus.stopwords.words('english'))

# ======================== LOAD MODELS ==============================
lg = pickle.load(open('logistic_regression.pkl','rb'))
tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl','rb'))
lb = pickle.load(open('label_encoder.pkl','rb'))

# ======================== FUNCTIONS ================================
def clean_text(text):
    stemmer = PorterStemmer()
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower()
    text = text.split()
    text = [stemmer.stem(word) for word in text if word not in stopwords]
    return " ".join(text)

def predict_emotion(input_text):
    cleaned_text = clean_text(input_text)
    input_vectorized = tfidf_vectorizer.transform([cleaned_text])

    probs = lg.predict_proba(input_vectorized)[0]
    pred_index = np.argmax(probs)

    predicted_emotion = lb.inverse_transform([pred_index])[0]
    confidence = probs[pred_index]

    return predicted_emotion, confidence

# ======================== PAGE CONFIG ==============================
st.set_page_config(
    page_title="Emotion Detection App",
    page_icon="🎭",
    layout="centered"
)

# ======================== SIDEBAR ==============================
st.sidebar.title("ℹ️ About the App")
st.sidebar.write("""
This app detects **human emotions from text** using:

- TF-IDF Vectorization  
- Logistic Regression  
- NLP preprocessing (NLTK)

**Supported emotions:**
- 😊 Joy
- 😨 Fear
- 😡 Anger
- ❤️ Love
- 😢 Sadness
- 😲 Surprise
""")

st.sidebar.markdown("---")
st.sidebar.write("Developed by **Hem Modi**")

# ======================== MAIN UI ==============================
st.markdown(
    "<h1 style='text-align: center;'>🎭 Emotion Detection System</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center; font-size:18px;'>"
    "Enter a sentence and let AI detect the underlying emotion"
    "</p>",
    unsafe_allow_html=True
)

st.markdown("---")

# ======================== INPUT ==============================
user_input = st.text_area(
    "📝 Enter your text:",
    height=120,
    placeholder="Example: I am feeling very excited about my new job!"
)

# ======================== PREDICTION ==============================
if st.button("🔍 Predict Emotion", use_container_width=True):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text before predicting.")
    else:
        with st.spinner("Analyzing emotion..."):
            emotion, confidence = predict_emotion(user_input)

        # Emotion emojis
        emoji_map = {
            "joy": "😊",
            "fear": "😨",
            "anger": "😡",
            "love": "❤️",
            "sadness": "😢",
            "surprise": "😲"
        }

        emoji = emoji_map.get(emotion.lower(), "🎭")

        st.markdown("---")
        st.subheader("📊 Prediction Result")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                f"<h2>{emoji} {emotion.capitalize()}</h2>",
                unsafe_allow_html=True
            )

        with col2:
            st.metric(
                label="Confidence",
                value=f"{confidence*100:.2f}%"
            )

        st.progress(float(confidence))

        st.success("✅ Emotion detected successfully!")

# ======================== FOOTER ==============================
st.markdown("---")
st.markdown(
    "<p style='text-align:center; font-size:14px;'>"
    "Built with ❤️ using NLP & Machine Learning"
    "</p>",
    unsafe_allow_html=True
)
