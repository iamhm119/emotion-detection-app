# 🎭 Emotion Detection using NLP & Machine Learning

This project is a **Text-based Emotion Detection system** that classifies a given sentence into one of six human emotions using **Natural Language Processing (NLP)** and **Machine Learning** techniques.

The application is implemented with a clean NLP pipeline and a simple Streamlit-based interface for local interaction.

---

## 🔍 Emotions Supported
- 😊 Joy  
- 😨 Fear  
- 😡 Anger  
- ❤️ Love  
- 😢 Sadness  
- 😲 Surprise  

---

## 🧠 Methodology

1. **Text Preprocessing**
   - Lowercasing
   - Removal of non-alphabetic characters
   - Stopword removal using NLTK
   - Stemming using Porter Stemmer

2. **Feature Extraction**
   - TF-IDF (Term Frequency–Inverse Document Frequency) Vectorization

3. **Model**
   - Logistic Regression (multi-class classification)

The model is trained using classical machine learning techniques to ensure:
- Fast inference
- Interpretability
- Efficient performance on medium-sized datasets

---

## 🛠️ Tech Stack

- Python
- Scikit-learn
- NLTK
- NumPy
- Streamlit (for local UI)

---


---

## ▶️ Run Locally

To run the application locally:

```bash
pip install -r requirements.txt
streamlit run app.py


