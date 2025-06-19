import os
import joblib
import re
import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK datasets (only once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load model and vectorizer (replace paths as needed)
@st.cache_resource
def load_models():
    model = joblib.load('fake_news_model.pkl')  # Replace with your path
    vectorizer = joblib.load('tfidf_vectorizer.pkl')  # Replace with your path
    return model, vectorizer

model, tfidf_vectorizer = load_models()
lemmatizer = WordNetLemmatizer()

# Text preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # Remove special chars/numbers
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

# Prediction function
def predict(text):
    try:
        processed_text = preprocess_text(text)
        features = tfidf_vectorizer.transform([processed_text]).toarray()
        prediction = model.predict(features)[0]
        
        if prediction == 0:
            st.error("#### This is likely FAKE NEWS! ⚠️")
            return "FAKE"
        else:
            st.success("#### This appears to be REAL NEWS. ✅")
            return "REAL"
            
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

# Streamlit UI
st.title("Fake News Detector")
user_input = st.text_area("Enter news article text:", height=200)

if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter some text!")
    elif len(user_input.split()) < 3:
        st.warning("Enter at least 3 words for accurate results.")
    else:
        result = predict(user_input)

make its correct
