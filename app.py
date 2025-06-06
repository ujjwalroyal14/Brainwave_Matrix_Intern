import os
import joblib
import streamlit as st

# Load models with error handling
@st.cache_resource
def load_models():
    try:
        model_path = os.path.join(os.path.dirname(__file__), 'models/fake_news_model.pkl')
        vectorizer_path = os.path.join(os.path.dirname(__file__), 'models/tfidf_vectorizer.pkl')
        return (
            joblib.load(model_path),
            joblib.load(vectorizer_path)
        )
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        st.stop()

model, vectorizer = load_models()
