import os
import joblib
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Initialize NLTK
nltk.download('stopwords')
nltk.download('wordnet')

# ---- Model Loading ----
@st.cache_resource
def load_models():
    try:
        # Get current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Paths to model files in root directory
        model_path = os.path.join(current_dir, 'fake_news_model.pkl')
        vectorizer_path = os.path.join(current_dir, 'tfidf_vectorizer.pkl')
        
        # Debug: Show loading paths
        st.write(f"Loading model from: {model_path}")
        st.write(f"Loading vectorizer from: {vectorizer_path}")
        
        # Check if files exist
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        if not os.path.exists(vectorizer_path):
            raise FileNotFoundError(f"Vectorizer file not found at {vectorizer_path}")
            
        return (
            joblib.load(model_path),
            joblib.load(vectorizer_path)
        )
        
    except Exception as e:
        st.error(f"""
        ❌ Model Loading Failed: {str(e)}
        
        Required files must be in your root directory:
        - fake_news_model.pkl
        - tfidf_vectorizer.pkl
        
        Please verify:
        1. Both files are uploaded to your GitHub repo's main branch
        2. Filenames match exactly (case-sensitive)
        3. Files were committed and pushed to GitHub
        """)
        st.stop()

model, vectorizer = load_models()

# ---- Rest of your Streamlit app ----
# (Include your text preprocessing and UI code here)
