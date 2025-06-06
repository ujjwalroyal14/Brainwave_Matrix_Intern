import streamlit as st
import pickle
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# ---- Model Loading with Robust Error Handling ----
@st.cache_resource
def load_model():
    try:
        # Get absolute path to model files
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, 'models', 'fake_news_model.pkl')
        vectorizer_path = os.path.join(base_dir, 'models', 'tfidf_vectorizer.pkl')
        
        # Verify files exist
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        if not os.path.exists(vectorizer_path):
            raise FileNotFoundError(f"Vectorizer file not found at {vectorizer_path}")
            
        # Load files with explicit encoding
        with open(model_path, 'rb') as model_file:
            model = pickle.load(model_file)
        with open(vectorizer_path, 'rb') as vectorizer_file:
            vectorizer = pickle.load(vectorizer_file)
            
        return model, vectorizer
        
    except Exception as e:
        st.error(f"""
        ❌ Model loading failed: {str(e)}
        
        Required files:
        1. models/fake_news_model.pkl
        2. models/tfidf_vectorizer.pkl
        
        Please ensure:
        - Both files exist in the 'models' directory
        - Files were saved with the same Python version
        - Files weren't corrupted during upload
        """)
        st.stop()

model, tfidvect = load_model()

# Rest of your Streamlit app code...
