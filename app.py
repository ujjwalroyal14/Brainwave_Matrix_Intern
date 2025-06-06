import nltk
nltk.download('stopwords')
nltk.download('wordnet')
import os
import joblib
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ---- App Configuration ----
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="wide"
)

# ---- Model Loading ----
@st.cache_resource
def load_models():
    try:
        # Get absolute paths
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, 'models', 'fake_news_model.pkl')
        vectorizer_path = os.path.join(base_dir, 'models', 'tfidf_vectorizer.pkl')
        
        # Verify files exist
        if not all(os.path.exists(p) for p in [model_path, vectorizer_path]):
            raise FileNotFoundError("Model files not found in 'models/' directory")
            
        return joblib.load(model_path), joblib.load(vectorizer_path)
    except Exception as e:
        st.error(f"""
        ❌ Model loading failed: {str(e)}
        
        Please ensure:
        1. The 'models/' directory exists
        2. It contains 'fake_news_model.pkl' and 'tfidf_vectorizer.pkl'
        3. Files are uploaded to Streamlit Cloud
        """)
        st.stop()

model, vectorizer = load_models()

# ---- Text Preprocessing ----
def preprocess_text(text):
    """Replicates training preprocessing exactly"""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    
    # Only remove stopwords if vectorizer was trained without them
    if not hasattr(vectorizer, 'stop_words_'):
        stop_words = set(stopwords.words('english'))
        words = [w for w in words if w not in stop_words]
    
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(w) for w in words]
    return ' '.join(words)

# ---- Streamlit UI ----
st.title("🔍 Fake News Detector")
text_input = st.text_area("Paste news article here:", height=200)

if st.button("Analyze"):
    if not text_input or len(text_input.split()) < 3:
        st.warning("Please enter at least 3 words for analysis")
    else:
        with st.spinner("Analyzing content..."):
            try:
                # Process and predict
                text_clean = preprocess_text(text_input)
                X = vectorizer.transform([text_clean])
                pred = model.predict(X)[0]
                proba = model.predict_proba(X)[0][1]
                
                # Display results
                if pred == 1:
                    st.error("## ⚠️ Prediction: Fake News")
                else:
                    st.success("## ✅ Prediction: Real News")
                
                st.progress(int(proba*100))
                st.metric("Confidence Score", f"{proba*100:.1f}%")
                
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
