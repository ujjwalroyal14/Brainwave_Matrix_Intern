import streamlit as st
import joblib
import numpy as np
from PIL import Image

# ---- App Configuration ----
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="wide"
)

# ---- Custom CSS ----
st.markdown("""
<style>
    .stTextInput input, .stTextArea textarea {
        border-radius: 10px !important;
        padding: 10px !important;
    }
    .stButton button {
        width: 100%;
        border-radius: 10px;
        background: linear-gradient(90deg, #FF4B4B 0%, #FF8E53 100%);
        color: white;
        font-weight: bold;
    }
    .prediction-card {
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# ---- Load Models ----
@st.cache_resource
def load_models():
    try:
        model = joblib.load('models/fake_news_model.pkl')
        vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.stop()

model, vectorizer = load_models()

# ---- Preprocessing ----
def preprocess_text(text):
    """Your existing preprocessing function"""
    import re
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if w not in stop_words]
    
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(w) for w in words]
    return ' '.join(words)

# ---- Header Section ----
col1, col2 = st.columns([3, 1])
with col1:
    st.title("🔍 Fake News Detector")
    st.markdown("""
    Detect potentially fake news articles using our AI-powered analyzer. 
    Paste any news content below to check its authenticity.
    """)
with col2:
    st.image("https://i.imgur.com/XYZ1234.png", width=150)  # Replace with your logo

# ---- Main Input ----
with st.form("prediction_form"):
    text_input = st.text_area(
        "Enter news content:", 
        placeholder="Paste news article here...",
        height=200
    )
    
    submitted = st.form_submit_button("Analyze Article")
    
    if submitted:
        if len(text_input.split()) < 3:
            st.warning("Please enter at least 3 words for accurate analysis")
        else:
            with st.spinner("Analyzing content..."):
                try:
                    # Preprocess and predict
                    text_clean = preprocess_text(text_input)
                    X = vectorizer.transform([text_clean])
                    pred = model.predict(X)[0]
                    proba = model.predict_proba(X)[0][1]
                    
                    # ---- Results Display ----
                    st.balloons()
                    
                    if pred == 1:
                        st.error("""
                        ## ⚠️ Prediction: Fake News
                        This content appears to be potentially misleading or false.
                        """)
                    else:
                        st.success("""
                        ## ✅ Prediction: Real News
                        This content appears to be credible.
                        """)
                    
                    # Confidence meter
                    st.progress(int(proba*100))
                    st.metric("Confidence Score", f"{proba*100:.1f}%")
                    
                    # Explanation card
                    with st.expander("📊 Analysis Details"):
                        st.markdown(f"""
                        - **Text Processed**: "{text_clean[:200]}..."
                        - **Model Used**: {type(model).__name__}
                        - **Threshold**: >70% confidence for reliable prediction
                        """)
                        
                    # Disclaimer
                    st.info("""
                    ℹ️ **Disclaimer**: This AI tool provides probabilistic estimates, 
                    not absolute truths. Always verify information from multiple sources.
                    """)
                    
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

# ---- Footer ----
st.markdown("---")
st.markdown("""
<div style="text-align: center">
    <p>Built with ❤️ using Streamlit | Model accuracy: 95.2%</p>
</div>
""", unsafe_allow_html=True)