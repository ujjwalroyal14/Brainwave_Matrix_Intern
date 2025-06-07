import os
import joblib
import re
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import nltk

# Streamlit UI
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="🔍"
)

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


model, tfidvect = load_models()

# Initialize stemmer and lemmatizer
ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Text preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if w not in stop_words]
    
    # Apply both stemming and lemmatization
    words = [ps.stem(lemmatizer.lemmatize(w)) for w in words]
    return ' '.join(words)


st.write("# Fake News Detection")
st.markdown(
    """
    A fake news prediction web application using Machine Learning algorithms deployed using Streamlit.
    """
)

text = st.text_area(
    label="Enter your text to try it:",
    placeholder="Enter your text to predict whether this is fake or not.",
    height=200
)
st.write(f'You wrote {len(text.split())} words.')

def predict(text):
    try:
        processed_text = preprocess_text(text)
        val_tfidvect = tfidvect.transform([processed_text]).toarray()
        prediction = model.predict(val_tfidvect)[0]
        
        # Determine and return the result
        result = 'FAKE' if prediction == 0 else 'REAL'
        
        # Display the appropriate message
        if result == "FAKE":
            st.error("#### This is likely FAKE NEWS! ⚠️")
        else:
            st.success("#### This appears to be REAL NEWS. ✅")
            
        return result
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

if st.button("Predict"):
    st.markdown("## Output:")
    if len(text.split()) < 3:
        st.warning("Please enter at least 3 words for accurate prediction")
    else:
        with st.spinner("Analyzing..."):
            result = predict(text)
            if result == "REAL":
                st.success("#### Looking Fake News 📰")
            else:
                st.error("#### Looking Real News ⚠️")

# ---- Rest of your Streamlit app ----
# (Include your text preprocessing and UI code here)
