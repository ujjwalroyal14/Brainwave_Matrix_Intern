import os
import joblib
import re
import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.exceptions import NotFittedError

# Download NLTK datasets (only once)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Load model and vectorizer with error handling
@st.cache_resource
def load_models():
    try:
        model = joblib.load('fake_news_model.pkl')  # Replace with your path
        vectorizer = joblib.load('tfidf_vectorizer.pkl')  # Replace with your path
        return model, vectorizer
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

model, tfidf_vectorizer = load_models()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Improved text preprocessing
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove special characters and numbers but keep basic punctuation
    text = re.sub(r'[^a-zA-Z\s.,!?]', '', text)
    
    # Tokenize and lemmatize
    words = nltk.word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    # Remove short words (length < 2)
    words = [word for word in words if len(word) > 1]
    
    return ' '.join(words)

# Enhanced prediction function
def predict(text):
    try:
        processed_text = preprocess_text(text)
        features = tfidf_vectorizer.transform([processed_text])
        prediction = model.predict(features)[0]
        
        # Get prediction probability for confidence score
        proba = model.predict_proba(features)[0]
        confidence = max(proba) * 100
        
        return prediction, confidence
        
    except NotFittedError:
        st.error("Model is not fitted properly. Please check the model files.")
        return None, None
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None

# Streamlit UI with enhancements
def main():
    st.set_page_config(page_title="Fake News Detector", page_icon="📰")
    
    st.title("📰 Fake News Detector")
    st.markdown("""
    This tool helps identify potentially fake news articles. 
    Paste the text of a news article below to analyze its authenticity.
    """)
    
    with st.expander("ℹ️ About this tool"):
        st.write("""
        - Uses machine learning to classify news articles as REAL or FAKE
        - Works best with complete articles (at least 100 words)
        - Model trained on a dataset of labeled real and fake news articles
        - Text preprocessing includes lemmatization and stopword removal
        """)
    
    user_input = st.text_area("Enter news article text:", height=200,
                             placeholder="Paste the news article text here...")
    
    if st.button("Analyze Article", type="primary"):
        if not user_input.strip():
            st.warning("Please enter some text to analyze!")
        elif len(user_input.split()) < 10:  # More reasonable minimum
            st.warning("For better results, please enter at least 10 words.")
        else:
            with st.spinner("Analyzing the article..."):
                prediction, confidence = predict(user_input)
                
                if prediction is not None and confidence is not None:
                    col1, col2 = st.columns(2)
                    
                    if prediction == 0:  # FAKE
                        with col1:
                            st.error("### ⚠️ Likely FAKE News")
                            st.metric("Confidence", f"{confidence:.1f}%")
                        with col2:
                            st.warning("This content appears to be unreliable. Verify with trusted sources.")
                    else:  # REAL
                        with col1:
                            st.success("### ✅ Likely REAL News")
                            st.metric("Confidence", f"{confidence:.1f}%")
                        with col2:
                            st.info("This content appears credible, but always verify important information.")
                    
                    # Show some processed text for transparency
                    with st.expander("Show processed text"):
                        processed = preprocess_text(user_input)
                        st.write(processed[:500] + ("..." if len(processed) > 500 else ""))

if __name__ == "__main__":
    main()
