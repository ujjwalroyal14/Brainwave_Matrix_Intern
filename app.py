import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

model = pickle.load(open('model/ffake_news_model.pkl', 'rb'))
tfidvect = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))


st.set_page_config(
    page_title="Fake News Detection",
    page_icon="assets/logo.jpg"
)


st.write("# Fake News Detection")
st.markdown(
    """
        A fake news prediction web application using Machine Learning algorithms deployed using streamlit community cloud.
""")
st.markdown("## Input:")

text = st.text_area(
    label="Enter your text to try it.",
    placeholder="Enter your text to predict whether this is fake or not.",
    height=200
)
st.write(f'You wrote {len(text.split())} words.')


# Load model and vectorizer to predict the output
def predict(text):
    val_tfidvect = tfidvect.transform([text]).toarray()
    prediction = 'FAKE' if model.predict(val_tfidvect) == 0 else 'REAL'
    return prediction


if st.button("Predict"):
    st.markdown("## Output:")
    if predict(text) == "REAL":
        st.markdown("#### Looking Real News📰")
    else:
        st.markdown("#### Looking Spam⚠️News📰")
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
