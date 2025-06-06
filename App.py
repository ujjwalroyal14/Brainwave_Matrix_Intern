from flask import Flask, request, jsonify
import joblib
import numpy as np
from functools import wraps
import logging
from werkzeug.middleware.proxy_fix import ProxyFix

# Initialize Flask app
app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model and vectorizer
try:
    model = joblib.load('models/fake_news_model.pkl')
    tfidf = joblib.load('models/tfidf_vectorizer.pkl')
    logger.info("Model and vectorizer loaded successfully")
except Exception as e:
    logger.error(f"Error loading model files: {str(e)}")
    raise

# Text cleaning function (must match training preprocessing)
def clean_text(text):
    """Clean and preprocess text input"""
    import re
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # Remove special chars
    words = text.split()
    
    # Remove stopwords if used during training
    if hasattr(tfidf, 'stop_words_'):
        stop_words = set(stopwords.words('english'))
        words = [w for w in words if w not in stop_words]
    
    # Lemmatization if used during training
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(w) for w in words]
    
    return ' '.join(words)

# API key authentication (optional)
API_KEYS = {"your_api_key_here": "client1"}

def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        api_key = request.headers.get('X-API-KEY')
        if api_key not in API_KEYS:
            logger.warning(f"Unauthorized access attempt from {request.remote_addr}")
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated

@app.route('/predict', methods=['POST'])
@require_api_key  # Remove if not using authentication
def predict():
    """Main prediction endpoint"""
    try:
        # Validate input
        if not request.is_json:
            logger.error("Invalid content type")
            return jsonify({"error": "Content-Type must be application/json"}), 400
            
        data = request.get_json()
        if 'text' not in data or not isinstance(data['text'], str):
            logger.error("Invalid input format")
            return jsonify({"error": "Missing or invalid 'text' field"}), 400
        
        # Clean and vectorize text
        text = clean_text(data['text'])
        if len(text.split()) < 3:  # Minimum word check
            return jsonify({"error": "Text too short"}), 400
            
        vec = tfidf.transform([text])
        
        # Make prediction
        prediction = model.predict(vec)[0]
        probability = model.predict_proba(vec)[0][1]  # Probability of being fake
        
        logger.info(f"Prediction successful for text: {text[:50]}...")
        
        return jsonify({
            "prediction": int(prediction),
            "class": "fake" if prediction == 1 else "real",
            "confidence": float(probability),
            "text_processed": text[:200]  # For debugging
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "vectorizer_loaded": tfidf is not None
    })

if __name__ == '__main__':
    # Production config - use gunicorn in production
    app.run(host='0.0.0.0', port=5000, debug=False)