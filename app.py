from flask import Flask, request, jsonify, render_template
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Preprocessing setup
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
    return text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify_text():
    data = request.get_json()
    text = data['text']
    processed_text = preprocess_text(text)

    # Load vectorizer and model
    vectorizer = joblib.load("vectorizer.pkl")
    model = joblib.load("text_classification_model.pkl")

    # Vectorize input
    vectorized_text = vectorizer.transform([processed_text])

    # Predict
    prediction = model.predict(vectorized_text)[0]

    # Map numerical label to sentiment string
    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    sentiment = label_map.get(prediction, "Unknown")

    return jsonify({"prediction": sentiment})

if __name__ == '__main__':
    app.run(debug=True)
