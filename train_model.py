import pandas as pd
import re
import string
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('stopwords')
nltk.download('wordnet')

# Load dataset
df = pd.read_csv('sentiment_analysis.csv')  # or your actual CSV file name

# Only keep rows with sentiment values we care about
df = df[['text', 'sentiment']].dropna()

# Update the sentiment mapping to handle 3 classes: positive (2), neutral (1), and negative (0)
def map_sentiment(label):
    label = label.strip().lower()
    if label == 'positive':
        return 2
    elif label == 'neutral':
        return 1
    elif label == 'negative':
        return 0
    else:
        return None
df['sentiment'] = df['sentiment'].apply(map_sentiment)

# Preprocessing setup
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
    return text

# Clean text
df['cleaned'] = df['text'].apply(preprocess_text)
df = df[df['cleaned'].str.strip() != '']  # Remove any empty rows

# Features and labels
X = df['cleaned']
y = df['sentiment']

# Vectorization
vectorizer = TfidfVectorizer()
X_vect = vectorizer.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.2, random_state=42)

# Model training
model = LogisticRegression(multi_class='multinomial', solver='lbfgs')  # Multinomial logistic regression
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.2f}")

# Save model and vectorizer
joblib.dump(vectorizer, 'vectorizer.pkl')
joblib.dump(model, 'text_classification_model.pkl')
print("Model and vectorizer saved.")
