import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the trained model
def predict_sentiment(user_prompt):
    model_filename = 'Sent_Analy_classifier_model.joblib'
    loaded_model = joblib.load(model_filename)

# Load the pre-trained TF-IDF vectorizer
    vectorizer_filename = 'tfidf_vectorizer_1.joblib'
    tfidf_vectorizer = joblib.load(vectorizer_filename)

    def preprocess_text(text):
        tokens = word_tokenize(text.lower())
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [token for token in tokens if token.isalpha() and token not in stop_words and len(token) > 1]
        return ' '.join(filtered_tokens)

# Preprocess the prompt text (replace 'prompt_text' with the actual prompt)

    preprocessed_prompt = preprocess_text(user_prompt)

# Transform the preprocessed prompt text into TF-IDF features using the pre-trained vectorizer
    prompt_tfidf = tfidf_vectorizer.transform([preprocessed_prompt])

# Predict the sentiment label of the prompt
    predicted_sentiment = loaded_model.predict(prompt_tfidf)[0]

    sentiment_names = {1: 'joy', 2: 'fear', 3: 'anger',4: 'sad',5: 'distress',6: 'shame'}  
    predicted_sentiment_name = sentiment_names.get(predicted_sentiment)
    return (predicted_sentiment_name)

