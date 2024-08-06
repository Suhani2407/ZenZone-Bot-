import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import joblib

# Download NLTK resources if not already downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Load ISEAR dataset (replace 'url' with the actual dataset URL)
url = 'isear_csv.csv'

# Try reading the CSV file with different encodings
try:
    df = pd.read_csv(url, names=['EMOT', 'SIT'], encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv(url, sep=',', encoding='ISO-8859-1')
    df = df[['EMOT','SIT']]

# Preprocess text data
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token.isalpha() and token not in stop_words and len(token) > 1]
    return ' '.join(filtered_tokens)

df['processed_text'] = df['SIT'].apply(preprocess_text)

# Filter out rows with empty 'processed_text' after preprocessing
df = df[df['processed_text'].apply(lambda x: len(x.strip()) > 0)]

# Check if we have enough samples to perform train-test split
if len(df) < 2:
    print("Warning: Not enough samples to perform train-test split.")
    # Handle this case appropriately (e.g., skip model training)
else:
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(df['processed_text'], df['EMOT'], test_size=0.2, random_state=42)

    # Check if we have enough samples for training
    if len(X_train) < 2:
        print("Warning: Insufficient samples for training.")
        # Handle this case appropriately (e.g., skip model training)
    else:
        # Feature extraction using TF-IDF with n-grams
        tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, min_df=5)
        X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
        X_test_tfidf = tfidf_vectorizer.transform(X_test)

        # Initialize and train Random Forest classifier
        rf_classifier = RandomForestClassifier(n_estimators=200, random_state=42)
        rf_classifier.fit(X_train_tfidf, y_train)

        # Evaluate the model
        y_pred = rf_classifier.predict(X_test_tfidf)

        # Calculate accuracy and other metrics
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")

        print("\nClassification Report:")
        print(classification_report(y_test,y_pred))
# Save the trained model to a file
model_filename = 'Sent_Analy_classifier_model.joblib'
joblib.dump(rf_classifier, model_filename)

#save tfidf vectorizer

joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer_1.joblib')