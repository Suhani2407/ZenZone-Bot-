
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import joblib

# Load the labeled dataset
data = pd.read_csv('updated_dataset.csv')  # replace 'labeled_data.csv' with your dataset filename

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['user_prompt'], data['intent'], test_size=0.2, random_state=42)

# Initialize the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=1000)  # adjust max_features as needed

# Transform text data into TF-IDF features
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Initialize and train a logistic regression classifier
intent_classifier = SVC(kernel='linear')
intent_classifier.fit(X_train_tfidf, y_train)

# Make predictions on the testing data
y_pred = intent_classifier.predict(X_test_tfidf)

# Evaluate the model
print(classification_report(y_test, y_pred))

#save the model
joblib.dump(intent_classifier, 'intent_classifier_model_1.joblib')

#save vectoriser
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.joblib')