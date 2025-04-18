import mlflow
from mlflow.tracking import MlflowClient
mlflow.set_tracking_uri("http://44.204.159.230:5000/")

#load model from the model registry
def load_model_from_registry(model_name,mode_version):
   model_uri = f"models:/{model_name}/{mode_version}"
   model =mlflow.pyfunc.load_model(model_uri)
   return model

# example usage
model=load_model_from_registry("my_model","1")
print("model Loaded suceessfully!")

import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources (only once)
nltk.download('stopwords')
nltk.download('wordnet')

# Preprocessing function
def preprocess_comment(comment):
    comment = comment.lower().strip()
    comment = re.sub(r'\n', ' ', comment)
    comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)
    stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
    comment = ' '.join([word for word in comment.split() if word not in stop_words])
    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

# Load the TF-IDF vectorizer
vectorizer = joblib.load("tfidf_vectorizer.pkl")  # Make sure this file exists in the same dir

# Input comment
raw_comment = "good job"
preprocessed = preprocess_comment(raw_comment)

# Vectorize and predict
X = vectorizer.transform([preprocessed])
prediction = model.predict(X)

# Output
print(f"Comment: {raw_comment}")
print(f"Predicted Sentiment: {prediction[0]}")
