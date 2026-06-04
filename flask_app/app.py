from flask import Flask, render_template, request

import os
import pickle
import pandas as pd
import mlflow
import numpy as np

from prometheus_client import CollectorRegistry, Counter, generate_latest, CONTENT_TYPE_LATEST,Histogram
import time
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import re
import dagshub
from src.logger import logging

import warnings

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)


def preprocess_text(text):
    lemmatizer=WordNetLemmatizer()
    stop_words=set(stopwords.words('english'))
    
    text = re.sub(r'http?://\s+|www\.\S+', '', text)
    text=' '.join([word for word in text.split() if word not in stop_words])
    text=''.join([char for char in text if not char.isdigit()])
    text=text.lower()
    text=re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text=text.replace(':',"")
    text=re.sub(r'\s+', ' ', text).strip()
    text=' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return text

def remove_small_sentences(df):
    """Remove sentences that are shorter than the specified minimum length."""
    
    for i in range(len(df)):
        if len(df.loc[i,'review'].split())<3:
            df.text.iloc[i]=np.nan
            

repo_owner="mondolpranto83"
dagshub_token=os.getenv("MLFLOW_TOKEN")
if not dagshub_token:
    logging.warning("DAGsHub token not found in environment variables. Please set MLFLOW_TOKEN to enable DAGsHub integration.")
os.environ["MLFLOW_TRACKING_PASSWORD"]=dagshub_token
os.environ["MLFLOW_TRACKING_USERNAME"]=repo_owner
dagshub_url="https://dagshub.com"

repo_name="SentimentOps"
mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")

dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
def load_model_version(model_name:str):
    """Load the trained model from the specified path."""
    try:
        client = mlflow.MlflowClient()
        model_version = client.get_latest_versions(model_name,stages=["Staging"])
        if not model_version:
            logging.error(f"No model found in Staging for {model_name}")
            model_version = client.get_latest_versions(model_name,stages=["None"])
        return model_version[0].version if model_version else None
    except Exception as e:
        logging.error(f"Error loading model from {model_name}: {e}")
        raise
    
app = Flask(__name__)

registry=CollectorRegistry()
Request_count=Counter("request_count","Number of requests received",registry=registry,labelnames=["method","endpoint"])
Request_latency=Histogram("request_latency_seconds","Latency of requests in seconds",registry=registry,labelnames=["method","endpoint"])
Prediction_count=Counter("prediction_count","Number of predictions made",registry=registry,labelnames=["result"])


model_name="Sentiment_Analysis_Model"
model=load_model_version(model_name)
model_uri=f"models:/{model_name}/{model}"

print(f"Loading model from {model_uri}...")
loaded_model=mlflow.pyfunc.load_model(model_uri)
print("Model loaded successfully.")
vectorizer_path=os.path.join("models","count_vectorizer.pkl")
with open(vectorizer_path,"rb") as file:
    loaded_vectorizer=pickle.load(file)
print("Vectorizer loaded successfully.")

@app.route("/")
def home():
    Request_count.labels(method="GET",endpoint="/").inc()
    start_time=time.time()
    response=render_template("index.html",result=None)
    Request_latency.labels(method="GET",endpoint="/").observe(time.time()-start_time)
    return response

@app.route("/predict",methods=["POST"])
def predict():
    Request_count.labels(method="POST",endpoint="/predict").inc()
    start_time=time.time()
    try:
        review=request.form["review"]
        
        print(f"Received review for prediction: {review}")
        preprocessed_review=preprocess_text(review)
        review_vector=loaded_vectorizer.transform([preprocessed_review])
        prediction=loaded_model.predict(review_vector)
        print(f"Prediction result: {prediction}")
        result="Positive" if prediction[0]==1 else "Negative"
        Prediction_count.labels(result=result).inc()
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        result="Error processing the review. Please try again."
    Request_latency.labels(method="POST",endpoint="/predict").observe(time.time()-start_time)
    return render_template("index.html",result=result)

@app.route("/metrics")
def metrics():
    return generate_latest(registry), 200, {"Content-Type": CONTENT_TYPE_LATEST}


if __name__=="__main__":
    app.run(debug=True,host="0.0.0.0",port=5000)