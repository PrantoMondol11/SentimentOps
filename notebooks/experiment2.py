import setuptools
import os
import re
import string
import pandas as pd

pd.set_option('future.no_silent_downcasting', True)

import numpy as np
import mlflow 
import mlflow.sklearn
import dagshub
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import scipy.sparse


import warnings
warnings.simplefilter("ignore",UserWarning)
warnings.filterwarnings("ignore")

CONFIG={
    "data_path":"notebooks/data.csv",
    "test_size":0.2,
    "mlflow_tracking_uri":"https://dagshub.com/mondolpranto83/SentimentOps.mlflow",
    "datashub_repo":"SentimentOps",
    "dagshub_owner":"mondolpranto83",
    "experiment_name":"Bow vs Tfidf",
}
mlflow.set_tracking_uri(CONFIG["mlflow_tracking_uri"])
dagshub.init(repo_name=CONFIG["datashub_repo"], repo_owner=CONFIG["dagshub_owner"],mlflow=True)
mlflow.set_experiment(CONFIG["experiment_name"])


def lemmatization(text):
    lemmatizer=WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

def remove_stopwords(text):
    stop_words=set(stopwords.words('english'))
    return ' '.join([word for word in text.split() if word not in stop_words])

def remove_numbers(text):
    return ''.join([i for i in text if not i.isdigit()])

def remove_punctuation(text):
    return re.sub(f"[{re.escape(string.punctuation)}]", " ", text)

def lower_case(text):
    return text.lower()

def preprocess_text(df):
    try:
        df['review']=df['review'].apply(lower_case)
        df['review']=df['review'].apply(remove_punctuation)
        df['review']=df['review'].apply(remove_numbers)
        df['review']=df['review'].apply(remove_stopwords)
        df['review']=df['review'].apply(lemmatization)
        return df
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return df
    
    
def load_data(path):
    try:
        df=pd.read_csv(path)
        df=preprocess_text(df)
        df=df[df["sentiment"].isin(["positive","negative"])]
        df["sentiment"]=df["sentiment"].replace({"positive":1,"negative":0}).infer_objects(copy=False)
        return df
    except Exception as e:
        print(f"Error in loading data: {e}")
        raise 
         
VECTORIZERS={
    "tfidf":TfidfVectorizer(max_features=5000),
    "bow":CountVectorizer(max_features=5000)
}

ALGORITHMS={
     "logistic_regression":LogisticRegression(),
     "MULTINOMIAL_NB":MultinomialNB(),
     "xgboost":XGBClassifier(),
     "random_forest":RandomForestClassifier(),
        "gradient_boosting":GradientBoostingClassifier() 
}


def train_and_evaluate(df):
    with mlflow.start_run(run_name="All Experments") as parent_run:
        for algo_name,algorithms in ALGORITHMS.items():
            for vec_name,vectorizer in VECTORIZERS.items():
                with mlflow.start_run(run_name=f"{algo_name} with {vec_name}",nested=True) as child_run:
                    try:
                        x=vectorizer.fit_transform(df['review'])
                        y=df['sentiment']
                        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=CONFIG['test_size'],random_state=42)
                        
                        mlflow.log_params({
                            "algorithm":algo_name,
                            "vectorizer":vec_name,
                            "test_size":CONFIG['test_size'] 
                        }) 
                        
                        model=algorithms
                        model.fit(x_train,y_train)
                        log_model_params(algo_name,model)
                        
                        y_pred=model.predict(x_test)
                        metrics = {
                                    "accuracy": accuracy_score(y_test, y_pred),
                                    "precision": precision_score(y_test, y_pred, zero_division=0),
                                    "recall": recall_score(y_test, y_pred, zero_division=0),
                                    "f1_score": f1_score(y_test, y_pred, zero_division=0)
                                 }

                        
                        mlflow.log_metrics(metrics) 
                        
                        input_example=x_test[0:5] if not scipy.sparse.issparse(x_test) else x_test[0:5].toarray()
                        mlflow.sklearn.log_model(model,"model",input_example = input_example.astype("float32"))
                        
                        print(f"\nAlgorithm: {algo_name}, Vectorizer: {vec_name}")
                        print(f"Metrics: {metrics}\n")
 
                    except Exception as e:
                        print(f"Error in training and evaluation for {algo_name} with {vec_name}: {e}")
                        mlflow.log_param("error",str(e))


def log_model_params(algo_name,model):
    params_to_log={}
    if algo_name=="logistic_regression":
        params_to_log["c"]=model.C
    elif algo_name=="MULTINOMIAL_NB":
        params_to_log['alpha']=model.alpha
    elif algo_name=="xgboost":
        params_to_log['n_estimators']=model.n_estimators
    elif algo_name=="random_forest":
        params_to_log["n_estimators"]=model.n_estimators
    elif algo_name=="gradient_boosting":
        params_to_log["n_estimators"]=model.n_estimators
        params_to_log["learning_rate"]=model.learning_rate
        params_to_log["max_depth"]=model.max_depth
        
    mlflow.log_params(params_to_log)
    
if __name__=="__main__":
    df=load_data(CONFIG["data_path"])
    train_and_evaluate(df)
    
    