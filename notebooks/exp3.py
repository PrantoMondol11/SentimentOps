import os
import re
import string
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub

from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import warnings
warnings.simplefilter("ignore",UserWarning)
warnings.filterwarnings("ignore")

mlflow.set_tracking_uri("https://dagshub.com/mondolpranto83/SentimentOps.mlflow")
dagshub.init(repo_name="SentimentOps", repo_owner="mondolpranto83",mlflow=True)
mlflow.set_experiment("Hyperparameter Tuning")


# def normalize_text(df):
#     try:
#         df["review"]=df["review"].apply(lower_case)
#         df["review"]=df["review"].apply(remove_stop_words)
#         df["review"]=df["review"].apply(remove_numbers)
#         df["review"]=df["review"].apply(remove_punctuations)
#         df["review"]=df["review"].apply(lematization)
#         return df 
    
#     except Exception as e:
#         print(f"Error in text normalization: {e}")
#         raise
    

    
# VECTORIZER={
#     "tfidf":TfidfVectorizer(),
#     "bow":CountVectorizer()
# }




def preprocess_data(text):
    """Text preprocessing and vectorization"""
    lematizer=WordNetLemmatizer()
    stop_words=set(stopwords.words("english"))
    
    text=text.lower()
    text=re.sub(r'\d+', '',text)
    text=re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text=re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text=" ".join([lematizer.lemmatize(word) for word in text.split() if word not in stop_words])
    return text.strip()


def load_and_prepare_data(path):
    try:
        df=pd.read_csv(path)
        df["review"]=df["review"].astype(str).apply(preprocess_data)
        df=df[df["sentiment"].isin(["positive","negative"])]
        df["sentiment"]=df["sentiment"].map({"positive":1,"negative":0}).infer_objects(copy=False)
        
        vectorizer=TfidfVectorizer()
        x=vectorizer.fit_transform(df["review"])
        y=df["sentiment"].values

        return train_test_split(x,y,test_size=0.2,random_state=42), vectorizer
    except Exception as e:
        print(f"Error in loading data: {e}")
        raise
    
def train_and_log_model(x_train,x_test,y_train,y_test,vectorizer):
    """Train Logistic Regression model and log to MLflow"""
    param_grid={
        "C":[0.01,0.1,1,10],
        "penalty":["l1","l2"],
        "solver":["liblinear"]}
        
        
    try:
        with mlflow.start_run():
            grid_search=GridSearchCV(LogisticRegression(),param_grid,cv=5,scoring="f1",n_jobs=-1)
            grid_search.fit(x_train,y_train)
            
            for params,mean_score,std_score in zip(grid_search.cv_results_["params"],
                                                   grid_search.cv_results_["mean_test_score"],
                                                   grid_search.cv_results_["std_test_score"]):
                with mlflow.start_run(run_name=f"LR with params:{params}",nested=True):
                    model=LogisticRegression(**params)
                    model.fit(x_train,y_train)
                    y_pred=model.predict(x_test)
                    
                    metrics={
                        "accuracy":accuracy_score(y_test,y_pred),
                        "precision":precision_score(y_test,y_pred),
                        "recall":recall_score(y_test,y_pred),
                        "f1_score":f1_score(y_test,y_pred),
                        "mean_cv_score":mean_score,
                        "std_cv_score":std_score}
                    mlflow.log_params(params)
                    mlflow.log_metrics(metrics)
                    
                    print(f"Logged model with params: {params} and metrics: {metrics}")
            best_params=grid_search.best_params_
            mlflow.log_params({"best_params":best_params})
            best_model=grid_search.best_estimator_
            mlflow.sklearn.log_model(best_model,"best_logistic_regression_model")
            best_f1=grid_search.best_score_
            mlflow.log_metric("best_f1_score",best_f1)
            
            print(f"Best Logistic Regression model logged with params: {best_params} and best F1 score: {best_f1}")
    except Exception as e:
        print(f"Error in training and logging model: {e}")
        raise
    
if __name__=="__main__":
    data_path="notebooks/data.csv"
    (x_train,x_test,y_train,y_test),vectorizer=load_and_prepare_data(data_path)
    train_and_log_model(x_train,x_test,y_train,y_test,vectorizer)