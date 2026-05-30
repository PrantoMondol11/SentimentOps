import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
import yaml
from src.logger import logging  
from src.features.feature_engineerring import load_params,load_data,save_data,save_model

def train_model(X_train:pd.DataFrame,y_train:pd.Series) -> LogisticRegression:
    """Train a Logistic Regression model on the given training data"""
    try:
        model=LogisticRegression(C=1.0,solver='liblinear',penalty='l2')
        model.fit(X_train,y_train)
        logging.info("Model trained successfully.")
        return model
    except Exception as e:
        logging.error(f"Error training the model: {e}")
        raise
    
    
def main():
    try:
        # params=load_params("./params.yaml")
        # params=load_params("./params.yaml")
        train_data=load_data("./data/processed/train_bow.csv")
        
        X_train=train_data.drop("label",axis=1)
        y_train=train_data["label"]
        
        trained_model=train_model(X_train,y_train)
        
        
        save_model(trained_model,"./models/logistic_regression_model.pkl")
        logging.info("Model saved")
        
    except Exception as e:
        logging.error("Failed to complete model building: %s",e)
        print(f"error:{e}")
        
        
if __name__=="__main__":
    main()