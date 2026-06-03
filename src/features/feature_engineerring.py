
import pandas as pd
import os
from sklearn.feature_extraction.text import  CountVectorizer

import yaml
from src.logger import logging
import pickle


def load_params(params_path:str) -> dict:
    """Load parameters from yaml file"""
    try:
        with open(params_path, "r") as file:
           params = yaml.safe_load(file)
        logging.debug(f"Parameters loaded successfully from {params_path}")
        return params
    except FileNotFoundError:
        logging.error(f"Parameters file not found at {params_path}")
        raise
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file at {params_path}: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error loading parameters from {params_path}: {e}")
        raise
    
def load_data(data_url:str) -> pd.DataFrame:
    """Load data from a given URL or local path"""
    try:
        df = pd.read_csv(data_url)
        df.fillna('', inplace=True)
        logging.debug(f"Data loaded successfully from {data_url}")
        return df
    except FileNotFoundError:
        logging.error(f"Data file not found at {data_url}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error loading data from {data_url}: {e}")
        raise
    
def apply_count_vectorizer(train_df:pd.DataFrame,test_df:pd.DataFrame,max_feature:int) -> tuple:
    """Apply CountVectorizer to the text data and return transformed train and test data"""
    try:
        logging.info("Applying CountVectorizer to the text data...")
        vectorizer=CountVectorizer(max_features=max_feature)
        x_train=train_df['review'].values
        x_test=test_df['review'].values
        y_train=train_df['sentiment'].values
        y_test=test_df['sentiment'].values
        
        X_train_bow=vectorizer.fit_transform(x_train)
        X_test_bow=vectorizer.transform(x_test)
        
        test_df_bow=pd.DataFrame(X_test_bow.toarray())
        train_df_bow=pd.DataFrame(X_train_bow.toarray())
        
        test_df_bow["label"]=y_test
        train_df_bow["label"]=y_train
        
        return train_df_bow,test_df_bow,vectorizer
    
    except KeyError as e:
        logging.error(f"Missing expected column in data: {e}")
        raise   
    except Exception as e:
        logging.error(f"Error in applying CountVectorizer: {e}")
        raise
def save_model(model,path:str) -> None:
    """Save the CountVectorizer object to a file using pickle"""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as file:
            pickle.dump(model, file)
        logging.info(f"model saved successfully at {path}")
    except Exception as e:
        logging.error(f"Error saving modelr: {e}")
        raise
    
def save_data(train_data:pd.DataFrame,test_data:pd.DataFrame,data_path:str) -> None:
    """Save preprocessed data to a specified path"""
    try:
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        train_data.to_csv(os.path.join(data_path,"train_bow.csv"),index=False)
        test_data.to_csv(os.path.join(data_path,"test_bow.csv"),index=False)
        logging.info(f"Preprocessed data saved successfully at {data_path}")
    except Exception as e:
        logging.error(f"Error saving preprocessed data: {e}")
        raise

def main():
    try:
        params=load_params("./params.yaml")
        
        train_data=load_data("./data/interim/train_preprocessed.csv")
        test_data=load_data("./data/interim/test_preprocessed.csv")
        
        max_feature=params["feature_engineering"]["max_features"]
        
        train_bow,test_bow,vectorizer=apply_count_vectorizer(train_data,test_data,max_feature)
        
        save_model(vectorizer,"./models/count_vectorizer.pkl")
        
        save_data(train_bow,test_bow,"./data/processed")
    except Exception as e:
        logging.error(f"Error in feature engineering process: {e}")
        
if __name__=='__main__':
    main()