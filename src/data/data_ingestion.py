import numpy as np
import pandas as pd

pd.set_option("future.no_silent_downcasting", True)

import os
from sklearn.model_selection import train_test_split
import yaml
import logging
from src.logger import logging
# from src.connections import s3_connection

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
        if data_url.startswith("s3://"):
            s3 = s3_connection()
            bucket_name, key = data_url.replace("s3://", "").split("/", 1)
            obj = s3.get_object(Bucket=bucket_name, Key=key)
            df = pd.read_csv(obj['Body'])
        else:
            df = pd.read_csv(data_url)
        logging.debug(f"Data loaded successfully from {data_url}")
        return df
    except FileNotFoundError:
        logging.error(f"Data file not found at {data_url}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error loading data from {data_url}: {e}")
        raise
    
def preprocess_data(df:pd.DataFrame) -> pd.DataFrame:
    """Basic text preprocessing: lowercasing and removing punctuation"""
    try:
        logging.info("Preprocessing data .....")
        final_df=df[df["sentiment"].isin(["positive","negative"])]
        final_df["sentiment"]=final_df["sentiment"].map({"positive":1,"negative":0})
        logging.info("Data preprocessing completed successfully")
        return final_df
    except KeyError as e:
        logging.error(f"Missing expected column in data: {e}")
        raise 
    except Exception as e:
        logging.error(f"Error in preprocessing data: {e}")
        raise
    
def save_data(train_data:pd.DataFrame,test_data:pd.DataFrame,data_path:str) -> None:
    """Save preprocessed data to a specified path"""
    try:
        raw_data_path=os.path.join(data_path,"raw")
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path,"train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path,"test.csv"), index=False)
        logging.info(f"Preprocessed data saved successfully to {raw_data_path}")
    except Exception as e:
        logging.error(f"Error saving preprocessed data to {raw_data_path}: {e}")
        raise
    
def main():
    try:
        params=load_params(params_path="params.yaml")
        test_size=params["data_ingestion"]["test_size"]
        df=load_data(data_url="https://raw.githubusercontent.com/vikashishere/Datasets/refs/heads/main/data.csv")
        final_df=preprocess_data(df)
        train_data, test_data = train_test_split(final_df, test_size=0.2, random_state=42)
        save_data(train_data, test_data, data_path='./data')
    except Exception as e:
        logging.error(f"Data ingestion failed: {e}")
        raise


if __name__=="__main__":
    main()