import json
import logging
import os
import mlflow
import pickle

import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore",UserWarning) 
import dagshub



repo_owner="mondolpranto83"
dagshub_token=os.getenv("DAGSHUB_TOKEN")
if not dagshub_token:
    logging.warning("DAGsHub token not found in environment variables. Please set DAGSHUB_TOKEN to enable DAGsHub integration.")
os.environ["MLFLOW_TRACKING_PASSWORD"]=dagshub_token
os.environ["MLFLOW_TRACKING_USERNAME"]=repo_owner
dagshub_url="https://dagshub.com"
print("Token exists:", dagshub_token is not None)
print("DAGsHub URL:", dagshub_url)
repo_name="SentimentOps"
mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")

def load_model_info(file_path:str) -> dict:
    """Load model information and evaluation metrices from a specified path"""
    try:
        with open(file_path, "r") as file:
            model_info = json.load(file)
        logging.info(f"Model information loaded successfully from {file_path}")
        return model_info
    except FileNotFoundError:
        logging.error(f"Model information file not found at {file_path}")
        raise
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing JSON file at {file_path}: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error loading model information from {file_path}: {e}")
        raise


def register_model(model_name:str, model_info:dict)-> None:
    try:
        model_uri =  model_info["model_uri"]


        client = mlflow.tracking.MlflowClient()

        print("Model URI:", model_uri)

        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=model_name,
            
        )
        client.set_registered_model_alias(
        name=model_name,
        alias="candidate",
        version=model_version
        )

        print("Registered version:", model_version.version)

    except Exception as e:
        print("FULL ERROR:", repr(e))
        raise
    
def main():
    try:
        model_info=load_model_info("./reports/experiment_info.json")
        register_model(model_name="Sentiment_Analysis_Model",model_info=model_info)
    except Exception as e:
        logging.error(f"Error in main function: {e}")
        raise
    
if __name__=="__main__":
    main()