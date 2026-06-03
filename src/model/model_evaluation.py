import mlflow
import dagshub
import json
import mlflow.sklearn
import pickle
from src.logger import logging
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score as auc_score
import os
from src.features.feature_engineerring import load_data

mlflow.set_tracking_uri("https://dagshub.com/mondolpranto83/SentimentOps.mlflow")
dagshub.init(repo_owner="mondolpranto83", repo_name="SentimentOps",mlflow=True)


def load_model(model_path:str):
    """Load a trained model from a specified path using pickle"""
    try:
        with open(model_path, "rb") as file:
            model = pickle.load(file)
        logging.info(f"Model loaded successfully from {model_path}")
        return model
    except FileNotFoundError:
        logging.error(f"Model file not found at {model_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading model from {model_path}: {e}")
        raise

def save_model_info(model_uri: str, model_id: str, file_path: str) -> None:
    """Save model information for model registration"""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        model_info = {
            "model_uri": model_uri,
            "model_id": model_id
        }

        with open(file_path, "w") as file:
            json.dump(model_info, file, indent=4)

        logging.info(f"Model information saved successfully at {file_path}")

    except Exception as e:
        logging.error(f"Error saving model information: {e}")
        raise
    
    
def evaluate_model(model, X_test, y_test):
    """Evaluate the model on the test data and return the accuracy"""
    try:
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]  # Get probabilities for the positive class
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = auc_score(y_test, y_pred_proba)
        
        
        metrices={
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "auc_score": auc
        }
        logging.info(f"Model evaluation completed successfully. Accuracy: {accuracy}")
        return metrices
    except Exception as e:
        logging.error(f"Error evaluating the model: {e}")
        raise
    
def save_evaluation_results(metrices:dict, results_path:str) -> None:
    """Save the evaluation results to a specified path using pickle"""
    try:
        os.makedirs(os.path.dirname(results_path),exist_ok=True)
        with open(results_path, "w") as file:
            json.dump(metrices, file,indent=4)
        logging.info(f"Evaluation results saved successfully at {results_path}")
    except Exception as e:
        logging.error(f"Error saving evaluation results to {results_path}: {e}")
        raise
    
def main():
    mlflow.set_experiment("My_dvc_experiment")
    with mlflow.start_run() as run:
        try:
            model=load_model("./models/logistic_regression_model.pkl")
            test_data=load_data("./data/processed/test_bow.csv")
        
            X_test=test_data.drop("label",axis=1)
            y_test=test_data["label"]
        
            metrices=evaluate_model(model,X_test,y_test)
            
            for metric_name,metric_value in metrices.items():
                mlflow.log_metric(metric_name,metric_value)
                
            if hasattr(model,"get_params"):
                params=model.get_params()
                
                for param_name,param_value in params.items():
                    mlflow.log_param(param_name,param_value)
                    
            logged_model= mlflow.sklearn.log_model(sk_model=model,
                                     artifact_path="model")
            # log the entire evaluation report as an artifact
            
            
            save_model_info(logged_model.model_uri, logged_model.model_id, "./reports/experiment_info.json")
        
            save_evaluation_results(metrices,"./results/evaluation_results.json")
            mlflow.log_artifact("./results/evaluation_results.json")
            
            print("Model URI:", logged_model.model_uri)
            print("Model ID:", logged_model.model_id)
            print("Run ID:", run.info.run_id)
        except Exception as e:
            logging.error(f"Failed to complete model evaluation: {e}")
            print(f"error:{e}")
        
if __name__=='__main__':
    main()