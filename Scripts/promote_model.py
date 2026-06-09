import dagshub
import mlflow
import os
import logging

def promote_model():
    """Promote a model version to a specified stage in the MLflow Model Registry."""
    
    repo_owner="mondolpranto83"
    dagshub_token=os.getenv("DAGSHUB_TOKEN")
    if not dagshub_token:
        logging.warning("DAGsHub token not found in environment variables. Please set DAGSHUB_TOKEN to enable DAGsHub integration.")
    os.environ["MLFLOW_TRACKING_PASSWORD"]=dagshub_token
    os.environ["MLFLOW_TRACKING_USERNAME"]=repo_owner
    dagshub_url="https://dagshub.com"

    repo_name="SentimentOps"
    mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")

    
    client = mlflow.tracking.MlflowClient()
    
    get_latest_version = client.get_latest_versions(model_name, stages=[target_stage])[0].version
    
    prod_ver=client.get_latest_versions(model_name, stages=["Production"])
    
    for ver in prod_ver:
        client.transition_model_version_stage(
            name=model_name,
            version=ver.version,
            stage="Archived"
            )
        
    client.transition_model_version_stage(
        name=model_name,
        version=get_latest_version,
        stage="Production"
    )
    
    
if __name__ == "__main__":
    promote_model()

