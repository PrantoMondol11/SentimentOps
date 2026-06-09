import os
import mlflow


def promote_model(model_name):

    repo_owner = "mondolpranto83"
    dagshub_token = os.getenv("DAGSHUB_TOKEN")

    os.environ["MLFLOW_TRACKING_USERNAME"] = repo_owner
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    mlflow.set_tracking_uri(
        f"https://dagshub.com/{repo_owner}/SentimentOps.mlflow"
    )

    client = mlflow.tracking.MlflowClient()

    # Get candidate model
    candidate = client.get_model_version_by_alias(
        model_name,
        "candidate"
    )

    # Promote to champion
    client.set_registered_model_alias(
        name=model_name,
        alias="champion",
        version=candidate.version
    )

    print(
        f"Version {candidate.version} promoted to champion"
    )


if __name__ == "__main__":
    promote_model("Sentiment_Analysis_Model")