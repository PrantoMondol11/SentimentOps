import unittest
import pandas as pd
import os
import pickle
import mlflow

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


class TestModelRegistration(unittest.TestCase):

    @staticmethod
    def get_latest_versions(model_name, stages):
        client = mlflow.tracking.MlflowClient()
        latest_version = client.get_latest_versions(
            model_name,
            stages=stages
        )
        return latest_version

    @classmethod
    def setUpClass(cls):

        repo_owner = "mondolpranto83"

        dagshub_token = os.getenv("DAGSHUB_TOKEN")

        if not dagshub_token:
            raise ValueError("DAGSHUB_TOKEN not found.")

        os.environ["MLFLOW_TRACKING_USERNAME"] = repo_owner
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        dagshub_url = "https://dagshub.com"
        repo_name = "SentimentOps"

        mlflow.set_tracking_uri(
            f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow"
        )

        cls.model_name = "Sentiment_Analysis_Model"

        cls.version = cls.get_latest_versions(
            cls.model_name,
            ["Staging"]
        )[0].version

        cls.model_uri = f"models:/{cls.model_name}/{cls.version}"

        cls.model = mlflow.pyfunc.load_model(
            cls.model_uri
        )

        with open(
            "./models/count_vectorizer.pkl",
            "rb"
        ) as f:
            cls.vectorizer = pickle.load(f)

        cls.holdout_data = pd.read_csv(
            "./data/processed/test_bow.csv"
        )

    def test_model_loaded_properly(self):

        self.assertIsNotNone(self.model)
        self.assertIsNotNone(self.vectorizer)
        self.assertIsNotNone(self.holdout_data)

    def test_model_signature(self):

        input_text = "This movie was fantastic! I loved it."

        input_data = self.vectorizer.transform(
            [input_text]
        )

        input_df = pd.DataFrame(
            input_data.toarray(),
            columns=self.vectorizer.get_feature_names_out()
        )

        prediction = self.model.predict(input_df)

        self.assertEqual(
            input_df.shape[1],
            len(self.vectorizer.get_feature_names_out())
        )

        self.assertEqual(
            len(prediction),
            input_df.shape[0]
        )

        self.assertEqual(
            len(prediction.shape),
            1
        )

    def test_model_performance(self):

        X_test = self.holdout_data.drop(
            "sentiment",
            axis=1
        )

        y_test = self.holdout_data["sentiment"]

        y_pred = self.model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0
        )

        self.assertGreaterEqual(accuracy, 0.4)
        self.assertGreaterEqual(precision, 0.4)
        self.assertGreaterEqual(recall, 0.4)
        self.assertGreaterEqual(f1, 0.4)


if __name__ == "__main__":
    unittest.main()