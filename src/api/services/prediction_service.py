import time
import uuid
import pandas as pd

from src.pipeline.prediction_pipeline import CustomerChurnPredictor


class PredictionService:
    """
    Service layer between API and ML pipeline.
    """

    def __init__(self, predictor: CustomerChurnPredictor):
        self.predictor = predictor
        self.model_version = "v1.0.0"

    def run_batch(self, df: pd.DataFrame) -> dict:
        request_id = str(uuid.uuid4())
        start = time.time()

        predictions_df = self.predictor.predict(df)

        latency_ms = round((time.time() - start) * 1000, 2)
        churn_rate = round(predictions_df["churn_prediction"].mean(), 4)

        return {
            "request_id": request_id,
            "latency_ms": latency_ms,
            "churn_rate": churn_rate,
            "predictions_df": predictions_df,
            "model_version": self.model_version,
        }