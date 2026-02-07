"""
Customer Churn Prediction Pipeline.

Responsibilities:
- Load final trained model and operating threshold
- Validate inference input
- Generate churn probabilities and predictions
- Persist prediction outputs

NOTE:
This component is inference-only.
No training, evaluation, or preprocessing logic is included here.
"""

import os
import sys

import pandas as pd

from src.exception import CustomerChurnException
from src.logging import logging
from src.utils.main_utils import load_object, read_json_file
from src.constants.training_pipeline import (
    FINAL_MODEL_PATH,
    OPERATING_THRESHOLD_FILE_PATH,
)


class CustomerChurnPredictor:
    """
    Handles inference for the Customer Churn Prediction system.

    This class:
    - Loads the selected trained model pipeline
    - Applies the learned operating threshold
    - Produces churn probabilities and binary predictions
    """

    def __init__(self) -> None:
        """
        Initialize predictor by loading model artifacts.

        Raises:
            CustomerChurnException: If model or threshold loading fails
        """
        try:
            logging.info("[PREDICTOR INIT] Loading trained model")
            self.model = load_object(FINAL_MODEL_PATH)

            logging.info("[PREDICTOR INIT] Loading operating threshold")
            threshold_data = read_json_file(OPERATING_THRESHOLD_FILE_PATH)['threshold']

            if isinstance(threshold_data, dict):
                self.threshold = threshold_data.get("operating_threshold")
            else:
                self.threshold = threshold_data

            if self.threshold is None:
                raise ValueError("Operating threshold not found or invalid")

            logging.info(
                "[PREDICTOR INIT] Predictor initialized | "
                f"threshold={self.threshold}"
            )

        except Exception as e:
            logging.exception("[PREDICTOR INIT] Initialization failed")
            raise CustomerChurnException(e, sys)

    # ============================================================
    # Validation
    # ============================================================
    @staticmethod
    def _validate_input(input_df: pd.DataFrame) -> None:
        """
        Validate inference input.

        Args:
            input_df (pd.DataFrame): Input features

        Raises:
            ValueError: If input is invalid
        """
        if input_df is None:
            raise ValueError("Input DataFrame is None")

        if not isinstance(input_df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")

        if input_df.empty:
            raise ValueError("Input DataFrame is empty")

    # ============================================================
    # Prediction
    # ============================================================
    def predict(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate churn predictions.

        Args:
            input_df (pd.DataFrame): Feature DataFrame (no target column)

        Returns:
            pd.DataFrame: Original input with prediction columns appended
        """
        try:
            logging.info(
                "[PREDICTION] Prediction request received | "
                f"rows={len(input_df)}"
            )

            self._validate_input(input_df)

            if not hasattr(self.model, "predict_proba"):
                raise AttributeError(
                    "Loaded model does not support probability prediction"
                )

            churn_probabilities = self.model.predict_proba(input_df)[:, 1]
            churn_predictions = (
                churn_probabilities >= self.threshold
            ).astype(int)

            output_df = input_df.copy()
            output_df["churn_probability"] = churn_probabilities.round(4)
            output_df["churn_prediction"] = churn_predictions

            logging.info(
                "[PREDICTION] Prediction completed | "
                f"churn_rate={round(churn_predictions.mean(), 4)}"
            )

            return output_df

        except Exception as e:
            logging.exception("[PREDICTION] Prediction failed")
            raise CustomerChurnException(e, sys)

    # ============================================================
    # Persistence
    # ============================================================
    @staticmethod
    def save_predictions(
        prediction_df: pd.DataFrame,
        output_file_path: str,
    ) -> None:
        """
        Persist prediction results to disk.

        Args:
            prediction_df (pd.DataFrame): Prediction output
            output_file_path (str): Destination CSV path
        """
        try:
            if prediction_df is None or prediction_df.empty:
                raise ValueError("Prediction DataFrame is empty or None")

            os.makedirs(
                os.path.dirname(output_file_path),
                exist_ok=True,
            )

            prediction_df.to_csv(output_file_path, index=False)

            logging.info(
                "[PREDICTION SAVE] Predictions saved | "
                f"path={output_file_path}, rows={len(prediction_df)}"
            )

        except Exception as e:
            logging.exception("[PREDICTION SAVE] Failed to save predictions")
            raise CustomerChurnException(e, sys)


# testing
if __name__ == "__main__":
    test_df = pd.read_csv(r"/workspaces/saas-churn-risk-ml-system-01/rough/processed_data_01.csv")
    predictor = CustomerChurnPredictor()
    output = predictor.predict(test_df)
    print(output)