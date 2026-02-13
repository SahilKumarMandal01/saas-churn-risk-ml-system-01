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
from datetime import datetime, timezone
from src.utils.main_utils import load_object, read_json_file, read_yaml_file
from src.constants.training_pipeline import (
    FINAL_MODEL_PATH,
    OPERATING_THRESHOLD_FILE_PATH,
    REFERENCE_SCHEMA
)


class CustomerChurnPredictor:
    """
    Handles inference for the Customer Churn Prediction system.
    """

    # Identifier columns allowed in inference input
    IDENTIFIER_COLUMNS = ["CustomerID"]

    def __init__(self) -> None:
        try:
            logging.info("[PREDICTOR INIT] Loading trained model")
            self.model = load_object(FINAL_MODEL_PATH)

            logging.info("[PREDICTOR INIT] Loading operating threshold")
            threshold_data = read_json_file(
                OPERATING_THRESHOLD_FILE_PATH
            )["threshold"]

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
        Validate inference input against reference schema while allowing
        identifier columns.
        """
        try:
            if input_df is None:
                raise ValueError("Input DataFrame is None")

            if not isinstance(input_df, pd.DataFrame):
                raise TypeError("Input must be a pandas DataFrame")

            if input_df.empty:
                raise ValueError("Input DataFrame is empty")

            schema = read_yaml_file(REFERENCE_SCHEMA)

            dataset_config = schema["dataset"]
            columns_schema = schema["columns"]
            dtype_mapping = schema["dtype_mapping"]

            target_column = dataset_config["target_column"]

            identifier_columns = set(
                CustomerChurnPredictor.IDENTIFIER_COLUMNS
            )

            # Target column must not be present
            if target_column in input_df.columns:
                raise ValueError(
                    f"Target column '{target_column}' must not be provided during inference"
                )

            schema_columns = set(columns_schema.keys())
            input_columns = set(input_df.columns)

            # Remove identifier columns from validation scope
            feature_columns = input_columns - identifier_columns

            # Missing required feature columns
            missing_columns = [
                col for col, meta in columns_schema.items()
                if meta.get("required", False)
                and col not in feature_columns
            ]

            if missing_columns:
                raise ValueError(
                    f"Missing required columns: {missing_columns}"
                )

            # Unknown columns (excluding identifiers)
            unknown_columns = feature_columns - schema_columns

            if unknown_columns:
                raise ValueError(
                    f"Unknown feature columns detected: {list(unknown_columns)}"
                )

            # Column-level validation
            for col, meta in columns_schema.items():

                if col not in feature_columns:
                    continue

                series = input_df[col]

                # Nullable enforcement
                if not meta.get("nullable", True):
                    if series.isna().any():
                        raise ValueError(
                            f"Column '{col}' contains null values"
                        )

                # Dtype validation
                expected_dtype = meta.get("expected_dtype")
                if expected_dtype:
                    allowed_types = dtype_mapping.get(
                        expected_dtype, []
                    )

                    if str(series.dtype) not in allowed_types:
                        raise TypeError(
                            f"Column '{col}' has dtype {series.dtype}, "
                            f"expected one of {allowed_types}"
                        )

                # Allowed categorical values
                if "allowed_values" in meta:
                    invalid_values = set(series.dropna().unique()) - set(
                        meta["allowed_values"]
                    )

                    if invalid_values:
                        raise ValueError(
                            f"Column '{col}' contains invalid values: "
                            f"{invalid_values}"
                        )

                # Numeric range checks
                if "min" in meta:
                    if (series.dropna() < meta["min"]).any():
                        raise ValueError(
                            f"Column '{col}' has values below minimum "
                            f"{meta['min']}"
                        )

                if "max" in meta:
                    if (series.dropna() > meta["max"]).any():
                        raise ValueError(
                            f"Column '{col}' has values above maximum "
                            f"{meta['max']}"
                        )

        except Exception as e:
            logging.exception("[INPUT VALIDATION] Validation failed")
            raise CustomerChurnException(e, sys)

    # ============================================================
    # Prediction
    # ============================================================
    def predict(self, input_df: pd.DataFrame) -> pd.DataFrame:
        try:
            logging.info(
                "[PREDICTION] Prediction request received | "
                f"rows={len(input_df)}"
            )

            self._validate_input(input_df)

            identifier_df = input_df[
                [col for col in self.IDENTIFIER_COLUMNS
                 if col in input_df.columns]
            ]

            model_input_df = input_df.drop(
                columns=self.IDENTIFIER_COLUMNS,
                errors="ignore"
            )

            if not hasattr(self.model, "predict_proba"):
                raise AttributeError(
                    "Loaded model does not support probability prediction"
                )

            churn_probabilities = self.model.predict_proba(
                model_input_df
            )[:, 1]

            churn_predictions = (
                churn_probabilities >= self.threshold
            ).astype(int)

            output_df = input_df.copy()
            output_df["churn_probability"] = churn_probabilities.round(4)
            output_df["TimeStamped"] = datetime.now(timezone.utc).isoformat()

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
    test_df = pd.read_csv(
        r"/workspaces/saas-churn-risk-ml-system-01/rough/processed_data_01.csv"
    )
    predictor = CustomerChurnPredictor()
    output = predictor.predict(test_df)
    print(output)
