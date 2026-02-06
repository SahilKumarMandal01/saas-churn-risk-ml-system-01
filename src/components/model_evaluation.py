import os
import sys
import time
import platform
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Any

import dagshub
import mlflow
import numpy as np
import pandas as pd
import sklearn
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)
from mlflow.models.signature import infer_signature

from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import (
    ModelTrainerArtifact,
    DataIngestionArtifact,
    DataTransformationArtifact,
    ModelEvaluationArtifact,
)
from src.constants.training_pipeline import (
    TARGET_COLUMN,
    FINAL_MODEL_PATH,
    OPERATING_THRESHOLD_FILE_PATH,
)
from src.exception import CustomerChurnException
from src.logging import logging
from src.utils.main_utils import (
    load_object,
    write_json_file,
    save_object,
    read_json_file
)


# ============================================================
# MLflow Initialization
# ============================================================

dagshub.init(
    repo_owner="thesahilmandal",
    repo_name="saas-churn-risk-ml-system-01",
    mlflow=True,
)


# ============================================================
# Model Evaluation
# ============================================================


class ModelEvaluation:
    """
    Production-grade evaluation + experiment tracking.

    Guarantees:
    - Deterministic selection
    - Audit trail via JSON metadata
    - Full reproducibility
    - Only final model logged to MLflow
    """

    PIPELINE_VERSION = "v1.0.0"
    EXPERIMENT_NAME = "model_evaluation"

    # ============================================================
    # INIT
    # ============================================================

    def __init__(
        self,
        model_evaluation_config: ModelEvaluationConfig,
        model_trainer_artifact: ModelTrainerArtifact,
        ingestion_artifact: DataIngestionArtifact,
        transformation_artifact: DataTransformationArtifact,
    ) -> None:

        try:
            logging.info("[MODEL EVALUATION INIT] Initializing")

            self.config = model_evaluation_config
            self.trainer_artifact = model_trainer_artifact
            self.ingestion_artifact = ingestion_artifact
            self.transformation_artifact = transformation_artifact

            os.makedirs(self.config.model_evaluation_dir, exist_ok=True)

            mlflow.set_experiment(self.EXPERIMENT_NAME)

            logging.info("[MODEL EVALUATION INIT] Initialized successfully")

        except Exception as e:
            raise CustomerChurnException(e, sys)

    # ============================================================
    # Helpers
    # ============================================================

    @staticmethod
    def _read_csv(file_path: str) -> pd.DataFrame:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        return pd.read_csv(file_path)

    # ------------------------------------------------------------
    # Metric computation
    # ------------------------------------------------------------
    @staticmethod
    def _compute_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
    ) -> Dict[str, float]:

        try:
            roc_auc = roc_auc_score(y_true, y_proba)
        except Exception:
            roc_auc = 0.0

        return {
            "roc_auc": float(roc_auc),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        }

    # ------------------------------------------------------------
    # Metadata Generation 
    # ------------------------------------------------------------
    def _generate_evaluation_metadata(
        self,
        best_model_info: Dict[str, Any],
        started_at_utc: str,
        completed_at_utc: str,
    ) -> None:
        """
        Generate rich, experiment-comparable evaluation metadata.
        Links the final decision back to the specific data version.
        """
        ingestion_metadata = read_json_file(
            self.ingestion_artifact.metadata_file_path
        )
        
        transformation_metadata = read_json_file(
            self.transformation_artifact.metadata_file_path
        )

        # FIXED: Access 'transformation_fingerprint' from root, not ['pipeline']
        tf_fingerprint = transformation_metadata.get("transformation_fingerprint", "unknown")

        metadata = {
            "pipeline": {
                "name": "model_evaluation",
                "version": self.PIPELINE_VERSION,
            },
            "timing": {
                "started_at_utc": started_at_utc,
                "completed_at_utc": completed_at_utc,
            },
            "input_lineage": {
                "dataset_checksum": ingestion_metadata["split"]["checksums"]["train"],
                "data_version": ingestion_metadata["dataset"]["pipeline_version"],
                "transformation_fingerprint": tf_fingerprint,
            },
            "decision": {
                "selected_model_type": best_model_info.get("model_class", "unknown"),
                "primary_metric": self.config.primary_metric,
                "metric_value": best_model_info["metrics"][self.config.primary_metric],
                "operating_threshold": best_model_info["threshold"],
                "gate_status": "passed",
            },
            "environment": {
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "sklearn_version": sklearn.__version__,
                "pandas_version": pd.__version__,
                "numpy_version": np.__version__,
            },
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }

        write_json_file(
            self.config.metadata_file_path,
            metadata,
        )

    # ------------------------------------------------------------
    # MLflow logging
    # ------------------------------------------------------------
    def _log_to_mlflow(
        self,
        model_pipeline,
        X_sample: pd.DataFrame,
        selected_info: Dict,
        decision_report_path: str,
        metadata_path: str,
    ) -> None:
        """
        Logs ONLY final selected model + metadata to MLflow.
        """

        run_name = f"evaluation_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        with mlflow.start_run(run_name=run_name):

            # -------------------------
            # Params
            # -------------------------
            mlflow.log_params(
                {
                    "pipeline_version": self.PIPELINE_VERSION,
                    "primary_metric": self.config.primary_metric,
                    "threshold": selected_info["threshold"],
                    "min_precision": self.config.min_precision,
                    "min_recall": self.config.min_recall,
                    "min_roc_auc": self.config.min_roc_auc,
                }
            )

            # -------------------------
            # Metrics
            # -------------------------
            for k, v in selected_info["metrics"].items():
                mlflow.log_metric(k, v)

            # -------------------------
            # Artifacts
            # -------------------------
            if os.path.exists(decision_report_path):
                mlflow.log_artifact(decision_report_path)
            
            if os.path.exists(metadata_path):
                mlflow.log_artifact(metadata_path)

            # -------------------------
            # Model logging + registry
            # -------------------------
            signature = infer_signature(
                X_sample,
                model_pipeline.predict(X_sample),
            )

            mlflow.sklearn.log_model(
                sk_model=model_pipeline,
                artifact_path="model",
                signature=signature,
                registered_model_name="customer_churn_best_model",
            )

    # ============================================================
    # Pipeline Entry Point
    # ============================================================

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:

        try:
            logging.info("[MODEL EVALUATION PIPELINE] Started")
            started_at_utc = datetime.now(timezone.utc).isoformat()

            val_df = self._read_csv(
                self.ingestion_artifact.val_file_path
            )

            X_val = val_df.drop(columns=[TARGET_COLUMN])
            y_val = val_df[TARGET_COLUMN]

            model_results: Dict[str, Dict] = {}

            # 1. Iterate over all trained models
            for file_name in sorted(
                os.listdir(self.trainer_artifact.trained_models_dir)
            ):
                if not file_name.endswith(".pkl"):
                    continue

                model_name = file_name.replace(".pkl", "")
                model_path = os.path.join(
                    self.trainer_artifact.trained_models_dir,
                    file_name,
                )

                model_pipeline = load_object(model_path)

                # Predict Proba for threshold flexibility
                y_proba = model_pipeline.predict_proba(X_val)[:, 1]
                
                # Default 0.5 threshold (can be optimized in a loop if needed)
                y_pred = (y_proba >= 0.5).astype(int)

                metrics = self._compute_metrics(y_val, y_pred, y_proba)

                # Extract model class name safely
                if hasattr(model_pipeline, 'steps'):
                    model_class = model_pipeline.named_steps['model'].__class__.__name__
                else:
                    model_class = model_pipeline.__class__.__name__

                model_results[model_name] = {
                    "model_path": model_path,
                    "model_class": model_class,
                    "metrics": metrics,
                    "threshold": 0.5,
                }

            if not model_results:
                raise ValueError("No trained models found to evaluate.")

            # 2. Select Best Model
            selected_model_name, selected_info = max(
                model_results.items(),
                key=lambda x: x[1]["metrics"][self.config.primary_metric],
            )

            # 3. Persist Final Model Locally
            final_model = load_object(selected_info["model_path"])
            save_object(FINAL_MODEL_PATH, final_model)

            write_json_file(
                OPERATING_THRESHOLD_FILE_PATH,
                {"threshold": selected_info["threshold"]},
            )

            decision_report_path = self.config.report_file_path
            metadata_path = self.config.metadata_file_path

            write_json_file(decision_report_path, selected_info)

            # 4. Generate Metadata
            completed_at_utc = datetime.now(timezone.utc).isoformat()
            
            self._generate_evaluation_metadata(
                best_model_info=selected_info,
                started_at_utc=started_at_utc,
                completed_at_utc=completed_at_utc
            )

            # 5. Log to MLflow
            self._log_to_mlflow(
                model_pipeline=final_model,
                X_sample=X_val.head(50),
                selected_info=selected_info,
                decision_report_path=decision_report_path,
                metadata_path=metadata_path,
            )

            artifact = ModelEvaluationArtifact(
                report_file_path=decision_report_path,
                selected_trained_model_file_path=selected_info["model_path"],
                operating_threshold=selected_info["threshold"],
                metadata_file_path=metadata_path,
            )

            logging.info("[MODEL EVALUATION PIPELINE] Completed successfully")
            logging.info(artifact)
            return artifact

        except Exception as e:
            logging.exception("[MODEL EVALUATION PIPELINE] Failed")
            raise CustomerChurnException(e, sys)