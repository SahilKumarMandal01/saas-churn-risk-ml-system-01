"""
Data Transformation Pipeline.

Responsibilities:
- Build model-aware preprocessing pipelines
- Fit preprocessors on training data only (no leakage)
- Persist fitted preprocessors and rich transformation metadata
"""

import os
import sys
import json
import hashlib
import platform
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Any

import pandas as pd
import sklearn

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
)
from src.exception import CustomerChurnException
from src.logging import logging
from src.utils.main_utils import write_json_file, save_object, read_json_file
from src.constants.training_pipeline import TARGET_COLUMN


NUMERICAL_FEATURES: List[str] = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]


class DataTransformation:
    """
    Production-grade data transformation pipeline.

    Guarantees:
    - Validation-gated execution
    - No data leakage
    - Deterministic preprocessing
    - Experiment-comparable transformation metadata
    """

    PIPELINE_VERSION = "1.0.0"

    def __init__(
        self,
        transformation_config: DataTransformationConfig,
        ingestion_artifact: DataIngestionArtifact,
        validation_artifact: DataValidationArtifact,
    ) -> None:
        try:
            logging.info("[DATA TRANSFORMATION INIT] Initializing")

            if not validation_artifact.validation_status:
                raise ValueError(
                    "Data validation failed. Transformation aborted."
                )

            self.config = transformation_config
            self.ingestion_artifact = ingestion_artifact
            self.validation_artifact = validation_artifact

            os.makedirs(
                self.config.data_transformation_dir,
                exist_ok=True,
            )

            logging.info(
                "[DATA TRANSFORMATION INIT] Initialized | "
                f"pipeline_version={self.PIPELINE_VERSION}"
            )

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

    @staticmethod
    def _compute_hash(obj: Any) -> str:
        """
        Compute a stable SHA-256 hash for JSON-serializable objects.
        """
        payload = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _get_feature_groups(
        X: pd.DataFrame,
    ) -> Tuple[List[str], List[str]]:
        """
        Split features into numerical and categorical groups
        with explicit validation.
        """
        missing = [c for c in NUMERICAL_FEATURES if c not in X.columns]
        if missing:
            raise ValueError(
                f"Expected numerical features missing: {missing}"
            )

        numerical_features = list(NUMERICAL_FEATURES)
        categorical_features = [
            col for col in X.columns if col not in numerical_features
        ]

        return numerical_features, categorical_features

    # ============================================================
    # Preprocessor Builders
    # ============================================================
    def _build_linear_preprocessor(
        self, num_features: List[str], cat_features: List[str]
    ) -> ColumnTransformer:
        """
        Preprocessor for linear models.
        """
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                (
                    "encoder",
                    OneHotEncoder(
                        drop="first",
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
                )
            ]
        )

        return ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, num_features),
                ("cat", categorical_pipeline, cat_features),
            ],
            remainder="drop",
        )

    def _build_tree_preprocessor(
        self, num_features: List[str], cat_features: List[str]
    ) -> ColumnTransformer:
        """
        Preprocessor for tree-based models.
        """
        numeric_pipeline = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="median"))]
        )

        categorical_pipeline = Pipeline(
            steps=[
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
                )
            ]
        )

        return ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, num_features),
                ("cat", categorical_pipeline, cat_features),
            ],
            remainder="drop",
        )

    # ============================================================
    # Metadata
    # ============================================================
    def _extract_encoder_metadata(
        self, preprocessor: ColumnTransformer, cat_features: List[str]
    ) -> Dict[str, Any]:
        """
        Extract category cardinalities and feature names
        from a fitted ColumnTransformer.
        """
        metadata: Dict[str, Any] = {
            "categorical_cardinality": {},
            "output_feature_names": [],
        }

        for name, transformer, features in preprocessor.transformers_:
            if name == "cat":
                encoder: OneHotEncoder = transformer.named_steps["encoder"]
                for feature, categories in zip(features, encoder.categories_):
                    metadata["categorical_cardinality"][feature] = len(categories)

                metadata["output_feature_names"].extend(
                    encoder.get_feature_names_out(features).tolist()
                )

            if name == "num":
                metadata["output_feature_names"].extend(features)

        metadata["output_feature_count"] = len(
            metadata["output_feature_names"]
        )

        return metadata

    def _generate_metadata(
        self,
        X_train: pd.DataFrame,
        num_features: List[str],
        cat_features: List[str],
        linear_preprocessor: ColumnTransformer,
        tree_preprocessor: ColumnTransformer,
    ) -> Dict[str, Any]:
        """
        Generate experiment-comparable transformation metadata.
        """
        ingestion_metadata = read_json_file(
            self.ingestion_artifact.metadata_file_path
        )

        linear_meta = self._extract_encoder_metadata(
            linear_preprocessor, cat_features
        )
        tree_meta = self._extract_encoder_metadata(
            tree_preprocessor, cat_features
        )

        transformation_config = {
            "numerical_features": num_features,
            "categorical_features": cat_features,
            "linear_pipeline": {
                "imputer": "median",
                "scaler": "StandardScaler",
                "encoder": "OneHotEncoder(drop='first')",
            },
            "tree_pipeline": {
                "imputer": "median",
                "encoder": "OneHotEncoder",
            },
        }

        return {
            "pipeline": {
                "name": "data_transformation",
                "version": self.PIPELINE_VERSION,
            },
            "input": {
                "dataset_checksum": ingestion_metadata["split"]["checksums"]["train"],
                "rows": len(X_train),
                "features": len(X_train.columns),
            },
            "feature_groups": {
                "numerical": num_features,
                "categorical": cat_features,
            },
            "output_schema": {
                "linear": linear_meta,
                "tree": tree_meta,
            },
            "transformation_fingerprint": {
                "config_hash": self._compute_hash(transformation_config),
            },
            "upstream_validation": {
                "passed": self.validation_artifact.validation_status,
                "schema_version": ingestion_metadata["dataset"]["pipeline_version"],
            },
            "environment": {
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "sklearn_version": sklearn.__version__,
                "pandas_version": pd.__version__,
            },
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }

    # ============================================================
    # Pipeline Entry Point
    # ============================================================
    def initiate_data_transformation(
        self,
    ) -> DataTransformationArtifact:
        try:
            logging.info("[DATA TRANSFORMATION PIPELINE] Started")

            train_df = self._read_csv(
                self.ingestion_artifact.train_file_path
            )

            if TARGET_COLUMN not in train_df.columns:
                raise ValueError(
                    f"Target column '{TARGET_COLUMN}' not found in training data"
                )

            X_train = train_df.drop(columns=[TARGET_COLUMN])

            num_features, cat_features = self._get_feature_groups(X_train)

            logging.info(
                "[DATA TRANSFORMATION] Training data loaded | "
                f"rows={len(X_train)}, cols={len(X_train.columns)}"
            )

            # -------- Linear Preprocessor --------
            linear_preprocessor = self._build_linear_preprocessor(
                num_features, cat_features
            )
            linear_preprocessor.fit(X_train)

            save_object(
                self.config.lr_preprocessor_file_path,
                linear_preprocessor,
            )

            # -------- Tree Preprocessor --------
            tree_preprocessor = self._build_tree_preprocessor(
                num_features, cat_features
            )
            tree_preprocessor.fit(X_train)

            save_object(
                self.config.tree_preprocessor_file_path,
                tree_preprocessor,
            )

            # -------- Metadata --------
            metadata = self._generate_metadata(
                X_train,
                num_features,
                cat_features,
                linear_preprocessor,
                tree_preprocessor,
            )

            write_json_file(
                self.config.metadata_file_path,
                metadata,
            )

            artifact = DataTransformationArtifact(
                tree_preprocessor_file_path=(
                    self.config.tree_preprocessor_file_path
                ),
                linear_preprocessor_file_path=(
                    self.config.lr_preprocessor_file_path
                ),
                metadata_file_path=self.config.metadata_file_path,
            )

            logging.info(
                "[DATA TRANSFORMATION PIPELINE] Completed successfully"
            )
            logging.info(artifact)
            
            return artifact

        except Exception as e:
            logging.exception(
                "[DATA TRANSFORMATION PIPELINE] Failed"
            )
            raise CustomerChurnException(e, sys)