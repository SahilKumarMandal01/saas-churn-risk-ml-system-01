"""
Model Registry Pipeline.

Responsibilities:
- Promote approved models from evaluation pipeline
- Maintain immutable versioned model storage
- Update production model pointer
- Maintain registry metadata for lineage and rollback

Design Goals:
- Simple filesystem-based registry
- Deterministic model promotion
- Immutable versioning
- Easy rollback capability
- Clean separation from training and evaluation

IMPORTANT:
This pipeline does NOT train or evaluate models.
It only promotes already approved models.
"""

import os
import sys
import shutil
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from src.entity.config_entity import ModelRegistryConfig
from src.entity.artifact_entity import ModelEvaluationArtifact
from src.exception import CustomerChurnException
from src.logging import logging
from src.utils.main_utils import read_json_file, write_json_file


class ModelRegistry:
    """
    Production-oriented Model Registry.

    Responsibilities:
        1. Check approval status from evaluation pipeline
        2. Create new immutable model version
        3. Register model artifact
        4. Update production model pointer
        5. Maintain registry metadata

    Guarantees:
        - Only approved models are promoted
        - Registered versions are immutable
        - Production model always points to latest approved version
        - Registry remains auditable and deterministic
    """

    PIPELINE_VERSION = "1.0.0"

    # ============================================================
    # INIT
    # ============================================================

    def __init__(
        self,
        config: ModelRegistryConfig,
        evaluation_artifact: ModelEvaluationArtifact,
    ) -> None:
        try:
            logging.info("[MODEL REGISTRY INIT] Initializing")

            self.config = config
            self.evaluation_artifact = evaluation_artifact

            os.makedirs(self.config.registry_dir, exist_ok=True)
            os.makedirs(
                os.path.dirname(self.config.production_model_file_path),
                exist_ok=True,
            )

            self.registry_metadata_path = os.path.join(
                self.config.registry_dir,
                "registry_metadata.json",
            )

            logging.info(
                "[MODEL REGISTRY INIT] Initialized | "
                f"pipeline_version={self.PIPELINE_VERSION}"
            )

        except Exception as e:
            raise CustomerChurnException(e, sys)

    # ============================================================
    # HELPERS
    # ============================================================

    def _load_registry_metadata(self) -> Dict[str, Any]:
        """
        Load registry metadata if exists, otherwise initialize.
        """
        if not os.path.exists(self.registry_metadata_path):
            return {
                "current_production_version": None,
                "registered_versions": [],
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            }

        return read_json_file(self.registry_metadata_path)

    def _get_next_version(self, metadata: Dict[str, Any]) -> str:
        """
        Determine next model version.
        """
        versions = metadata.get("registered_versions", [])

        if not versions:
            return "v1"

        latest_version = max(int(v[1:]) for v in versions)
        return f"v{latest_version + 1}"

    def _register_model(self, version: str) -> str:
        """
        Copy approved model into versioned registry directory.
        """
        version_dir = os.path.join(self.config.registry_dir, version)
        os.makedirs(version_dir, exist_ok=True)

        destination_model_path = os.path.join(version_dir, "model.pkl")

        shutil.copy2(
            self.evaluation_artifact.best_model_path,
            destination_model_path,
        )

        logging.info(
            f"[MODEL REGISTRY] Model registered | version={version}"
        )

        return destination_model_path

    def _update_production_model(self, version_model_path: str) -> None:
        """
        Update production model pointer.
        """
        shutil.copy2(
            version_model_path,
            self.config.production_model_file_path,
        )

        logging.info("[MODEL REGISTRY] Production model updated")

    def _update_registry_metadata(
        self,
        metadata: Dict[str, Any],
        version: str,
    ) -> None:
        """
        Update registry metadata after successful promotion.
        """
        metadata.setdefault("registered_versions", []).append(version)
        metadata["current_production_version"] = version
        metadata["last_updated_at_utc"] = datetime.now(
            timezone.utc
        ).isoformat()

        write_json_file(self.registry_metadata_path, metadata)

        logging.info("[MODEL REGISTRY] Registry metadata updated")

    # ============================================================
    # ENTRY POINT
    # ============================================================

    def initiate_model_registry(self) -> Optional[str]:
        """
        Execute Model Registry pipeline.

        Returns:
            Registered version name if promotion occurs,
            otherwise None.
        """
        try:
            logging.info("[MODEL REGISTRY PIPELINE] Started")

            # --------------------------------------------------
            # Approval Gate
            # --------------------------------------------------
            if not self.evaluation_artifact.approval_status:
                logging.info(
                    "[MODEL REGISTRY] Model not approved. "
                    "Skipping registration."
                )
                return None

            # --------------------------------------------------
            # Load Registry State
            # --------------------------------------------------
            metadata = self._load_registry_metadata()

            # --------------------------------------------------
            # Determine Version
            # --------------------------------------------------
            version = self._get_next_version(metadata)

            # --------------------------------------------------
            # Register Model Version
            # --------------------------------------------------
            version_model_path = self._register_model(version)

            # --------------------------------------------------
            # Update Production Model
            # --------------------------------------------------
            self._update_production_model(version_model_path)

            # --------------------------------------------------
            # Update Metadata
            # --------------------------------------------------
            self._update_registry_metadata(metadata, version)

            logging.info(
                "[MODEL REGISTRY PIPELINE] Completed | "
                f"production_version={version}"
            )

            return version

        except Exception as e:
            logging.exception("[MODEL REGISTRY PIPELINE] Failed")
            raise CustomerChurnException(e, sys)
