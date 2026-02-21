# """
# Customer Churn Orchestrator

# Responsibilities:
# - Execute monitoring pipeline
# - Decide whether retraining is required
# - Trigger training pipeline if needed
# - Maintain run-level metadata
# - Ensure idempotent execution via lock file
# - Isolate failures and preserve system consistency

# Design Principles:
# - Deterministic
# - Idempotent
# - Observable
# - Production-safe
# - Extensible
# """

# import os
# import sys
# import json
# import traceback
# from datetime import datetime, timezone
# from pathlib import Path
# from typing import Dict, Any

# from src.components.model_monitoring import ModelMonitoring
# from src.entity.config_entity import ModelMonitoringConfig, OrchestratorConfig
# from src.pipeline.training_pipeline import TrainingPipeline
# from src.exception import CustomerChurnException
# from src.logging import logging
# from src.utils.main_utils import write_json_file
# from src.constants.pipeline_constants import LOCK_FILE_PATH

# class CustomerChurnOrchestrator:
#     """
#     Production-grade orchestrator for automated monitoring and retraining.
#     """
#     PIPELINE_VERSION = "1.0.0"

#     # ============================================================
#     # INIT
#     # ============================================================

#     def __init__(self, orchestrator_config: OrchestratorConfig) -> None:
#         try:
#             logging.info("[ORCHESTRATOR INIT] Initializing")

#             self.config = orchestrator_config
#             self.training_pipeline = TrainingPipeline()
#             self.monitoring_pipeline = ModelMonitoring(
#                 ModelMonitoringConfig()
#             )

#             os.makedirs(self.config.artifact_dir, exist_ok=True)
            
#             logging.info(
#                 f"[ORCHESTRATOR INIT] Initialized | run_id={self.config.run_id}"
#             )

#         except Exception as e:
#             raise CustomerChurnException(e, sys)

#     # ============================================================
#     # LOCKING MECHANISM
#     # ============================================================

#     def _acquire_lock(self) -> None:
#         if os.path.exists(LOCK_FILE_PATH):
#             raise RuntimeError(
#                 "Another orchestrator run is in progress."
#             )

#         with open(LOCK_FILE_PATH, "w") as f:
#             f.write(self.config.run_id)

#     def _release_lock(self) -> None:
#         if os.path.exists(LOCK_FILE_PATH):
#             os.remove(LOCK_FILE_PATH)

#     # ============================================================
#     # RUN METADATA
#     # ============================================================

#     def _initialize_run_metadata(self) -> Dict[str, Any]:
#         return {
#             "pipeline": {
#                 "name": "orchestrator",
#                 "version": self.PIPELINE_VERSION,
#             },
#             "run_id": self.config.run_id,
#             "status": "STARTED",
#             "retraining_triggered": False,
#             "timestamps": {
#                 "started_at_utc": datetime.now(
#                     timezone.utc
#                 ).isoformat()
#             },
#         }

#     def _finalize_run_metadata(
#         self,
#         metadata: Dict[str, Any],
#         status: str,
#     ) -> None:
#         metadata["status"] = status
#         metadata["timestamps"][
#             "completed_at_utc"
#         ] = datetime.now(timezone.utc).isoformat()

#         metadata_path = self.config.metadata_path
#         write_json_file(str(metadata_path), metadata)

#     # ============================================================
#     # MAIN EXECUTION
#     # ============================================================

#     def run(self) -> None:
#         metadata = self._initialize_run_metadata()

#         try:
#             logging.info("[ORCHESTRATOR] Starting run")

#             # Acquire lock
#             self._acquire_lock()

#             # ----------------------------------------------------
#             # Step 1: Monitoring
#             # ----------------------------------------------------
#             logging.info("[ORCHESTRATOR] Running monitoring pipeline")

#             monitoring_artifact = (
#                 self.monitoring_pipeline.initiate_model_monitoring()
#             )

#             metadata["monitoring"] = {
#                 "artifact_dir": monitoring_artifact.artifact_dir,
#                 "retraining_required": monitoring_artifact.retraining_required,
#             }

#             # Persist monitoring snapshot
#             monitoring_snapshot_path = self.config.monitoring_snapshot_path

#             write_json_file(
#                 str(monitoring_snapshot_path),
#                 {
#                     "artifact_dir": monitoring_artifact.artifact_dir,
#                     "retraining_required": monitoring_artifact.retraining_required,
#                 },
#             )

#             # ----------------------------------------------------
#             # Step 2: Decision Logic
#             # ----------------------------------------------------
#             if monitoring_artifact.retraining_required:
#                 logging.info(
#                     "[ORCHESTRATOR] Retraining required. Triggering training pipeline."
#                 )

#                 metadata["retraining_triggered"] = True

#                 # ------------------------------------------------
#                 # Step 3: Training
#                 # ------------------------------------------------
#                 registry_artifact = (
#                     self.training_pipeline.run_pipeline()
#                 )

#                 metadata["training"] = {
#                     "registry_artifact": str(registry_artifact)
#                 }

#                 self._finalize_run_metadata(metadata, "SUCCESS")

#             else:
#                 logging.info(
#                     "[ORCHESTRATOR] No retraining required."
#                 )

#                 self._finalize_run_metadata(metadata, "SUCCESS")

#         except Exception as e:
#             logging.exception("[ORCHESTRATOR] Run failed")

#             metadata["error"] = {
#                 "message": str(e),
#                 "traceback": traceback.format_exc(),
#             }

#             self._finalize_run_metadata(metadata, "FAILED")

#             raise CustomerChurnException(e, sys)

#         finally:
#             self._release_lock()
#             logging.info("[ORCHESTRATOR] Run completed")


# if __name__ == "__main__":
#     try:
#         config = OrchestratorConfig()
#         orchestrator = CustomerChurnOrchestrator(config)
#         orchestrator.run()
#     except Exception as e:
#         logging.exception(
#             "Unhandled exception in orchestrator execution"
#         )
#         sys.exit(1)





"""
Customer Churn Orchestrator

Production orchestration layer responsible for:

- Monitoring execution
- Retraining decisioning
- Training pipeline triggering
- Run metadata lifecycle
- Idempotent execution guarantees

Design Goals:
    - Deterministic
    - Idempotent
    - Observable
    - Failure Isolated
    - Production Safe
"""

from __future__ import annotations

import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from src.components.model_monitoring import ModelMonitoring
from src.entity.config_entity import (
    ModelMonitoringConfig,
    OrchestratorConfig,
)
from src.pipeline.training_pipeline import TrainingPipeline
from src.exception import CustomerChurnException
from src.logging import logging
from src.utils.main_utils import write_json_file
from src.constants.pipeline_constants import LOCK_FILE_PATH


# ============================================================
# LOCK MANAGER
# ============================================================

class OrchestratorLock:
    """
    Ensures single orchestrator execution (idempotency guard).
    """

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id

    def acquire(self) -> None:
        if os.path.exists(LOCK_FILE_PATH):
            raise RuntimeError(
                "Another orchestrator run is already in progress."
            )

        with open(LOCK_FILE_PATH, "w") as f:
            f.write(self.run_id)

    def release(self) -> None:
        if os.path.exists(LOCK_FILE_PATH):
            os.remove(LOCK_FILE_PATH)


# ============================================================
# METADATA MANAGER
# ============================================================

class RunMetadataManager:
    """Handles orchestrator run metadata lifecycle."""

    PIPELINE_VERSION = "1.0.0"

    def __init__(self, config: OrchestratorConfig) -> None:
        self.config = config

    def initialize(self) -> Dict[str, Any]:
        return {
            "pipeline": {
                "name": "orchestrator",
                "version": self.PIPELINE_VERSION,
            },
            "run_id": self.config.run_id,
            "status": "STARTED",
            "retraining_triggered": False,
            "timestamps": {
                "started_at_utc": datetime.now(
                    timezone.utc
                ).isoformat()
            },
        }

    def finalize(self, metadata: Dict[str, Any], status: str) -> None:
        metadata["status"] = status
        metadata["timestamps"]["completed_at_utc"] = (
            datetime.now(timezone.utc).isoformat()
        )

        write_json_file(
            str(self.config.metadata_path),
            metadata,
        )


# ============================================================
# ORCHESTRATOR
# ============================================================

class CustomerChurnOrchestrator:
    """
    Production-grade automated monitoring and retraining orchestrator.
    """

    def __init__(self, config: OrchestratorConfig) -> None:
        try:
            logging.info("[ORCHESTRATOR INIT] Initializing")

            self.config = config

            os.makedirs(self.config.artifact_dir, exist_ok=True)

            # Pipelines
            self.training_pipeline = TrainingPipeline()
            self.monitoring_pipeline = ModelMonitoring(
                ModelMonitoringConfig()
            )

            # Helpers
            self.lock = OrchestratorLock(config.run_id)
            self.metadata_manager = RunMetadataManager(config)

            logging.info(
                "[ORCHESTRATOR INIT] Initialized | run_id=%s",
                self.config.run_id,
            )

        except Exception as e:
            raise CustomerChurnException(e, sys)

    # ========================================================
    # MONITORING STEP
    # ========================================================

    def _execute_monitoring(self, metadata: Dict[str, Any]) -> Any:
        logging.info("[ORCHESTRATOR] Running monitoring pipeline")

        monitoring_artifact = (
            self.monitoring_pipeline.initiate_model_monitoring()
        )

        metadata["monitoring"] = {
            "artifact_dir": monitoring_artifact.artifact_dir,
            "retraining_required": monitoring_artifact.retraining_required,
        }

        write_json_file(
            str(self.config.monitoring_snapshot_path),
            metadata["monitoring"],
        )

        return monitoring_artifact

    # ========================================================
    # TRAINING STEP
    # ========================================================

    def _execute_training(self, metadata: Dict[str, Any]) -> None:
        logging.info(
            "[ORCHESTRATOR] Retraining required. Triggering training pipeline."
        )

        metadata["retraining_triggered"] = True

        registry_artifact = self.training_pipeline.run_pipeline()

        metadata["training"] = {
            "registry_artifact": str(registry_artifact)
        }

    # ========================================================
    # MAIN EXECUTION
    # ========================================================

    def run(self) -> None:
        metadata = self.metadata_manager.initialize()

        try:
            logging.info("[ORCHESTRATOR] Starting run")

            self.lock.acquire()

            # ---------------- Monitoring ----------------
            monitoring_artifact = self._execute_monitoring(metadata)

            # ---------------- Decision ----------------
            if monitoring_artifact.retraining_required:
                self._execute_training(metadata)
            else:
                logging.info(
                    "[ORCHESTRATOR] No retraining required."
                )

            self.metadata_manager.finalize(metadata, "SUCCESS")

        except Exception as e:
            logging.exception("[ORCHESTRATOR] Run failed")

            metadata["error"] = {
                "message": str(e),
                "traceback": traceback.format_exc(),
            }

            self.metadata_manager.finalize(metadata, "FAILED")

            raise CustomerChurnException(e, sys)

        finally:
            self.lock.release()
            logging.info("[ORCHESTRATOR] Run completed")


# ============================================================
# ENTRYPOINT
# ============================================================

def main() -> None:
    try:
        config = OrchestratorConfig()
        orchestrator = CustomerChurnOrchestrator(config)
        orchestrator.run()
    except Exception:
        logging.exception(
            "Unhandled exception in orchestrator execution"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()