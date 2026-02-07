import sys
from typing import Optional

from src.components.etl import CustomerChurnETL
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_training import ModelTrainer
from src.components.model_evaluation import ModelEvaluation

from src.entity.config_entity import (
    TrainingPipelineConfig,
    ETLconfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainingConfig,
    ModelEvaluationConfig,
)

from src.entity.artifact_entity import (
    ETLArtifact,
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ModelEvaluationArtifact,
)

from src.logging import logging
from src.exception import CustomerChurnException

from src.cloud.s3_syncer import S3Sync
from src.constants.training_pipeline import TRAINING_BUCKET_NAME


class TrainingPipeline:
    """
    Orchestrates the end-to-end machine learning training lifecycle.

    Responsibilities
    ----------------
    1. Sequential orchestration of all ML pipeline stages.
    2. Managing artifact flow between stages.
    3. Providing consistent structured logging.
    4. Synchronizing generated artifacts to cloud storage.

    Design Principles
    -----------------
    - Single responsibility per stage
    - Explicit dependency passing between stages
    - Deterministic execution order
    - Fail-fast behavior
    - Production-grade observability
    """

    # ============================================================
    # Initialization
    # ============================================================

    def __init__(self) -> None:
        """
        Initialize pipeline configuration and infrastructure dependencies.
        """
        try:
            self.pipeline_config = TrainingPipelineConfig()
            self.s3_sync = S3Sync()

            logging.info("==================================================")
            logging.info("TRAINING PIPELINE INITIALIZED")
            logging.info("==================================================\n")

        except Exception as e:
            raise CustomerChurnException(e, sys)

    # ============================================================
    # Logging Helpers
    # ============================================================

    @staticmethod
    def _log_stage_start(stage_name: str) -> None:
        logging.info(f">>>>>> {stage_name} STARTED <<<<<<")

    @staticmethod
    def _log_stage_end(stage_name: str) -> None:
        logging.info(f">>>>>> {stage_name} COMPLETED <<<<<<\n")

    # ============================================================
    # Pipeline Stages
    # ============================================================

    def start_etl(self) -> ETLArtifact:
        """
        Stage 1: Execute ETL pipeline.
        """
        try:
            self._log_stage_start("Stage 1: ETL")

            etl_config = ETLconfig(
                training_pipeline_config=self.pipeline_config
            )
            etl = CustomerChurnETL(etl_config=etl_config)
            artifact = etl.initiate_etl()

            self._log_stage_end("Stage 1: ETL")
            return artifact

        except Exception as e:
            logging.exception("ETL stage failed")
            raise CustomerChurnException(e, sys)

    def start_data_ingestion(self) -> DataIngestionArtifact:
        """
        Stage 2: Execute data ingestion pipeline.
        """
        try:
            self._log_stage_start("Stage 2: Data Ingestion")

            ingestion_config = DataIngestionConfig(
                training_pipeline_config=self.pipeline_config
            )
            ingestion = DataIngestion(config=ingestion_config)
            artifact = ingestion.initiate_data_ingestion()

            self._log_stage_end("Stage 2: Data Ingestion")
            return artifact

        except Exception as e:
            logging.exception("Data ingestion stage failed")
            raise CustomerChurnException(e, sys)

    def start_data_validation(
        self, ingestion_artifact: DataIngestionArtifact
    ) -> DataValidationArtifact:
        """
        Stage 3: Execute data validation pipeline.
        """
        try:
            self._log_stage_start("Stage 3: Data Validation")

            validation_config = DataValidationConfig(
                training_pipeline_config=self.pipeline_config
            )

            validation = DataValidation(
                config=validation_config,
                ingestion_artifact=ingestion_artifact,
            )

            artifact = validation.initiate_data_validation()

            self._log_stage_end("Stage 3: Data Validation")
            return artifact

        except Exception as e:
            logging.exception("Data validation stage failed")
            raise CustomerChurnException(e, sys)

    def start_data_transformation(
        self,
        ingestion_artifact: DataIngestionArtifact,
        validation_artifact: DataValidationArtifact,
    ) -> DataTransformationArtifact:
        """
        Stage 4: Execute data transformation pipeline.
        """
        try:
            self._log_stage_start("Stage 4: Data Transformation")

            transformation_config = DataTransformationConfig(
                training_pipeline_config=self.pipeline_config
            )

            transformation = DataTransformation(
                transformation_config=transformation_config,
                ingestion_artifact=ingestion_artifact,
                validation_artifact=validation_artifact,
            )

            artifact = transformation.initiate_data_transformation()

            self._log_stage_end("Stage 4: Data Transformation")
            return artifact

        except Exception as e:
            logging.exception("Data transformation stage failed")
            raise CustomerChurnException(e, sys)

    def start_model_training(
        self,
        ingestion_artifact: DataIngestionArtifact,
        transformation_artifact: DataTransformationArtifact,
    ) -> ModelTrainerArtifact:
        """
        Stage 5: Execute model training pipeline.
        """
        try:
            self._log_stage_start("Stage 5: Model Training")

            training_config = ModelTrainingConfig(
                training_pipeline_config=self.pipeline_config
            )

            trainer = ModelTrainer(
                model_trainer_config=training_config,
                ingestion_artifact=ingestion_artifact,
                transformation_artifact=transformation_artifact,
            )

            artifact = trainer.initiate_model_training()

            self._log_stage_end("Stage 5: Model Training")
            return artifact

        except Exception as e:
            logging.exception("Model training stage failed")
            raise CustomerChurnException(e, sys)

    def start_model_evaluation(
        self,
        training_artifact: ModelTrainerArtifact,
        ingestion_artifact: DataIngestionArtifact,
        transformation_artifact: DataTransformationArtifact,
    ) -> ModelEvaluationArtifact:
        """
        Stage 6: Execute model evaluation pipeline.
        """
        try:
            self._log_stage_start("Stage 6: Model Evaluation")

            evaluation_config = ModelEvaluationConfig(
                training_pipeline_config=self.pipeline_config
            )

            evaluation = ModelEvaluation(
                model_evaluation_config=evaluation_config,
                model_trainer_artifact=training_artifact,
                ingestion_artifact=ingestion_artifact,
                transformation_artifact=transformation_artifact,
            )

            artifact = evaluation.initiate_model_evaluation()

            self._log_stage_end("Stage 6: Model Evaluation")
            return artifact

        except Exception as e:
            logging.exception("Model evaluation stage failed")
            raise CustomerChurnException(e, sys)

    # ============================================================
    # Cloud Synchronization
    # ============================================================

    def sync_artifacts_to_s3(self) -> None:
        """
        Synchronize generated artifacts and final model to S3.

        Executed only after successful pipeline completion.
        """
        try:
            logging.info("Starting artifact synchronization to S3")

            artifacts_s3_key = (
                f"artifacts/{self.pipeline_config.timestamp}"
            )
            artifacts_s3_url = (
                f"s3://{TRAINING_BUCKET_NAME}/{artifacts_s3_key}"
            )

            self.s3_sync.sync_folder_to_s3(
                folder=self.pipeline_config.artifact_dir,
                aws_bucket_url=artifacts_s3_url,
            )

            logging.info(f"Artifacts synced to: {artifacts_s3_url}")

            final_model_s3_key = (
                f"final_model/{self.pipeline_config.timestamp}"
            )
            final_model_s3_url = (
                f"s3://{TRAINING_BUCKET_NAME}/{final_model_s3_key}"
            )

            self.s3_sync.sync_folder_to_s3(
                folder="final_model",
                aws_bucket_url=final_model_s3_url,
            )

            logging.info(f"Final model synced to: {final_model_s3_url}")
            logging.info("Artifact synchronization completed successfully\n")

        except Exception as e:
            logging.exception("Artifact synchronization failed")
            raise CustomerChurnException(e, sys)

    # ============================================================
    # Pipeline Execution
    # ============================================================

    def run_pipeline(self) -> ModelEvaluationArtifact:
        """
        Execute the full training pipeline sequentially.
        """
        try:
            logging.info("==================================================")
            logging.info("TRAINING PIPELINE EXECUTION STARTED")
            logging.info("==================================================\n")

            etl_artifact = self.start_etl()

            ingestion_artifact = self.start_data_ingestion()

            validation_artifact = self.start_data_validation(
                ingestion_artifact=ingestion_artifact
            )

            transformation_artifact = self.start_data_transformation(
                ingestion_artifact=ingestion_artifact,
                validation_artifact=validation_artifact,
            )

            training_artifact = self.start_model_training(
                ingestion_artifact=ingestion_artifact,
                transformation_artifact=transformation_artifact,
            )

            evaluation_artifact = self.start_model_evaluation(
                training_artifact=training_artifact,
                ingestion_artifact=ingestion_artifact,
                transformation_artifact=transformation_artifact,
            )

            self.sync_artifacts_to_s3()

            logging.info("==================================================")
            logging.info("TRAINING PIPELINE EXECUTION COMPLETED")
            logging.info("==================================================\n")

            return evaluation_artifact

        except Exception as e:
            logging.exception("Training pipeline execution failed")
            raise CustomerChurnException(e, sys)


if __name__ == "__main__":
    try:
        pipeline = TrainingPipeline()
        pipeline.run_pipeline()
    except Exception as e:
        logging.exception("Unhandled exception in main execution")
        print(e)
