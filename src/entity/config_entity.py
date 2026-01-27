import os, sys
from datetime import datetime

from src.constants import training_pipeline
from src.exception import CustomerChurnException


class TrainingPipelineConfig:
    def __init__(self, timestamp: datetime | None = None) -> None:
        try:
            raw_timestamp = timestamp or datetime.now()
            formatted_timestamp = raw_timestamp.strftime("%m_%d_%Y_%H_%M_%S")

            self.artifact_name: str = training_pipeline.ARTIFACT_DIR
            self.artifact_dir: str = os.path.join(
                self.artifact_name,
                formatted_timestamp
            )
            self.timestamp: str = formatted_timestamp

        except Exception as e:
            raise CustomerChurnException(e, sys)


class ETLconfig:
    def __init__(
        self,
        training_pipeline_config: TrainingPipelineConfig
    ) -> None:
        try:
            self.etl_dir: str = os.path.join(
                training_pipeline_config.artifact_dir,
                training_pipeline.ETL_DIR_NAME
            )
            self.metadata_file_path: str = os.path.join(
                self.etl_dir,
                training_pipeline.ETL_METADATA_FILE_NAME
            )
            self.raw_data_dir: str = os.path.join(
                self.etl_dir,
                training_pipeline.ETL_RAW_DATA_DIR_NAME
            )
            self.delete_old_data = training_pipeline.ETL_DELETE_OLD_DATA
        except Exception as e:
            raise CustomerChurnException(e, sys)


class DataIngestionConfig:
    def __init__(
        self,
        training_pipeline_config: TrainingPipelineConfig
    ) -> None:
        try:
            self.data_ingestion_dir: str = os.path.join(
                training_pipeline_config.artifact_dir,
                training_pipeline.DATA_INGESTION_DIR_NAME
            )
            self.train_file_path: str = os.path.join(
                self.data_ingestion_dir,
                training_pipeline.DATA_INGESTION_TRAIN_FILE_NAME
            )
            self.test_file_path: str = os.path.join(
                self.data_ingestion_dir,
                training_pipeline.DATA_INGESTION_TEST_FILE_NAME
            )
            self.val_file_path: str = os.path.join(
                self.data_ingestion_dir,
                training_pipeline.DATA_INGESTION_VAL_FILE_NAME
            )
            self.schema_file_path: str = os.path.join(
                self.data_ingestion_dir,
                training_pipeline.DATA_INGESTION_SCHEMA_FILE_NAME
            )
            self.metadata_file_path: str = os.path.join(
                self.data_ingestion_dir,
                training_pipeline.DATA_INGESTION_METADATA_FILE_NAME
            )
            self.train_temp_split_ratio: float = (
                training_pipeline.DATA_INGESTION_TRAIN_TEMP_SPLIT_RATIO
            )
            self.test_val_split_ratio: float = (
                training_pipeline.DATA_INGESTION_TEST_VAL_SPLIT_RATIO
            )
            self.random_state = training_pipeline.RANDOM_STATE
            self.database_name = training_pipeline.DATA_INGESTION_DATABASE_NAME
            self.collection_name = training_pipeline.DATA_INGESTION_COLLECTION_NAME
            self.database_url = training_pipeline.DATA_INGESTION_MONGODB_URL
            
        except Exception as e:
            raise CustomerChurnException(e, sys)
        

class DataValidationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        try:
            self.data_validation_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.DATA_VALIDATION_DIR_NAME,
            )
            self.validation_report_file_path: str = os.path.join(
                self.data_validation_dir,
                training_pipeline.DATA_VALIDATION_REPORT_FILE_NAME
            )
            self.reference_schema_file_path = training_pipeline.DATA_VALIDATION_REFERENCE_SCHEMA

        except Exception as e:
            raise CustomerChurnException(e, sys)


class DataTransformationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        try:
            self.data_transformation_dir: str = os.path.join(
                training_pipeline_config.artifact_dir,
                training_pipeline.DATA_TRANSFORMATION_DIR_NAME
            )
            self.lr_preprocessor_file_path: str = os.path.join(
                self.data_transformation_dir,
                training_pipeline.DATA_TRANSFORMATION_LINEAR_PREPROCESSOR_FILE_NAME
            )  
            self.tree_preprocessor_file_path: str = os.path.join(
                self.data_transformation_dir,
                training_pipeline.DATA_TRANSFORMATION_TREE_PREPROCESSOR_FILE_NAME
            )   
            self.metadata_file_path: str = os.path.join(
                self.data_transformation_dir,
                training_pipeline.DATA_TRANSFORMATION_METADATA_FILE_NAME
            )
                    
        except Exception as e:
            raise CustomerChurnException(e, sys)