import os, sys
from datetime import datetime

from src.constants import pipeline_constants
from src.exception import CustomerChurnException


class TrainingPipelineConfig:
    def __init__(self, timestamp: datetime | None = None) -> None:
        try:
            raw_timestamp = timestamp or datetime.now()
            formatted_timestamp = raw_timestamp.strftime("%m_%d_%Y_%H_%M_%S")

            self.artifact_name: str = pipeline_constants.ARTIFACT_DIR
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
                pipeline_constants.ETL_DIR_NAME
            )
            self.metadata_file_path: str = os.path.join(
                self.etl_dir,
                pipeline_constants.ETL_METADATA_FILE_NAME
            )
            self.raw_data_dir: str = os.path.join(
                self.etl_dir,
                pipeline_constants.ETL_RAW_DATA_DIR_NAME
            )
            self.delete_old_data = pipeline_constants.ETL_DELETE_OLD_DATA
        except Exception as e:
            raise CustomerChurnException(e, sys)