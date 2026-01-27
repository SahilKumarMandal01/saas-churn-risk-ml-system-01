from pathlib import Path
import os
import numpy as np
from dotenv import load_dotenv

# Load environment variables once at module import
load_dotenv()

# -------------------------------------------------------------------------
# Global Pipeline Constants
# -------------------------------------------------------------------------

TARGET_COLUMN: str = "Churn"
ARTIFACT_DIR: Path = Path("artifacts")
S3_BUCKET_NAME: str = "saas-customer-churn-ml"
RANDOM_STATE = 42


# -------------------------------------------------------------------------
# ETL Constants
# -------------------------------------------------------------------------

ETL_DIR_NAME: str = "01_etl"
ETL_METADATA_FILE_NAME: str = "metadata.json"
ETL_RAW_DATA_DIR_NAME: str = "raw_data"
ETL_DELETE_OLD_DATA: bool = True


# -------------------------------------------------------------------------
# Data Ingestion Constants
# -------------------------------------------------------------------------

DATA_INGESTION_DIR_NAME: str = "02_data_ingestion"

DATA_INGESTION_TRAIN_FILE_NAME: str = "train.csv"
DATA_INGESTION_TEST_FILE_NAME: str = "test.csv"
DATA_INGESTION_VAL_FILE_NAME: str = "val.csv"

DATA_INGESTION_SCHEMA_FILE_NAME: str = "ingestion_schema.json"
DATA_INGESTION_METADATA_FILE_NAME: str = "metadata.json"

DATA_INGESTION_TRAIN_TEMP_SPLIT_RATIO: float = 0.30
DATA_INGESTION_TEST_VAL_SPLIT_RATIO: float = 0.50

DATA_INGESTION_DATABASE_NAME: str | None = os.getenv("MONGODB_DATABASE")
DATA_INGESTION_COLLECTION_NAME: str | None = os.getenv("MONGODB_COLLECTION")
DATA_INGESTION_MONGODB_URL: str | None = os.getenv("MONGODB_URL")


# -------------------------------------------------------------------------
# Data Validation Constants
# -------------------------------------------------------------------------

DATA_VALIDATION_DIR_NAME: str = "03_data_validation"
DATA_VALIDATION_REPORT_FILE_NAME: str = "report.json"

DATA_VALIDATION_REFERENCE_SCHEMA: Path = Path("data_schema") / "schema.yaml"
