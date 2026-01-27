from pathlib import Path


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