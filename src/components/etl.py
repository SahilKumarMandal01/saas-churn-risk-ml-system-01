"""
ETL pipeline for Customer Churn data ingestion.

Responsibilities:
- Extract raw data from Kaggle
- Apply lightweight, non-destructive transformations
- Load cleaned data into MongoDB
- Generate metadata for auditability and observability

NOTE:
This ETL layer intentionally avoids feature engineering,
label processing, and modeling logic.
"""

import os
import sys
from datetime import datetime
from typing import List, Dict

import certifi
import pandas as pd
import pymongo
import subprocess
import warnings
from dotenv import load_dotenv

from src.exception import CustomerChurnException
from src.logging import logging
from src.utils.main_utils import write_json_file
from src.entity.config_entity import ETLconfig, TrainingPipelineConfig
from src.entity.artifact_entity import ETLArtifact

warnings.filterwarnings('ignore')
load_dotenv()


class CustomerChurnETL:
    """
    Orchestrates the end-to-end ETL pipeline for Customer Churn data.

    The pipeline follows a strictly sequential flow:
    Extract → Transform → Load → Metadata generation.

    This class is intentionally scoped to ingestion concerns only.
    """

    def __init__(self, etl_config: ETLconfig) -> None:
        """
        Initialize the ETL pipeline and validate required configuration.

        Args:
            etl_config (ETLconfig): ETL configuration object
        """
        try:
            self.config = etl_config

            self.mongodb_url = os.getenv("MONGODB_URL")
            self.database_name = os.getenv("MONGODB_DATABASE")
            self.collection_name = os.getenv("MONGODB_COLLECTION")
            self.data_source = os.getenv("DATA_SOURCE")

            self.ca_file = certifi.where()

            self._validate_env_variables()
            os.makedirs(self.config.etl_dir, exist_ok=True)

            logging.info("[ETL INIT] CustomerChurnETL initialized successfully.")

        except Exception as e:
            raise CustomerChurnException(e, sys)

    def _validate_env_variables(self) -> None:
        """
        Validate presence of mandatory environment variables.

        Raises:
            EnvironmentError: If any required variable is missing
        """
        required_vars = {
            "MONGODB_URL": self.mongodb_url,
            "MONGODB_DATABASE": self.database_name,
            "MONGODB_COLLECTION": self.collection_name,
            "DATA_SOURCE": self.data_source,
        }

        missing_vars = [key for key, value in required_vars.items() if not value]

        if missing_vars:
            raise EnvironmentError(
                f"Missing required environment variables: {missing_vars}"
            )

    def extract_data(self) -> pd.DataFrame:
        """
        Download dataset from Kaggle and load the first CSV file found.

        Returns:
            pd.DataFrame: Raw extracted dataset
        """
        try:
            logging.info("[ETL EXTRACT] Starting data extraction from Kaggle.")

            download_dir = self.config.raw_data_dir

            subprocess.run(
                [
                    "kaggle",
                    "datasets",
                    "download",
                    "-d",
                    self.data_source,
                    "-p",
                    download_dir,
                    "--unzip",
                ],
                check=True,
            )

            csv_files = [
                os.path.join(download_dir, f)
                for f in os.listdir(download_dir)
                if f.lower().endswith(".csv")
            ]

            if not csv_files:
                raise FileNotFoundError("No CSV files found after Kaggle download.")

            df = pd.read_csv(csv_files[0])

            if df.empty:
                raise ValueError("Extracted CSV file is empty.")

            logging.info(
                "[ETL EXTRACT] Data extracted successfully | "
                f"Rows: {df.shape[0]}, Columns: {df.shape[1]}"
            )

            return df

        except Exception as e:
            raise CustomerChurnException(e, sys)


    def transform_data(self, df: pd.DataFrame) -> List[Dict]:
        """
        Perform lightweight, non-destructive transformations.

        Transformations:
        - Remove duplicate rows
        - Strip whitespace from string columns
        - Append ingestion metadata

        Args:
            df (pd.DataFrame): Raw extracted data

        Returns:
            List[Dict]: Cleaned records ready for database insertion
        """
        try:
            logging.info("[ETL TRANSFORM] Starting data transformation.")

            df_clean = df.copy()
            initial_row_count = df_clean.shape[0]

            df_clean = df_clean.drop_duplicates().reset_index(drop=True)

            for column in df_clean.select_dtypes(include="object").columns:
                df_clean[column] = df_clean[column].str.strip()

            df_clean["data_source"] = self.data_source
            df_clean["ingested_at_utc"] = datetime.utcnow().date().isoformat()

            records = df_clean.to_dict(orient="records")

            logging.info(
                "[ETL TRANSFORM] Transformation completed | "
                f"Rows before: {initial_row_count}, Rows after: {len(records)}"
            )

            return records

        except Exception as e:
            raise CustomerChurnException(e, sys)

    def load_data(self, records: List[Dict]) -> int:
        """
        Load transformed records into MongoDB.

        Args:
            records (List[Dict]): Cleaned data records

        Returns:
            int: Number of records successfully inserted
        """
        if not records:
            logging.warning("[ETL LOAD] No records to insert. Skipping MongoDB load.")
            return 0

        try:
            logging.info("[ETL LOAD] Loading data into MongoDB.")

            with pymongo.MongoClient(
                self.mongodb_url,
                tlsCAFile=self.ca_file
            ) as client:

                collection = client[self.database_name][self.collection_name]

                if self.config.delete_old_data:
                    collection.delete_many({})

                result = collection.insert_many(records, ordered=True)

                inserted_count = len(result.inserted_ids)

            logging.info(
                "[ETL LOAD] Data loaded successfully | "
                f"Inserted records: {inserted_count}"
            )

            return inserted_count

        except Exception as e:
            logging.error(
                "[ETL LOAD] Failed to load data into MongoDB | "
                f"DB: {self.database_name}, Collection: {self.collection_name}"
            )
            raise CustomerChurnException(e, sys)

    def generate_metadata(
        self,
        raw_df: pd.DataFrame,
        records_inserted: int,
    ) -> None:
        """
        Generate and persist ETL metadata for auditability.

        Args:
            raw_df (pd.DataFrame): Raw extracted dataset
            records_inserted (int): Number of records inserted into MongoDB
        """
        try:
            logging.info("[ETL METADATA] Generating ETL metadata.")

            metadata = {
                "data_source": self.data_source,
                "extracted_at_utc": datetime.utcnow().isoformat(),
                "dataset": {
                    "rows_raw": raw_df.shape[0],
                    "columns": raw_df.shape[1],
                    "column_names": list(raw_df.columns),
                    "dtypes": raw_df.dtypes.astype(str).to_dict(),
                },
                "data_quality": {
                    "duplicate_rows_removed": int(raw_df.duplicated().sum()),
                    "missing_values_per_column": raw_df.isnull().sum().to_dict(),
                },
                "load_target": {
                    "database": self.database_name,
                    "collection": self.collection_name,
                    "records_inserted": records_inserted,
                },
            }

            write_json_file(
                file_path=self.config.metadata_file_path,
                content=metadata
            )

            logging.info(
                "[ETL METADATA] Metadata written successfully | "
                f"Path: {self.config.metadata_file_path}"
            )

        except Exception as e:
            raise CustomerChurnException(e, sys)

    def initiate_etl(self) -> ETLArtifact:
        """
        Execute the complete ETL pipeline sequentially.
        """
        try:
            logging.info("[ETL PIPELINE] ETL execution started.")

            raw_df = self.extract_data()
            records = self.transform_data(raw_df)
            inserted_count = self.load_data(records)

            self.generate_metadata(
                raw_df=raw_df,
                records_inserted=inserted_count,
            )

            artifact = ETLArtifact(
                raw_data_dir_path=self.config.raw_data_dir,
                metadata_file_path=self.config.metadata_file_path
            )
            logging.info("[ETL PIPELINE] ETL execution completed successfully.")
            logging.info(f"ETLArtifact: {artifact}")

            return artifact

        except Exception as e:
            raise CustomerChurnException(e, sys)



if __name__ == "__main__":
    training_pipeline_config = TrainingPipelineConfig()
    etl_config = ETLconfig(training_pipeline_config)
    etl = CustomerChurnETL(etl_config)
    artifact = etl.initiate_etl()
    print(artifact)