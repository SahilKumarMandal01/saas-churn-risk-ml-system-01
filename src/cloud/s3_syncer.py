"""
S3 Synchronization Utility.

Responsibilities:
- Sync local directories to Amazon S3
- Sync S3 directories back to local storage
- Provide robust error handling and logging
- Act as a thin, reliable wrapper over AWS CLI

NOTE:
- Uses AWS CLI intentionally (not boto3) to keep behavior
  consistent with infra tooling and IAM policies.
- Assumes AWS CLI is installed and credentials are configured.
"""

import os
import sys
import subprocess
from typing import List, Optional

from src.logging import logging
from src.exception import CustomerChurnException


class S3Sync:
    """
    Handles synchronization of directories between local filesystem and S3.
    """

    # ------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------
    @staticmethod
    def _validate_local_directory(path: str) -> None:
        """
        Validate that a local directory exists and is a directory.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Local directory does not exist: {path}"
            )

        if not os.path.isdir(path):
            raise NotADirectoryError(
                f"Path is not a directory: {path}"
            )

    @staticmethod
    def _run_command(command: List[str]) -> None:
        """
        Execute a shell command safely and validate its result.
        """
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.returncode != 0:
            logging.error(
                "[S3 SYNC] Command failed | "
                f"command={' '.join(command)} | "
                f"stderr={result.stderr.strip()}"
            )
            raise RuntimeError(result.stderr.strip())

        if result.stdout:
            logging.info(
                "[S3 SYNC] Command output | "
                f"{result.stdout.strip()}"
            )

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------
    def sync_folder_to_s3(
        self,
        folder: str,
        aws_bucket_url: str,
        extra_args: Optional[List[str]] = None,
    ) -> None:
        """
        Sync a local directory to an S3 location.

        Args:
            folder (str): Local directory path
            aws_bucket_url (str): Destination S3 URL (s3://bucket/path)
            extra_args (Optional[List[str]]): Extra AWS CLI flags
        """
        try:
            self._validate_local_directory(folder)

            command = ["aws", "s3", "sync", folder, aws_bucket_url]

            if extra_args:
                command.extend(extra_args)

            logging.info(
                "[S3 SYNC] Upload started | "
                f"local={folder}, destination={aws_bucket_url}"
            )

            self._run_command(command)

            logging.info(
                "[S3 SYNC] Upload completed | "
                f"destination={aws_bucket_url}"
            )

        except Exception as e:
            raise CustomerChurnException(e, sys)

    def sync_folder_from_s3(
        self,
        folder: str,
        aws_bucket_url: str,
        extra_args: Optional[List[str]] = None,
    ) -> None:
        """
        Sync an S3 directory to a local folder.

        Args:
            folder (str): Local destination directory
            aws_bucket_url (str): Source S3 URL (s3://bucket/path)
            extra_args (Optional[List[str]]): Extra AWS CLI flags
        """
        try:
            os.makedirs(folder, exist_ok=True)

            command = ["aws", "s3", "sync", aws_bucket_url, folder]

            if extra_args:
                command.extend(extra_args)

            logging.info(
                "[S3 SYNC] Download started | "
                f"source={aws_bucket_url}, destination={folder}"
            )

            self._run_command(command)

            logging.info(
                "[S3 SYNC] Download completed | "
                f"destination={folder}"
            )

        except Exception as e:
            raise CustomerChurnException(e, sys)