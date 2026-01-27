from dataclasses import dataclass


@dataclass(frozen=True)
class ETLArtifact:
    raw_data_dir_path: str
    metadata_file_path: str

    def __str__(self) -> str:
        return (
            "\nETLArtifact(\n"
            f"  raw_data_file_path = {self.raw_data_dir_path}\n"
            f"  metadata_file_path = {self.metadata_file_path}\n"
            ")"
        )


@dataclass(frozen=True)
class DataIngestionArtifact:
    """
    Artifact generated after the Data Ingestion stage.

    Attributes:
        train_file_path (str): Path to the training dataset
        test_file_path (str): Path to the test dataset
        val_file_path (str): Path to the validation dataset
        schema_file_path (str): Path to the schema file
        metadata_file_path (str): Path to metadata generated during ingestion
    """
    train_file_path: str
    test_file_path: str
    val_file_path: str
    schema_file_path: str
    metadata_file_path: str

    def __str__(self) -> str:
        return (
            "\nDataIngestionArtifact(\n"
            f"  train_file_path    = {self.train_file_path}\n"
            f"  test_file_path     = {self.test_file_path}\n"
            f"  val_file_path      = {self.val_file_path}\n"
            f"  schema_file_path   = {self.schema_file_path}\n"
            f"  metadata_file_path = {self.metadata_file_path}\n"
            ")"
        )
    