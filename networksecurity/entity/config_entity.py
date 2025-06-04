from datetime import datetime
import os

# Import constants for pipeline configuration (e.g., directory names, filenames, etc.)
from networksecurity.constant import training_pipeline

# Print pipeline metadata for quick verification (can be removed in production)
print(training_pipeline.PIPELINE_NAME)
print(training_pipeline.ARTIFACT_DIR)

# =============================================
# TrainingPipelineConfig: Manages general config
# =============================================

class TrainingPipelineConfig:
    """
    Configuration class to define base paths and naming for the entire ML training pipeline.

    This class creates a time-stamped directory structure for organizing all artifacts
    generated during the run (e.g., raw data, models, reports).
    """

    def __init__(self, timestamp=datetime.now()):
        """
        Initialize the pipeline configuration with a timestamp.

        Args:
            timestamp (datetime, optional): Defaults to current time. Used to create unique directory names.
        """
        timestamp = timestamp.strftime("%m_%d_%Y_%H_%M_%S")  # Converts timestamp to readable string

        self.pipeline_name = training_pipeline.PIPELINE_NAME  # e.g., "network_security_pipeline"
        self.artifact_name = training_pipeline.ARTIFACT_DIR  # Base artifact directory name, e.g., "artifact"
        self.artifact_dir = os.path.join(self.artifact_name, timestamp)  # Full path: artifact/<timestamp>
        self.timestamp: str = timestamp  # Store formatted timestamp for reuse

# =========================================================
# DataIngestionConfig: Configures how raw data is ingested
# =========================================================

class DataIngestionConfig:
    """
    Configuration class for the data ingestion phase.

    This defines where to store raw data, feature store files,
    training and test datasets, and how to split them.
    """

    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        """
        Initializes file paths and metadata required for data ingestion.

        Args:
            training_pipeline_config (TrainingPipelineConfig): Base pipeline config that provides root artifact directory.
        """

        # Root directory for all data ingestion outputs
        self.data_ingestion_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.DATA_INGESTION_DIR_NAME  # e.g., "data_ingestion"
        )

        # Path to save the raw feature store file (raw clean data used for analysis/EDA)
        self.feature_store_file_path: str = os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_FEATURE_STORE_DIR,  # e.g., "feature_store"
            training_pipeline.FILE_NAME  # e.g., "data.csv"
        )

        # Path to save the ingested training file after splitting
        self.training_file_path: str = os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_INGESTED_DIR,  # e.g., "ingested"
            training_pipeline.TRAIN_FILE_NAME
        )

        # Path to save the ingested testing file after splitting
        self.testing_file_path: str = os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_INGESTED_DIR,
            training_pipeline.TEST_FILE_NAME  # e.g., "test.csv"
        )

        # Ratio to split data into training and test sets (e.g., 0.2 means 80% train, 20% test)
        self.train_test_split_ratio: float = training_pipeline.DATA_INGESTION_TRAIN_TEST_SPLIT_RATION

        # MongoDB collection from where data will be fetched for ingestion
        self.collection_name: str = training_pipeline.DATA_INGESTION_COLLECTION_NAME

        # MongoDB database name
        self.database_name: str = training_pipeline.DATA_INGESTION_DATABASE_NAME
