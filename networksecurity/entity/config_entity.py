from datetime import datetime
import os

# Import constants for pipeline configuration (directory names, filenames, etc.)
from networksecurity.constant import training_pipeline

# Optional: print pipeline metadata (can be removed in production)
print(training_pipeline.PIPELINE_NAME)
print(training_pipeline.ARTIFACT_DIR)


# =============================================
# TrainingPipelineConfig: Manages base artifact paths
# =============================================
class TrainingPipelineConfig:
    """
    Configuration class to define base paths and naming for the ML training pipeline.
    It uses a timestamp to uniquely organize artifacts per pipeline execution.
    """

    def __init__(self, timestamp=datetime.now()):
        """
        Initialize pipeline configuration.

        Args:
            timestamp (datetime, optional): Defaults to current time.
                                             Used to generate unique directory for each run.
        """
        # Format timestamp to be filesystem-safe and human-readable
        timestamp = timestamp.strftime("%m_%d_%Y_%H_%M_%S")

        # Name of the pipeline (e.g., "network_security_pipeline")
        self.pipeline_name = training_pipeline.PIPELINE_NAME

        # Base name for artifact storage (e.g., "artifact")
        self.artifact_name = training_pipeline.ARTIFACT_DIR

        # Complete path for artifact directory including timestamp
        self.artifact_dir = os.path.join(self.artifact_name, timestamp)

        # Store formatted timestamp for downstream usage
        self.timestamp: str = timestamp


# =========================================================
# DataIngestionConfig: Defines config for data ingestion
# =========================================================
class DataIngestionConfig:
    """
    Configuration class for the data ingestion phase.
    Specifies paths for raw data, feature store, and training/testing datasets.
    """

    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        """
        Initialize all required paths for data ingestion.

        Args:
            training_pipeline_config (TrainingPipelineConfig): Base pipeline config.
        """
        # Root folder for all ingestion outputs (e.g., artifact/05_06_2025_10_34_20/data_ingestion/)
        self.data_ingestion_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.DATA_INGESTION_DIR_NAME
        )

        # Full path to store the feature store data (processed raw data)
        self.feature_store_file_path: str = os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_FEATURE_STORE_DIR,
            training_pipeline.FILE_NAME  # e.g., "data.csv"
        )

        # Path to store the ingested training dataset
        self.training_file_path: str = os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_INGESTED_DIR,
            training_pipeline.TRAIN_FILE_NAME
        )

        # Path to store the ingested testing dataset
        self.testing_file_path: str = os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_INGESTED_DIR,
            training_pipeline.TEST_FILE_NAME
        )

        # Split ratio for training and test datasets (e.g., 0.2 means 80% train, 20% test)
        self.train_test_split_ratio: float = training_pipeline.DATA_INGESTION_TRAIN_TEST_SPLIT_RATION

        # MongoDB collection name used for data ingestion
        self.collection_name: str = training_pipeline.DATA_INGESTION_COLLECTION_NAME

        # MongoDB database name
        self.database_name: str = training_pipeline.DATA_INGESTION_DATABASE_NAME


# ===================================================================================
# DataValidationConfig: Configuration class for validating raw, train, and test data
# ===================================================================================
class DataValidationConfig:
    """
    Configuration class for the data validation stage.
    Defines paths to valid/invalid train/test files and drift reports.
    """

    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        """
        Initialize validation configuration paths.

        Args:
            training_pipeline_config (TrainingPipelineConfig): Base pipeline config.
        """
        # Directory for all data validation artifacts (e.g., artifact/.../data_validation/)
        self.data_validation_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.DATA_VALIDATION_DIR_NAME
        )

        # Directory to store valid datasets (train and test)
        self.valid_data_dir: str = os.path.join(
            self.data_validation_dir,
            training_pipeline.DATA_VALIDATION_VALID_DIR
        )

        # Directory to store invalid datasets (train and test)
        self.invalid_data_dir: str = os.path.join(
            self.data_validation_dir,
            training_pipeline.DATA_VALIDATION_INVALID_DIR
        )

        # Path to store valid training dataset
        self.valid_train_file_path: str = os.path.join(
            self.valid_data_dir,
            training_pipeline.TRAIN_FILE_NAME
        )

        # Path to store valid testing dataset
        self.valid_test_file_path: str = os.path.join(
            self.valid_data_dir,
            training_pipeline.TEST_FILE_NAME
        )

        # Path to store invalid training dataset
        self.invalid_train_file_path: str = os.path.join(
            self.invalid_data_dir,
            training_pipeline.TRAIN_FILE_NAME
        )

        # Path to store invalid testing dataset
        self.invalid_test_file_path: str = os.path.join(
            self.invalid_data_dir,
            training_pipeline.TEST_FILE_NAME
        )

        # Full path to save the data drift report in YAML format
        self.drift_report_file_path: str = os.path.join(
            self.data_validation_dir,
            training_pipeline.DATA_VALIDATION_DRIFT_REPORT_DIR,
            training_pipeline.DATA_VALIDATION_DRIFT_REPORT_FILE_NAME
        )
# ========================================================================================
# DataTransformationConfig: Configuration class for transforming raw, train, and test data
# ========================================================================================

class DataTransformationConfig:
    """
    Configuration class for managing file paths and directory structure related to 
    the data transformation phase of the ML pipeline.

    Attributes:
        data_transformation_dir (str): Base directory for all data transformation artifacts.
        transformed_train_file_path (str): Path to the transformed training dataset (.npy format).
        transformed_test_file_path (str): Path to the transformed testing dataset (.npy format).
        transformed_object_file_path (str): Path to the serialized preprocessing object (e.g., imputer, scaler).
    """
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        # Directory where all transformation artifacts will be stored
        self.data_transformation_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.DATA_TRANSFORMATION_DIR_NAME
        )

        # Path for the transformed training data file (converted from .csv to .npy)
        self.transformed_train_file_path: str = os.path.join(
            self.data_transformation_dir,
            training_pipeline.DATA_TRANSFORMATION_TRANSFORMATED_DATA_DIR,
            training_pipeline.TRAIN_FILE_NAME.replace("csv", "npy")
        )

        # Path for the transformed testing data file (converted from .csv to .npy)
        self.transformed_test_file_path: str = os.path.join(
            self.data_transformation_dir,
            training_pipeline.DATA_TRANSFORMATION_TRANSFORMATED_DATA_DIR,
            training_pipeline.TEST_FILE_NAME.replace("csv", "npy")
        )

        # Path to save the preprocessing object used for transforming the data
        self.transformed_object_file_path: str = os.path.join(
            self.data_transformation_dir,
            training_pipeline.DATA_TRANSFORMATION_TRANSFORMATED_OBJECT_DIR,
            training_pipeline.PREPROCESSING_OBJECT_FILE_NAME
        )
