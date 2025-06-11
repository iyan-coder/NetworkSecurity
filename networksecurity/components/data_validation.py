# Basic imports to handle files, system errors, and data tables
import os  # Helps us work with file paths and folders
import sys  # Helps with system-specific error messages
import pandas as pd  # To load and work with data tables (DataFrames)
import numpy as np  # Helpful for working with numbers
from scipy.stats import ks_2samp  # Used for checking if two sets of numbers are similar (drift check)

# Project-specific imports (used in this ML pipeline)
from networksecurity.entity.artifact_entity import DataIngetionArtifact, DataValidationArtifact
from networksecurity.entity.config_entity import DataValidationConfig
from networksecurity.logger.logger import logger  # Used to log what’s happening (like print but professional)
from networksecurity.exception.exception import NetworkSecurityException  # Custom error handler
from networksecurity.constant.training_pipeline import SCHEMA_FILE_PATH  # File path where schema (rules) is saved
from networksecurity.utils.main_utils.utils import read_yaml_file, write_yaml_file  # To read and write YAML files

# DataValidation class handles checking and cleaning the data
class DataValidation:
    """
    This class checks:
    - if data follows rules (like column count and types),
    - if important columns exist (like numbers),
    - if the training and testing data have changed (data drift).
    """

    def __init__(self, data_ingestion_artifact: DataIngetionArtifact, data_validation_config: DataValidationConfig):
        """
        This sets up the class by loading:
        - Ingested data paths
        - Settings for validation
        - Schema (rules about data columns) from a YAML file
        """
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)  # Load schema that defines valid column structure
        except Exception as e:
            logger.error("Error during DataValidation initialization", exc_info=True)
            raise NetworkSecurityException(e, sys)

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        """
        This reads a CSV file and returns it as a DataFrame (table of data).
        """
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            logger.error("Failed to read file", exc_info=True)
            raise NetworkSecurityException(e, sys)

    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        """
        Checks if the table has the correct number of columns according to the schema.
        """
        try:
            expected_columns = self._schema_config["columns"]  # Get columns from the YAML schema
            expected_num_cols = len(expected_columns)  # Count how many columns are expected
            actual_num_cols = len(dataframe.columns)  # Count how many columns the data actually has

            logger.info(f"Expected columns count: {expected_num_cols}")
            logger.info(f"Actual columns count: {actual_num_cols}")

            return actual_num_cols == expected_num_cols  # True if matches, otherwise False
        except Exception as e:
            logger.error("Failed to validate number of columns", exc_info=True)
            raise NetworkSecurityException(e, sys)

    def detect_dataset_drift(self, base_df: pd.DataFrame, current_df: pd.DataFrame, threshold: float = 0.05) -> bool:
        """
        Compares training and test datasets to check if their distributions have changed (drift).
        Uses KS Test for this.
        """
        try:
            validation_status = True  # Assume everything is okay at first
            drift_report = {}  # Dictionary to store results for each column

            # Make sure both tables have the same columns
            assert base_df.columns.equals(current_df.columns), "Train and test columns do not match"

            for column in base_df.columns:
                base_col = base_df[column].dropna()  # Remove missing values
                curr_col = current_df[column].dropna()

                # Only do drift test if it's a number column
                if pd.api.types.is_numeric_dtype(base_col):
                    ks_result = ks_2samp(base_col, curr_col)  # Run KS test
                    p_value = ks_result.pvalue

                    drift_detected = p_value < threshold  # If p-value is small, drift is detected

                    drift_report[column] = {
                        "p_value": float(p_value),
                        "drift_status": bool(drift_detected)
                    }

                    # If any column has drift, set status to False
                    if drift_detected:
                        validation_status = False

            # Save the drift report to a YAML file
            drift_report_path = self.data_validation_config.drift_report_file_path
            os.makedirs(os.path.dirname(drift_report_path), exist_ok=True)
            write_yaml_file(file_path=drift_report_path, content=drift_report)

            return validation_status  # Return True if no drift, False if drift found

        except Exception as e:
            logger.error("Failed to detect dataset drift", exc_info=True)
            raise NetworkSecurityException(e, sys)

    def is_numerical_columns_exist(self, dataframe: pd.DataFrame) -> bool:
        """
        Checks if the DataFrame has any numerical columns.
        These are important for ML models.
        """
        numerical_cols = dataframe.select_dtypes(include=['number']).columns

        if len(numerical_cols) > 0:
            logger.info(f"Numerical columns found: {list(numerical_cols)}")
            return True
        else:
            logger.warning("No numerical columns found.")
            return False

    def initiate_data_validation(self) -> DataValidationArtifact:
        """
        This runs the whole validation process:
        - Load data
        - Check schema (column match)
        - Check for number columns
        - Detect drift
        - Save validated files
        - Return summary (artifact)
        """
        try:
            # Step 1: Get file paths
            train_file_path = self.data_ingestion_artifact.trained_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            # Step 2: Read both train and test datasets
            train_df = DataValidation.read_data(train_file_path)
            test_df = DataValidation.read_data(test_file_path)

            # Step 3: Validate column counts
            if not self.validate_number_of_columns(train_df):
                raise NetworkSecurityException("Train dataframe column count mismatch.", sys)
            if not self.validate_number_of_columns(test_df):
                raise NetworkSecurityException("Test dataframe column count mismatch.", sys)

            # Step 4: Make sure we have number columns
            if not self.is_numerical_columns_exist(train_df):
                raise NetworkSecurityException("Train dataframe lacks numerical columns.", sys)
            if not self.is_numerical_columns_exist(test_df):
                raise NetworkSecurityException("Test dataframe lacks numerical columns.", sys)

            # Step 5: Check if train and test have changed (data drift)
            validation_status = self.detect_dataset_drift(train_df, test_df)

            # Step 6: Save validated train/test data
            os.makedirs(os.path.dirname(self.data_validation_config.valid_train_file_path), exist_ok=True)
            train_df.to_csv(self.data_validation_config.valid_train_file_path, index=False, header=True)
            test_df.to_csv(self.data_validation_config.valid_test_file_path, index=False, header=True)

            # Step 7: Return validation summary
            return DataValidationArtifact(
                validation_status=validation_status,
                valid_train_file_path=self.data_validation_config.valid_train_file_path,
                valid_test_file_path=self.data_validation_config.valid_test_file_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                drift_report_file_path=self.data_validation_config.drift_report_file_path
            )

        except Exception as e:
            logger.error("Data validation process failed.", exc_info=True)
            raise NetworkSecurityException(e, sys)
