import os
import sys
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp

from networksecurity.entity.artifact_entity import DataIngetionArtifact, DataValidationArtifact
from networksecurity.entity.config_entity import DataValidationConfig
from networksecurity.logger.logger import logger
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.constant.training_pipeline import SCHEMA_FILE_PATH
from networksecurity.utils.main_utils.utils import read_yaml_file, write_yaml_file


class DataValidation:
    """
    This class handles all aspects of data validation including schema checks,
    numerical column existence, and data drift detection.
    """

    def __init__(
        self,
        data_ingestion_artifact: DataIngetionArtifact,
        data_validation_config: DataValidationConfig
    ):
        """
        Initializes the DataValidation object by loading schema configuration and artifacts.

        Args:
            data_ingestion_artifact (DataIngetionArtifact): Contains paths to training and testing data.
            data_validation_config (DataValidationConfig): Contains configurations for data validation.
        """
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            logger.error("Error during DataValidation initialization", exc_info=True)
            raise NetworkSecurityException(e, sys)

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        """
        Reads CSV file into a pandas DataFrame.

        Args:
            file_path (str): Path to the CSV file.

        Returns:
            pd.DataFrame: Loaded DataFrame.
        """
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            logger.error("Failed to read file", exc_info=True)
            raise NetworkSecurityException(e, sys)

    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        """
        Validates that the number of columns in the DataFrame matches the schema.

        Args:
            dataframe (pd.DataFrame): DataFrame to validate.

        Returns:
            bool: True if column count matches, False otherwise.
        """
        try:
            expected_columns = self._schema_config["columns"]
            number_of_columns = len(expected_columns)
            logger.info(f"Expected number of columns: {number_of_columns}")
            logger.info(f"Actual number of columns: {len(dataframe.columns)}")

            return len(dataframe.columns) == number_of_columns

        except Exception as e:
            logger.error("Failed to validate number of columns", exc_info=True)
            raise NetworkSecurityException(e, sys)

    def detect_dataset_drift(self, base_df: pd.DataFrame, current_df: pd.DataFrame, threshold: float = 0.05) -> bool:
        """
        Detects dataset drift between base and current datasets using the KS test.

        Args:
            base_df (pd.DataFrame): Reference dataset (typically training).
            current_df (pd.DataFrame): Current dataset (typically testing).
            threshold (float): P-value threshold for detecting drift.

        Returns:
            bool: True if no drift detected, False otherwise.
        """
        try:
            # Initialize drift status as True, assuming no drift unless detected
            status = True

            # Create an empty dictionary to store drift test results per column
            report = {}

            # Ensure both dataframes (train/test) have the same column structure
            assert base_df.columns.equals(current_df.columns), "Train and test columns do not match"

            # Iterate through each column for drift detection
            for column in base_df.columns:

                # Drop any missing values to avoid errors during statistical testing
                d1 = base_df[column].dropna()
                d2 = current_df[column].dropna()

                # Only apply KS test to numerical columns
                if pd.api.types.is_numeric_dtype(d1):
                
                    # Perform the two-sample Kolmogorov–Smirnov test to compare distributions
                    ks_test_result = ks_2samp(d1, d2)

                    # Extract the p-value from the test result
                    p_value = ks_test_result.pvalue

                    # If p-value is less than the threshold (default 0.05), consider it drift
                    drift_detected = p_value < threshold

                    # Store results for this column in the drift report
                report[column] = {
                    "p_value": float(p_value),
                    "drift_status": bool(drift_detected)   # 👈 cast to native bool
                    }


                # If any column has drift, set status to False
                if drift_detected:
                    status = False

            # Get the path where the drift report should be saved
            drift_report_file_path = self.data_validation_config.drift_report_file_path

            # Ensure the directory for the report exists
            os.makedirs(os.path.dirname(drift_report_file_path), exist_ok=True)

            # Write the report dictionary to a YAML file
            write_yaml_file(file_path=drift_report_file_path, content=report)

            # Return the overall validation status (True = no drift, False = drift detected)
            return status
        
        except Exception as e:
            logger.error("Failed to detect dataset drift")
            # Wrap and raise any exceptions with custom error handling
            raise NetworkSecurityException(e, sys)


    def is_numerical_columns_exist(self, dataframe: pd.DataFrame) -> bool:
        """
        Checks if the dataset contains any numerical columns.

        Args:
            dataframe (pd.DataFrame): The DataFrame to check.

        Returns:
            bool: True if numerical columns exist, False otherwise.
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
        Orchestrates the full data validation pipeline.

        Returns:
            DataValidationArtifact: Artifact object summarizing the validation results.
        """
        try:
            train_file_path = self.data_ingestion_artifact.trained_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            # Load data
            train_dataframe = DataValidation.read_data(train_file_path)
            test_dataframe = DataValidation.read_data(test_file_path)

            # Schema validation
            if not self.validate_number_of_columns(train_dataframe):
                raise NetworkSecurityException("Train dataframe does not contain all expected columns.", sys)
            if not self.validate_number_of_columns(test_dataframe):
                raise NetworkSecurityException("Test dataframe does not contain all expected columns.", sys)

            # Numerical column check
            if not self.is_numerical_columns_exist(train_dataframe):
                raise NetworkSecurityException("Train dataframe lacks numerical columns.", sys)
            if not self.is_numerical_columns_exist(test_dataframe):
                raise NetworkSecurityException("Test dataframe lacks numerical columns.", sys)

            # Drift detection
            validation_status = self.detect_dataset_drift(train_dataframe, test_dataframe)

            # Save validated data
            os.makedirs(os.path.dirname(self.data_validation_config.valid_train_file_path), exist_ok=True)
            train_dataframe.to_csv(self.data_validation_config.valid_train_file_path, index=True, header=True)
            test_dataframe.to_csv(self.data_validation_config.valid_test_file_path, index=True, header=True)

            # Create and return validation artifact
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
