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
    DataValidation handles validation of ingested data against a schema and checks
    for data drift between training and testing datasets.

    Key Responsibilities:
    - Validate the number of columns matches the schema.
    - Confirm numerical columns exist.
    - Detect distributional changes (data drift) using Kolmogorov-Smirnov test.
    - Save validated datasets and generate drift reports.
    """

    def __init__(
        self,
        data_ingestion_artifact: DataIngetionArtifact,
        data_validation_config: DataValidationConfig
    ):
        """
        Initializes the DataValidation instance by loading:
        - Data ingestion artifact containing file paths to datasets.
        - Validation configuration including output file paths.
        - Schema configuration from a YAML file.

        Args:
            data_ingestion_artifact (DataIngetionArtifact): Paths for train/test CSV files.
            data_validation_config (DataValidationConfig): Paths for validation outputs.
        """
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            # Load expected schema (column names and types) from YAML file
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            logger.error("Error during DataValidation initialization", exc_info=True)
            raise NetworkSecurityException(e, sys)

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        """
        Reads a CSV file into a pandas DataFrame.

        Args:
            file_path (str): Location of CSV file.

        Returns:
            pd.DataFrame: Loaded dataset.

        Raises:
            NetworkSecurityException: If file reading fails.
        """
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            logger.error("Failed to read file", exc_info=True)
            raise NetworkSecurityException(e, sys)

    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        """
        Checks if the dataframe contains the exact number of columns as defined in the schema.

        Args:
            dataframe (pd.DataFrame): DataFrame to validate.

        Returns:
            bool: True if columns count matches, False otherwise.
        """
        try:
            expected_columns = self._schema_config["columns"]
            expected_num_cols = len(expected_columns)
            actual_num_cols = len(dataframe.columns)

            logger.info(f"Expected columns count: {expected_num_cols}")
            logger.info(f"Actual columns count: {actual_num_cols}")

            return actual_num_cols == expected_num_cols

        except Exception as e:
            logger.error("Failed to validate number of columns", exc_info=True)
            raise NetworkSecurityException(e, sys)

    def detect_dataset_drift(self, base_df: pd.DataFrame, current_df: pd.DataFrame, threshold: float = 0.05) -> bool:
        """
        Detects data drift by comparing distributions of numerical columns between
        a base dataset (training) and a current dataset (testing) using KS test.

        Args:
            base_df (pd.DataFrame): Reference dataset (usually training).
            current_df (pd.DataFrame): Dataset to compare against base.
            threshold (float): Significance level for p-value to flag drift (default 0.05).

        Returns:
            bool: True if no drift detected (all p-values > threshold), False if drift found.

        Raises:
            NetworkSecurityException: If drift detection process fails.
        """
        try:
            # Assume no drift initially
            validation_status = True
            drift_report = {}

            # Check columns are identical before comparison
            assert base_df.columns.equals(current_df.columns), "Train and test columns do not match"

            for column in base_df.columns:
                # Drop missing values to avoid errors
                base_col = base_df[column].dropna()
                curr_col = current_df[column].dropna()

                # Only test numeric columns (skip categorical/string)
                if pd.api.types.is_numeric_dtype(base_col):
                    # KS test compares distribution similarity
                    ks_result = ks_2samp(base_col, curr_col)
                    p_value = ks_result.pvalue

                    # Drift if p-value < threshold (distributions differ significantly)
                    drift_detected = p_value < threshold

                    drift_report[column] = {
                        "p_value": float(p_value),
                        "drift_status": bool(drift_detected)
                    }

                    if drift_detected:
                        validation_status = False

            # Save drift report YAML file for monitoring
            drift_report_path = self.data_validation_config.drift_report_file_path
            os.makedirs(os.path.dirname(drift_report_path), exist_ok=True)
            write_yaml_file(file_path=drift_report_path, content=drift_report)

            return validation_status

        except Exception as e:
            logger.error("Failed to detect dataset drift", exc_info=True)
            raise NetworkSecurityException(e, sys)

    def is_numerical_columns_exist(self, dataframe: pd.DataFrame) -> bool:
        """
        Validates the existence of at least one numerical column in the dataset.

        Args:
            dataframe (pd.DataFrame): Dataset to check.

        Returns:
            bool: True if numerical columns are present, False otherwise.
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
        Executes the complete data validation process:
        - Reads train/test data.
        - Validates schema.
        - Checks for numerical columns.
        - Detects data drift.
        - Saves validated datasets.
        - Returns an artifact summarizing validation status and file paths.

        Returns:
            DataValidationArtifact: Contains validation status and file paths for
            valid/invalid datasets and drift report.

        Raises:
            NetworkSecurityException: If validation fails at any step.
        """
        try:
            train_file_path = self.data_ingestion_artifact.trained_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            # Load datasets
            train_df = DataValidation.read_data(train_file_path)
            test_df = DataValidation.read_data(test_file_path)

            # Schema validation
            if not self.validate_number_of_columns(train_df):
                raise NetworkSecurityException("Train dataframe column count mismatch.", sys)
            if not self.validate_number_of_columns(test_df):
                raise NetworkSecurityException("Test dataframe column count mismatch.", sys)

            # Check numerical columns existence
            if not self.is_numerical_columns_exist(train_df):
                raise NetworkSecurityException("Train dataframe lacks numerical columns.", sys)
            if not self.is_numerical_columns_exist(test_df):
                raise NetworkSecurityException("Test dataframe lacks numerical columns.", sys)

            # Detect data drift between train and test sets
            validation_status = self.detect_dataset_drift(train_df, test_df)

            # Save clean validated data for downstream tasks
            os.makedirs(os.path.dirname(self.data_validation_config.valid_train_file_path), exist_ok=True)
            train_df.to_csv(self.data_validation_config.valid_train_file_path, index=False, header=True)
            test_df.to_csv(self.data_validation_config.valid_test_file_path, index=False, header=True)

            # Return artifact containing info about validation results
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
