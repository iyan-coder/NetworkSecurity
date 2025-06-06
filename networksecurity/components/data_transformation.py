import sys
import os
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline

from networksecurity.constant.training_pipeline import TARGET_COLUMN
from networksecurity.constant.training_pipeline import DATA_TRANSFORMATION_IMPUTER_PARAMS
from networksecurity.entity.artifact_entity import DataTransformationArtifact, DataValidationArtifact
from networksecurity.entity.config_entity import DataTransformationConfig
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logger.logger import logger
from networksecurity.utils.main_utils.utils import save_numpy_array_data, save_object


class DataTransformation:
    def __init__(self, data_validation_artifact: DataValidationArtifact, data_tranformation_config: DataTransformationConfig):
        """
        Initialize the DataTransformation component with necessary artifacts and config.
        """
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_tranformation_config
        except Exception as e:
            logger.error("Error during DataTransformation initialization", exc_info=True)
            raise NetworkSecurityException(e, sys)

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        """
        Reads a CSV file into a pandas DataFrame.
        """
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            logger.error("Failed to read file", exc_info=True)
            raise NetworkSecurityException(e, sys)

    def get_data_transformer_object(self) -> Pipeline:
        """
        Creates and returns a data transformation pipeline using KNNImputer.
        """
        logger.info("Entered get_data_transformer_object method of Transformation class")
        try:
            # Initialize the imputer with params from config
            imputer = KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)
            logger.info(f"Initialized KNNImputer with {DATA_TRANSFORMATION_IMPUTER_PARAMS}")

            # Create a pipeline with the imputer (more steps can be added later)
            preprocessor = Pipeline([("imputer", imputer)])
            return preprocessor
        except Exception as e:
            logger.error("Failed to get data transformer object", exc_info=True)
            raise NetworkSecurityException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Runs the full transformation process: imputation + saving transformed data.
        """
        logger.info("Entered initiate_data_transformation method of DataTransformation class")
        try:
            # Load train and test validated data
            train_df = self.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = self.read_data(self.data_validation_artifact.valid_test_file_path)

            # Split input features and target from train
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN].replace(-1, 0)

            # Split input features and target from test
            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN].replace(-1, 0)

            # Initialize and fit transformer
            preprocessor = self.get_data_transformer_object()
            preprocessed_train_features = preprocessor.fit_transform(input_feature_train_df)
            preprocessed_test_features = preprocessor.transform(input_feature_test_df)

            # Combine transformed features with target for both sets
            train_arr = np.c_[preprocessed_train_features, np.array(target_feature_train_df)]
            test_arr = np.c_[preprocessed_test_features, np.array(target_feature_test_df)]

            # Save transformed arrays as .npy files
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)

            # Save the fitted transformation pipeline (for future inference use)
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)

            # Prepare and return artifact containing paths to transformed data
            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )

            return data_transformation_artifact

        except Exception as e:
            logger.error("Data Transformation process failed.", exc_info=True)
            raise NetworkSecurityException(e, sys)

        
