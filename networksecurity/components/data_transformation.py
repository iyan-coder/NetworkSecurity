# Importing necessary Python libraries
import os  # Helps with file and folder operations
import sys  # Helps us capture error messages and exceptions
import pandas as pd  # To work with data in table (DataFrame) format

# Importing machine learning tools
from sklearn.impute import KNNImputer  # This helps us fill in missing values using similar data
from sklearn.pipeline import Pipeline  # Helps us build a step-by-step data processing setup
from sklearn.compose import ColumnTransformer  # Lets us apply processing only to specific columns

# Importing project-specific constants and classes
from networksecurity.constant.training_pipeline import TARGET_COLUMN, DATA_TRANSFORMATION_IMPUTER_PARAMS
from networksecurity.entity.artifact_entity import DataTransformationArtifact, DataValidationArtifact
from networksecurity.entity.config_entity import DataTransformationConfig
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logger.logger import logger
from networksecurity.utils.main_utils.utils import save_object  # To save files (like transformers or lists)

# Define the DataTransformation class
class DataTransformation:
    def __init__(self, data_validation_artifact: DataValidationArtifact, data_transformation_config: DataTransformationConfig):
        """
        This sets up the DataTransformation class.
        It needs the output from data validation and some settings for transformation.
        """
        try:
            self.data_validation_artifact = data_validation_artifact  # Validated data info
            self.data_transformation_config = data_transformation_config  # Settings for transformation
        except Exception as e:
            logger.error("Error during DataTransformation initialization", exc_info=True)
            raise NetworkSecurityException(e, sys)

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        """
        Reads a CSV file and returns it as a pandas DataFrame (table format).
        """
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            logger.error("Failed to read CSV file", exc_info=True)
            raise NetworkSecurityException(e, sys)

    def get_data_transformer_object(self, feature_columns: list) -> ColumnTransformer:
        """
        Makes a machine that fills missing values for each column using KNN (looking at nearby values).
        """
        try:
            logger.info("Creating ColumnTransformer with KNNImputer...")

            # Create a step: imputer fills missing values in features
            num_pipeline = Pipeline(steps=[
                ("imputer", KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS))  # Using settings from config
            ])

            # Apply that step to all given feature columns
            preprocessor = ColumnTransformer(transformers=[
                ("num_pipeline", num_pipeline, feature_columns)  # Apply num_pipeline to selected columns
            ])

            return preprocessor  # Return this setup
        except Exception as e:
            logger.error("Failed to create data transformer object", exc_info=True)
            raise NetworkSecurityException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        This is the main function that does all the work:
        1. Reads validated data.
        2. Separates features and labels.
        3. Fills missing values.
        4. Saves transformed data and tools.
        5. Returns paths to these files.
        """
        logger.info("Starting data transformation process...")

        try:
            # Step 1: Read cleaned train and test CSV files into DataFrames
            train_df = self.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = self.read_data(self.data_validation_artifact.valid_test_file_path)

            # Step 2: Separate the input features and the output (target)
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN])  # Remove label column
            target_feature_train_df = train_df[TARGET_COLUMN].replace(-1, 0)  # Convert -1 to 0

            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN])  # Same for test set
            target_feature_test_df = test_df[TARGET_COLUMN].replace(-1, 0)

            # Step 3: Save which columns are features so we know the order later
            feature_columns = input_feature_train_df.columns.tolist()
            os.makedirs(os.path.dirname(self.data_transformation_config.feature_columns_file_path), exist_ok=True)
            save_object(self.data_transformation_config.feature_columns_file_path, feature_columns)

            # Step 4: Create and apply the transformer (imputer)
            preprocessor = self.get_data_transformer_object(feature_columns)
            transformed_train = preprocessor.fit_transform(input_feature_train_df)  # Fit and apply on training data
            transformed_test = preprocessor.transform(input_feature_test_df)  # Just apply on test data

            # Step 5: Make DataFrames again from transformed numpy arrays
            transformed_train_df = pd.DataFrame(transformed_train, columns=feature_columns)
            transformed_train_df[TARGET_COLUMN] = target_feature_train_df.values  # Add back the target column

            transformed_test_df = pd.DataFrame(transformed_test, columns=feature_columns)
            transformed_test_df[TARGET_COLUMN] = target_feature_test_df.values

            # Step 6: Make sure folders for saving results exist
            os.makedirs(os.path.dirname(self.data_transformation_config.transformed_train_file_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.data_transformation_config.transformed_test_file_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.data_transformation_config.transformed_object_file_path), exist_ok=True)

            # Step 7: Save the final transformed train and test datasets as CSV files
            transformed_train_df.to_csv(self.data_transformation_config.transformed_train_file_path, index=False)
            transformed_test_df.to_csv(self.data_transformation_config.transformed_test_file_path, index=False)

            # Step 8: Save the preprocessor so we can use it later on new data
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)

            logger.info("Data transformation completed successfully.")

            # Step 9: Return an object that tells where everything is saved
            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                feature_columns_file_path=self.data_transformation_config.feature_columns_file_path
            )

        except Exception as e:
            logger.error("Data transformation process failed.", exc_info=True)
            raise NetworkSecurityException(e, sys)
