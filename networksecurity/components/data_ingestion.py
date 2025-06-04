from networksecurity.exception.exception import NetworkSecurityException  # Custom exception for standardized error handling
from networksecurity.logger.logger import logger  # Logger for tracking pipeline execution

from networksecurity.entity.config_entity import DataIngestionConfig  # Configuration data class
from networksecurity.entity.artifact_entity import DataIngetionArtifact  # Artifact class for outputs

import os  # OS functions for file paths and directories
import sys  # For getting system-specific info in exception
import pandas as pd  # DataFrame for data manipulation
import numpy as np  # For handling missing values
import pymongo  # MongoDB connector
from sklearn.model_selection import train_test_split  # Train-test splitting

from dotenv import load_dotenv  # Load environment variables from .env file
load_dotenv()  # Load .env file into environment variables

MONGO_DB_URL = os.getenv("MONGO_DB_URL")  # Read MongoDB URL from .env

class DataIngestion:
    """
    Handles the data ingestion process:
    - Reads data from MongoDB.
    - Saves raw data to feature store.
    - Splits data into train and test sets.
    """

    def __init__(self, data_ingestion_config: DataIngestionConfig):
        """
        Constructor: Initializes DataIngestion with configuration object
        """
        try:
            self.data_ingestion_config = data_ingestion_config  # Save config for use in methods
            logger.info("DataIngestion initialized with config.")
        except Exception as e:
            logger.error("Error during DataIngestion initialization", exc_info=True)
            raise NetworkSecurityException(e, sys)

    def export_collection_as_dataframe(self) -> pd.DataFrame:
        """
        Connects to MongoDB, fetches collection data into a DataFrame.
        Cleans it by removing _id and replacing 'na' with NaN.
        """
        try:
            logger.info("Connecting to MongoDB for data export.")
            database_name = self.data_ingestion_config.database_name  # Get DB name from config
            collection_name = self.data_ingestion_config.collection_name  # Get collection name from config

            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)  # Connect to MongoDB
            logger.info(f"Connected to MongoDB: {MONGO_DB_URL}")

            collection = self.mongo_client[database_name][collection_name]  # Access specific collection
            logger.info(f"Accessing collection: {database_name}.{collection_name}")

            df = pd.DataFrame(list(collection.find()))  # Convert collection to DataFrame
            logger.info(f"Fetched {df.shape[0]} records from MongoDB collection: {collection_name}")

            if "_id" in df.columns:
                df.drop(columns=["_id"], inplace=True)  # Remove MongoDB default ID column
                logger.info("Dropped _id column from DataFrame")

            df.replace({"na": np.nan}, inplace=True)  # Replace 'na' with np.nan for standard missing values
            logger.info(f"Final dataframe shape after cleaning: {df.shape}")

            if df.empty:
                logger.warning("Warning: Exported DataFrame is empty.")

            return df

        except Exception as e:
            logger.error("Failed to export data from MongoDB", exc_info=True)
            raise NetworkSecurityException(e, sys)

    def export_data_into_feature_store(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Saves the cleaned DataFrame to a CSV file in the feature store path.
        """
        try:
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path  # Get output path
            dir_path = os.path.dirname(feature_store_file_path)  # Get directory
            os.makedirs(dir_path, exist_ok=True)  # Create directories if they don't exist
            logger.info(f"Saving data to feature store path: {feature_store_file_path}")

            dataframe.to_csv(feature_store_file_path, index=False, header=True)  # Save as CSV
            logger.info(f"Data exported to feature store at {feature_store_file_path}")

            return dataframe

        except Exception as e:
            logger.error("Failed to save data to feature store", exc_info=True)
            raise NetworkSecurityException(e, sys)

    def split_data_as_train_test(self, dataframe: pd.DataFrame):
        """
        Splits the data into training and testing datasets.
        Saves them to the artifact paths defined in config.
        """
        try:
            if dataframe.empty:
                raise ValueError("DataFrame is empty. Cannot perform train-test split.")

            logger.info("Splitting data into train and test sets.")
            train_set, test_set = train_test_split(
                dataframe,
                test_size=self.data_ingestion_config.train_test_split_ratio  # Get ratio from config
            )

            logger.info(f"Train set shape: {train_set.shape}, Test set shape: {test_set.shape}")

            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)  # Ensure target directory exists
            logger.info(f"Saving train and test files to: {dir_path}")

            train_set.to_csv(self.data_ingestion_config.training_file_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path, index=False, header=True)

            logger.info(f"Train data saved to: {self.data_ingestion_config.training_file_path}")
            logger.info(f"Test data saved to: {self.data_ingestion_config.testing_file_path}")

        except Exception as e:
            logger.error("Error occurred during train-test split", exc_info=True)
            raise NetworkSecurityException(e, sys)

    def initiate_data_ingestion(self) -> DataIngetionArtifact:
        """
        Orchestrates the full data ingestion pipeline.
        - Reads from MongoDB
        - Cleans and stores raw data
        - Splits data
        - Returns artifact containing file paths
        """
        try:
            logger.info("Starting data ingestion process.")

            dataframe = self.export_collection_as_dataframe()  # Step 1: Load data from MongoDB

            if dataframe is None or dataframe.empty:
                raise ValueError("Exported dataframe is None or empty.")

            dataframe = self.export_data_into_feature_store(dataframe)  # Step 2: Save raw data
            self.split_data_as_train_test(dataframe)  # Step 3: Split and save

            data_ingestion_artifact = DataIngetionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path
            )

            logger.info("Data ingestion completed successfully.")
            return data_ingestion_artifact  # Return result object

        except Exception as e:
            logger.error("Data ingestion process failed.", exc_info=True)
            raise NetworkSecurityException(e, sys)
