"""
Main pipeline runner script for the Network Security ML project.

This script sequentially executes the pipeline stages:
1. Data Ingestion
2. Data Validation
3. Data Transformation
4. Model Training
5. Model Evaluation

Each stage uses its respective configuration and component class.
Errors are logged and raised as custom exceptions.

Author: Your Name
Date: YYYY-MM-DD
"""

# Import necessary libraries
import os  # For handling file paths
import sys  # For handling system-level errors
import numpy as np  # For numerical operations
import pandas as pd  # For working with tabular data

# Import pipeline components
from networksecurity.components.model_trainer import ModelTrainer
from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation
from networksecurity.components.data_transformation import DataTransformation
from networksecurity.components.model_evaluator import ModelEvaluator

# Import configuration entities
from networksecurity.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    TrainingPipelineConfig,
    ModelTrainerConfig
)

# Custom error and logging handling
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logger.logger import logger

# Utility functions
from networksecurity.utils.main_utils.utils import get_latest_artifact_dir, load_object
from networksecurity.constant.training_pipeline import TARGET_COLUMN  # The name of the target/label column


if __name__ == "__main__":
    try:
        # ===============================
        # Step 1: Initialize Configuration
        # ===============================
        # Create the main training pipeline configuration
        training_pipeline_config = TrainingPipelineConfig()

        # ====================
        # Step 2: Data Ingestion
        # ====================
        logger.info("Starting data ingestion...")
        # Create and run the data ingestion process
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config)
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        logger.info("Data ingestion completed.")
        print(data_ingestion_artifact)  # Show what files/paths were created

        # ====================
        # Step 3: Data Validation
        # ====================
        logger.info("Starting data validation...")
        # Create and run the data validation process
        data_validation_config = DataValidationConfig(training_pipeline_config)
        data_validation = DataValidation(data_ingestion_artifact, data_validation_config)
        data_validation_artifact = data_validation.initiate_data_validation()
        logger.info("Data validation completed.")
        print(data_validation_artifact)

        # ======================
        # Step 4: Data Transformation
        # ======================
        logger.info("Starting data transformation...")
        # Create and run the data transformation process
        data_transformation_config = DataTransformationConfig(training_pipeline_config)
        data_transformation = DataTransformation(data_validation_artifact, data_transformation_config)
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        logger.info("Data transformation completed.")
        print(data_transformation_artifact)

        # ======================
        # Step 5: Model Trainer
        # ======================
        logger.info("Starting Model Trainer...")
        # Create and run model training
        model_trainer_config = ModelTrainerConfig(training_pipeline_config)
        model_trainer = ModelTrainer(model_trainer_config, data_transformation_artifact)
        model_trainer_artifact = model_trainer.initiate_model_trainer()
        logger.info("Model Trainer completed.")
        print(model_trainer_artifact)

        # ======================
        # Step 6: Model Evaluator
        # ======================
        logger.info("Starting Model evaluator...")

        # Load transformed train and test datasets
        train_df = pd.read_csv(data_transformation_artifact.transformed_train_file_path)
        test_df = pd.read_csv(data_transformation_artifact.transformed_test_file_path)

        # Load feature columns (used for selecting X)
        feature_columns = load_object(data_transformation_artifact.feature_columns_file_path)

        # Split into features (X) and target (y)
        X_train = train_df[feature_columns]
        y_train = train_df[TARGET_COLUMN]
        X_test = test_df[feature_columns]
        y_test = test_df[TARGET_COLUMN]

        # Locate the latest trained model file
        latest_artifact_dir = get_latest_artifact_dir()
        model_path = os.path.join(latest_artifact_dir, "model_trainer", "trained_model", "model.pkl")

        # Create evaluator and run evaluation
        model_evaluator = ModelEvaluator(
            data_transformation_artifact=data_transformation_artifact,
            model_path=model_path,
            mode="load_and_evaluate"
        )

        # Get best model and its performance
        best_model, train_metric, test_metric = model_evaluator.evaluate(X_train, y_train, X_test, y_test)

        logger.info("Model evaluator completed.")
        print("Best model evaluation completed.")
        print("Train Accuracy:", train_metric.accuracy_score)
        print("Test Accuracy:", test_metric.accuracy_score)

    except Exception as e:
        # Catch and log any error that happens during the pipeline
        logger.error("Pipeline execution failed.", exc_info=True)
        raise NetworkSecurityException(e, sys)
