"""
Main pipeline runner script for the Network Security ML project.

This script sequentially executes the pipeline stages:
1. Data Ingestion
2. Data Validation
3. Data Transformation

Each stage uses its respective configuration and component class. Errors are logged and 
raised as custom exceptions.

Author: Your Name
Date: YYYY-MM-DD
"""

import sys
import numpy as np
from networksecurity.components.model_trainer import ModelTrainer
from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation
from networksecurity.components.data_transformation import DataTransformation
from networksecurity.components.model_evaluator import ModelEvaluator
from networksecurity.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    TrainingPipelineConfig,
    ModelTrainerConfig
)
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logger.logger import logger

if __name__ == "__main__":
    try:
        # ===============================
        # Step 1: Initialize Configuration
        # ===============================
        training_pipeline_config = TrainingPipelineConfig()

        # ====================
        # Step 2: Data Ingestion
        # ====================
        logger.info("Starting data ingestion...")
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config)
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        logger.info("Data ingestion completed.")
        print(data_ingestion_artifact)

        # ====================
        # Step 3: Data Validation
        # ====================
        logger.info("Starting data validation...")
        data_validation_config = DataValidationConfig(training_pipeline_config)
        data_validation = DataValidation(data_ingestion_artifact, data_validation_config)
        data_validation_artifact = data_validation.initiate_data_validation()
        logger.info("Data validation completed.")
        print(data_validation_artifact)

        # ======================
        # Step 4: Data Transformation
        # ======================
        logger.info("Starting data transformation...")
        data_transformation_config = DataTransformationConfig(training_pipeline_config)
        data_transformation = DataTransformation(data_validation_artifact, data_transformation_config)
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        logger.info("Data transformation completed.")
        print(data_transformation_artifact)

        # ======================
        # Step 5: Model Trainer
        # ======================
        logger.info("Starting Model Trainer...")
        model_trainer_config = ModelTrainerConfig(training_pipeline_config)
        model_trainer = ModelTrainer(model_trainer_config, data_transformation_artifact)
        model_trainer_artifact = model_trainer.initiate_model_trainer()
        logger.info("Model Trainer completed.")
        print(model_trainer_artifact)

        # ======================
        # Step 6: Model evaluator
        # ======================


        logger.info("Starting Model evaluator...")

        # Load transformed data arrays
        train_arr = np.load(data_transformation_artifact.transformed_train_file_path)
        test_arr = np.load(data_transformation_artifact.transformed_test_file_path)

        X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
        X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

        # Initialize evaluator and run evaluation
        model_evaluator = ModelEvaluator(data_transformation_artifact,
                                         model_path="Artifacts/06_09_2025_15_46_32/model_trainer/trained_model/model.pkl",  # or wherever your trained model is saved
                                         mode="load_and_evaluate"
                    )
        best_model, train_metric, test_metric = model_evaluator.evaluate(X_train, y_train, X_test, y_test)

        logger.info("Model evaluator completed.")
        print("Best model evaluation completed.")
        print("Train Accuracy:", train_metric.accuracy_score)
        print("Test Accuracy:", test_metric.accuracy_score)


    except Exception as e:
        logger.error("Pipeline execution failed.", exc_info=True)
        raise NetworkSecurityException(e, sys)

        