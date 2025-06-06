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
from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation
from networksecurity.components.data_transformation import DataTransformation
from networksecurity.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    TrainingPipelineConfig
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

    except Exception as e:
        logger.error("Pipeline execution failed.", exc_info=True)
        raise NetworkSecurityException(e, sys)

        