import os
import sys
import numpy as np
import pandas as pd

"""
Module: training_pipeline.py

This module defines constants used throughout the Machine Learning pipeline
for a Network Security project. Centralizing constants improves maintainability, 
readability, and scalability. These values configure file names, paths, 
dataset handling, and database connectivity.
"""

# ============================================
# General Pipeline Configuration Constants
# ============================================

# Name of the entire ML pipeline (used in logs, directory naming, etc.)
PIPELINE_NAME: str = "NetworkSecurity"

# Root directory to store all pipeline artifacts (e.g., ingested data, models, logs)
ARTIFACT_DIR: str = "Artifacts"

# Name of the original CSV file expected as input
FILE_NAME: str = "phisingData.csv"

# File names to be generated after train-test split
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"

# Target column name in the dataset for supervised learning
TARGET_COLUMN: str = "Result"

# Path to the schema definition file for validating structure of input data
SCHEMA_FILE_PATH: str = os.path.join("data_schema", "schema.yaml")


# ============================================
# Data Ingestion Related Constants
# ============================================

# MongoDB collection from which data will be pulled
DATA_INGESTION_COLLECTION_NAME: str = "NetworkData"

# MongoDB database name
DATA_INGESTION_DATABASE_NAME: str = "Iyan-coder"

# Subdirectory inside artifacts folder for storing data ingestion outputs
DATA_INGESTION_DIR_NAME: str = "data_ingestion"

# Directory to store raw processed data (after minor cleaning but before splitting)
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"

# Directory to store the train and test split datasets
DATA_INGESTION_INGESTED_DIR: str = "ingested"

# Ratio to split the dataset into train and test sets (0.2 = 20% test data)
DATA_INGESTION_TRAIN_TEST_SPLIT_RATION: float = 0.2


# ============================================
# Data Validation Related Constants
# ============================================

# Root directory for data validation step
DATA_VALIDATION_DIR_NAME: str = "data_validation"

# Subdirectory for storing successfully validated datasets
DATA_VALIDATION_VALID_DIR: str = "validated"

# Subdirectory for storing failed/invalid datasets
DATA_VALIDATION_INVALID_DIR: str = "invalid"

# Subdirectory to store data drift reports
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"

# File name of the generated data drift report (YAML format)
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"
