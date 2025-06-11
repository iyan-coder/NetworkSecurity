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

SAVED_MODEL_DIR = os.path.join("saved_model")

MODEL_FILE_NAME = "model.pkl"

MODEL_CONFIG_FILE_NAME = os.path.join("config_dir", "model_config.yaml")

# Feature Name
FEATURE_NAME: str = "feature_columns.pkl"


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


# ============================
# Data Transformation Imputer
# ============================

# Directory name for storing all data transformation-related artifacts
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"

# Subdirectory to store transformed data (typically after preprocessing)
DATA_TRANSFORMATION_TRANSFORMATED_DATA_DIR: str = "transformed"

# Subdirectory to store serialized transformation objects like scalers or imputers
DATA_TRANSFORMATION_TRANSFORMATED_OBJECT_DIR: str = "transformed_object"



# =====================================
# KNN imputer parameters for missing values
# =====================================
# Dictionary containing parameters for the KNN imputer:
# - missing_values: the placeholder for missing values (np.nan in this case)
# - n_neighbors: number of neighboring samples to use for imputation
# - weights: strategy to weight the neighbors ('uniform' means equal weight)
DATA_TRANSFORMATION_IMPUTER_PARAMS: dict = {
    "missing_values": np.nan,
    "n_neighbors": 3,
    "weights": "uniform"
}

# File name to save the fitted preprocessing object (e.g., scaler, encoder, imputer)
PREPROCESSING_OBJECT_FILE_NAME: str = "preprocessing.pkl"

# ===============================
# Model Trainer Related Contants
# ===============================

# Directory name where all model trainer related files and artifacts will be stored
MODEL_TRAINER_DIR_NAME: str = "model_trainer"

# Subdirectory within the model trainer directory specifically for storing trained model files
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"

# Filename for the serialized trained model file (e.g., pickle format)
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl"

# Minimum acceptable accuracy score threshold for a model to be considered good enough
# Models with accuracy below this will be rejected or retrained
MODEL_TRAINER_EXPECTED_SCORE: float = 0.6

# Threshold for detecting overfitting or underfitting by comparing train-test accuracy difference
# If the difference exceeds this value, it may indicate overfitting or underfitting
MODEL_TRAINER_OVER_FITTING_UNDER_FITTING_THRESHOLD: float = 0.05
