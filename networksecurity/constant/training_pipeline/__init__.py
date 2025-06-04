import os
import sys
import numpy as np
import pandas as pd

"""
Module: training_pipeline.py

This module defines **constants** used throughout the Machine Learning pipeline
for a Network Security project. Centralizing constants improves maintainability, readability, 
and scalability. These values control file names, paths, dataset configuration, 
and database connectivity.
"""

# ==============================
# General Pipeline Configuration
# ==============================

# Name of the ML pipeline (used in logs, folders, etc.)
PIPELINE_NAME: str = "NetworkSecurity"

# Root directory to store all pipeline artifacts (e.g., data, models, logs)
ARTIFACT_DIR: str = "Artifacts"

# Name of the original raw CSV file (expected input format)
FILE_NAME: str = "phisingData.csv"

# File names after train-test split
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"

# Target variable (label) column used for supervised learning tasks
TARGET_COLUMN = "Result"


# ================================
# Data Ingestion Related Constants
# ================================

# Name of the MongoDB collection where raw data is stored
DATA_INGESTION_COLLECTION_NAME: str = "NetworkData"

# Name of the MongoDB database
DATA_INGESTION_DATABASE_NAME: str = "Iyan-coder"

# Subdirectory under artifact_dir where ingestion outputs will be stored
DATA_INGESTION_DIR_NAME: str = "data_ingestion"

# Subfolder to store raw feature data before train-test split
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"

# Subfolder to store the actual train and test files
DATA_INGESTION_INGESTED_DIR: str = "ingested"

# Train-test split ratio (0.2 means 20% test, 80% train)
DATA_INGESTION_TRAIN_TEST_SPLIT_RATION: float = 0.2
