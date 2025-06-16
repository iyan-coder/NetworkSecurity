
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
from networksecurity.constant.training_pipeline import TRAINING_BUCKET_NAME
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
from networksecurity.entity.artifact_entity import(
    DataIngetionArtifact,
    DataTransformationArtifact,
    DataValidationArtifact,
    ModelTrainerArtifact
)

# Utility functions
from networksecurity.constant.training_pipeline import TARGET_COLUMN
from networksecurity.utils.main_utils.utils import load_object, get_latest_artifact_dir
from networksecurity.cloud.s3_syncer import S3Sync

import numpy as np
import mlflow
import dagshub
dagshub.init(repo_owner='iyan-coder', repo_name='networksecurity', mlflow=True)

class TrainingPipeline:
    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()
        self.s3_sync = S3Sync()

    def start_data_ingestion(self):
        try:
            mlflow.set_experiment("NetworkSecurityModelTraining")
            
            self.data_ingestion_config = DataIngestionConfig(training_pipeline_config=self.training_pipeline_config)
            logger.info("start data Ingestion")
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artfifact = data_ingestion.initiate_data_ingestion()
            logger.info(f"Data Ingestion completed and artifact: {data_ingestion_artfifact}")
            return data_ingestion_artfifact
        except Exception as e:
            logger.error("Error!, Data ingestion failed", exc_info=True)
            raise NetworkSecurityException(e, sys)

    def start_data_validation(self, data_ingestion_artifact=DataIngetionArtifact):
        try:
            self.data_validation_config = DataValidationConfig(training_pipeline_config=self.training_pipeline_config)
            logger.info("start data validation")
            data_validation = DataValidation(data_validation_config=self.data_validation_config, 
                                             data_ingestion_artifact=data_ingestion_artifact)
            data_validation_artifact = data_validation.initiate_data_validation()
            logger.info(f"Data validation completed and artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            logger.error("Error!, Data validation failed",exc_info=True)
            raise NetworkSecurityException(e, sys)

    def start_data_transfomation(self, data_validation_artifact=DataValidationArtifact):
        try:
            self.data_transformation_config = DataTransformationConfig(training_pipeline_config=self.training_pipeline_config)
            logger.info("start data transfomation")
            data_transformation = DataTransformation(data_transformation_config=self.data_transformation_config,
                                                     data_validation_artifact=data_validation_artifact)
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            logger.info(f"Data transformation completed and artifact: {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            logger.error("Error!, Data transformation failed",exc_info=True)
            raise NetworkSecurityException(e, sys)

    def start_model_trainer(self, data_transformation_artifact=DataTransformationArtifact):
        try:
            self.model_trainer_config = ModelTrainerConfig(training_pipeline_config=self.training_pipeline_config)
            logger.info("start model trainer")
            model_trainer = ModelTrainer(data_transformation_artifact=data_transformation_artifact,
                                         model_trainer_config=self.model_trainer_config)      

            model_trainer_artifact = model_trainer.initiate_model_trainer()
            logger.info(f"""
                Model Trainer completed successfully.
                Model path: {model_trainer_artifact.trained_model_file_path}
                Train Accuracy: {model_trainer_artifact.train_metric_artifact.accuracy_score}
                Test Accuracy: {model_trainer_artifact.test_metric_artifact.accuracy_score}
                """)
            return model_trainer_artifact
        except Exception as e:
            logger.error("Error!, Model trainer failed", exc_info=True)
            raise NetworkSecurityException(e, sys)
    ### local artifact is going to s3 bucket
    def sync_artifact_dir_to_s3(self):
        try:
            aws_bucket_url = f"s3://{TRAINING_BUCKET_NAME}/artifact/{self.training_pipeline_config.timestamp}"
            self.s3_sync.sync_folder_to_s3(folder = self.training_pipeline_config.artifact_dir,aws_bucket_url=aws_bucket_url)
        except Exception as e:
            logger.error("sync artifact_dir to s3 failed!")
            raise NetworkSecurityException(e,sys)
    ### local artifact is going to s3 bucket
    def sync_saved_model_dir_to_s3(self):
        try:
            aws_bucket_url = f"s3://{TRAINING_BUCKET_NAME}/final_model/{self.training_pipeline_config.timestamp}"
            self.s3_sync.sync_folder_to_s3(folder = self.training_pipeline_config.model_dir,aws_bucket_url=aws_bucket_url)
        except Exception as e:
            logger.error("sync saved_model_dir to s3 failed!")
            raise NetworkSecurityException(e,sys)
    

    def run_training_pipeline(self):
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact )
            data_transformation_artifact = self.start_data_transfomation(data_validation_artifact)
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact )
            self.sync_artifact_dir_to_s3()
            self.sync_saved_model_dir_to_s3()
            logger.info("Run pipeline completed")
            return model_trainer_artifact
        
        except Exception as e:
            logger.error("failed to run pipeline",exc_info=True)
            raise NetworkSecurityException(e,sys)

