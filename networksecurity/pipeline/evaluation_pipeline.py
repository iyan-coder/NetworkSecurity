import sys
import os
# Import custom exception and logging utilities
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logger.logger import logger
import pandas as pd
# Import utility functions for loading artifacts and objects
from networksecurity.utils.main_utils.utils import get_latest_artifact_dir, load_object
# Import constant for the target/label column name
from networksecurity.constant.training_pipeline import TARGET_COLUMN
# Import training pipeline configuration class
from networksecurity.entity.config_entity import TrainingPipelineConfig
# Import model evaluator component for evaluating the trained model
from networksecurity.components.model_evaluator import ModelEvaluator
from networksecurity.entity.artifact_entity import DataTransformationArtifact


class ModelEvaluationPipeline:
    def __init__(self):
        # Load latest artifact folder
        latest_artifact_dir = get_latest_artifact_dir()

        # Base path to data transformation
        data_transformation_dir = os.path.join(latest_artifact_dir, "data_transformation")

        # Store paths you need later in evaluation
        self.transformed_train_file_path = os.path.join(data_transformation_dir, "transformed", "train.csv")
        self.transformed_test_file_path = os.path.join(data_transformation_dir, "transformed", "test.csv")
        self.feature_columns_file_path = os.path.join(data_transformation_dir, "transformed_object", "feature_columns.pkl")

        # Also store model path for evaluation
        self.model_path = os.path.join(latest_artifact_dir, "model_trainer", "trained_model", "model.pkl")

        # Init training config if needed
        self.training_pipeline_config = TrainingPipelineConfig()

    # Define the main method to run the model evaluation process
    def run_evaluation_pipeline(self):
        try:
            # Log the start of the evaluation pipeline
            logger.info("Starting standalone model evaluation pipeline...")

            # Load transformed train and test datasets
            train_df = pd.read_csv(self.transformed_train_file_path)
            test_df = pd.read_csv(self.transformed_test_file_path)

            # Load feature columns (used for selecting X)
            feature_columns = load_object(self.feature_columns_file_path)

            # Extract features and target column from the training set
            X_train = train_df[feature_columns]
            y_train = train_df[TARGET_COLUMN]

            # Extract features and target column from the test set
            X_test = test_df[feature_columns]
            y_test = test_df[TARGET_COLUMN]

            # Locate the latest trained model 
            model_path = self.model_path

            # Initialize the model evaluator in "load and evaluate" mode
            model_evaluator = ModelEvaluator(
                data_transformation_artifact=None,  # Not required in load-only mode
                model_path=model_path,
                mode="load_and_evaluate"
            )

            # Evaluate the loaded model on training and test data
            best_model, train_metric, test_metric = model_evaluator.evaluate(X_train, y_train, X_test, y_test)

            # Log the results of model evaluation
            logger.info(
                f"Model Evaluation Completed.\nTrain metric: {train_metric}\nTest metric: {test_metric}"
)
            return best_model, train_metric, test_metric
        # Handle and log any exception that occurs during evaluation
        except Exception as e:
            logger.error("Model evaluation pipeline failed", exc_info=True)
            raise NetworkSecurityException(e, sys)