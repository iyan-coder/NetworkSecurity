# Import necessary Python libraries
import os  # To create folders and handle file paths
import sys  # For system-specific exception handling
import pandas as pd  # Used to work with tabular data (DataFrames)
import mlflow
# Project-specific imports
from networksecurity.logger.logger import logger  # For logging messages
from networksecurity.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from networksecurity.entity.config_entity import ModelTrainerConfig
from networksecurity.utils.ml_utils.model.estimator import NetworkModel  # Custom model wrapper
from networksecurity.utils.main_utils.utils import save_object, load_object  # To save/load Python objects
from networksecurity.components.model_evaluator import ModelEvaluator  # Evaluates different models
from networksecurity.exception.exception import NetworkSecurityException  # Custom exception
from networksecurity.constant.training_pipeline import TARGET_COLUMN  # The name of the column we're predicting
import dagshub
dagshub.init(repo_owner='iyan-coder', repo_name='networksecurity', mlflow=True)


# ModelTrainer handles training and saving the best ML model
class ModelTrainer:
    """
    This class:
    - Loads the transformed train/test data.
    - Trains models and finds the best one using ModelEvaluator.
    - Saves the best model wrapped with the preprocessor.
    - Returns an artifact with the model path and performance scores.
    """

    def __init__(self, model_trainer_config: ModelTrainerConfig,
                 data_transformation_artifact: DataTransformationArtifact):
        """
        Constructor sets up necessary configs and paths.

        Args:
            model_trainer_config: Where to save the trained model.
            data_transformation_artifact: Where to get transformed train/test data and preprocessor.
        """
        logger.info("Initializing ModelTrainer...")
        self.model_trainer_config = model_trainer_config
        self.data_transformation_artifact = data_transformation_artifact

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        """
        Reads CSV file from disk and returns it as a pandas DataFrame.
        """
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            logger.error("Failed to read file", exc_info=True)
            raise NetworkSecurityException(e, sys)

    def _save_model(self, model, preprocessor):
        """
        Saves the trained model together with the preprocessor into a file.

        Args:
            model: Best trained model.
            preprocessor: The preprocessor used during data transformation.
        """
        try:
            logger.info("Saving the trained model with preprocessor...")

            # Create the directory for saving the model if it doesn't exist
            model_dir = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir, exist_ok=True)

            # Load the list of feature columns to make sure everything stays aligned
            feature_columns = load_object(self.data_transformation_artifact.feature_columns_file_path)

            # Wrap model with its preprocessor and column names
            network_model = NetworkModel(preprocessor=preprocessor, model=model, feature_columns=feature_columns)

            # Save the wrapped model to disk
            save_object(self.model_trainer_config.trained_model_file_path, obj=network_model)

            logger.info(f"Model saved successfully at: {self.model_trainer_config.trained_model_file_path}")

        except Exception as e:
            logger.error("Failed to save model.", exc_info=True)
            raise NetworkSecurityException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        """
        Runs the full training process:
        - Loads transformed data
        - Splits into features and labels
        - Trains and evaluates models
        - Saves best model
        - Returns an artifact with results
        """
        try:
            logger.info("Loading transformed train and test data...")
            with mlflow.start_run(run_name="Model_Training_Pipeline"):

                # Log basic params
                mlflow.log_param("pipeline_step", "ModelTrainer")
                mlflow.log_param("model_storage_path", self.model_trainer_config.trained_model_file_path)

            # Load the CSV files containing transformed data
            train_df = self.read_data(self.data_transformation_artifact.transformed_train_file_path)
            test_df = self.read_data(self.data_transformation_artifact.transformed_test_file_path)

            # Load feature column names (needed to separate features from labels)
            feature_columns = load_object(self.data_transformation_artifact.feature_columns_file_path)

            # Split the training data into X (features) and y (target)
            X_train = train_df[feature_columns]
            y_train = train_df[TARGET_COLUMN]

            # Split the test data the same way
            X_test = test_df[feature_columns]
            y_test = test_df[TARGET_COLUMN]

            logger.info("Successfully split features and labels from transformed arrays.")

            # Use ModelEvaluator to find the best-performing model

            logger.info("Evaluating models to find the best one...")
            evaluator = ModelEvaluator(self.data_transformation_artifact)
            best_model, train_metric, test_metric = evaluator.evaluate(X_train, y_train, X_test, y_test)

            # Load the preprocessor used during data transformation
            logger.info("Loading data preprocessor used during transformation...")
            preprocessor = load_object(self.data_transformation_artifact.transformed_object_file_path)

            # Save the model and preprocessor wrapped together
            logger.info("Saving the final trained model...")
            self._save_model(best_model, preprocessor)
            

            mlflow.log_metrics({
                                "final_train_accuracy": train_metric.accuracy_score,
                                "final_test_accuracy": test_metric.accuracy_score,
                                "final_train_f1": train_metric.f1_score,
                                "final_test_f1": test_metric.f1_score
                            })

            logger.info("Model training completed successfully.")

            # Return model training results
            return ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=train_metric,
                test_metric_artifact=test_metric
            )

        except Exception as e:
            logger.error("Failed to initiate model trainer.", exc_info=True)
            raise NetworkSecurityException(e, sys)