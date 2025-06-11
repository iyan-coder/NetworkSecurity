# These are import statements that bring in other code we need to use in this file.
import os  # Helps with file and folder paths
import sys  # Helps handle system-level errors
import mlflow  # Used to track model experiments
import mlflow.sklearn  # Helps log sklearn models to MLflow
import pandas as pd  # Used to work with tabular data like spreadsheets

# Custom logging, error handling, and utility functions from our own project
from networksecurity.logger.logger import logger  # Logs info and errors
from networksecurity.exception.exception import NetworkSecurityException  # Custom error class for our project
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score  # Calculates metrics like accuracy
from networksecurity.utils.ml_utils.model.model_factory import load_model_config  # Loads which ML models and parameters we want to try
from networksecurity.utils.main_utils.utils import evaluate_models, load_object  # Evaluates multiple models, loads saved files
from networksecurity.constant.training_pipeline import MODEL_CONFIG_FILE_NAME  # File name for model configuration

# Used to get info about model inputs/outputs for MLflow logging
from mlflow.models.signature import infer_signature  
import joblib  # Used to load or save trained models
class ModelEvaluator:
    """
    This class is in charge of:
    1. Trying out multiple machine learning models, training them, and picking the best one.
    2. OR, just loading an already trained model and checking how good it is.
    """
    def __init__(self, data_transformation_artifact, model_path=None, mode="train_and_evaluate"):
        """
        Sets up everything we need for evaluation.

        data_transformation_artifact: Information about the data (like feature names)
        model_path: Path to an already trained model (if we are loading instead of training)
        mode: Either "train_and_evaluate" or "load_and_evaluate"
        """
        logger.info("Initializing ModelEvaluator...")
        self.data_transformation_artifact = data_transformation_artifact
        self.mode = mode
        self.model_path = model_path

        if self.mode == "train_and_evaluate":
            # If we're training, we load all the models and settings we want to try.
            self.models, self.param_grid = load_model_config(MODEL_CONFIG_FILE_NAME)
            logger.info("Loaded model configurations and parameter grid.")

        elif self.mode == "load_and_evaluate":
            # If we're only evaluating an existing model
            if not model_path or not os.path.exists(model_path):
                raise NetworkSecurityException(f"Model path is invalid or does not exist: {model_path}", sys)
            logger.info(f"ModelEvaluator set to load mode. Model will be loaded from {model_path}.")

    def _track_with_mlflow(self, model, metric, model_name: str, stage: str, X_sample, y_sample):
        """
        Logs model and its metrics to MLflow so we can track performance and compare models later.

        model: the trained ML model
        metric: the metrics we calculated
        model_name: the name of the model
        stage: 'train' or 'test'
        X_sample: small sample of input data (just for MLflow example)
        y_sample: corresponding labels for the input
        """
        with mlflow.start_run(nested=True):  # Start a new MLflow run
            mlflow.set_tag("stage", stage)  # Tag whether it's training or testing
            mlflow.set_tag("model_name", model_name)  # Tag the model's name
            mlflow.log_param("model_name", model_name)  # Log the model name as a parameter

            # Create a dictionary of all the metrics
            metrics_to_log = {
                "f1_score": metric.f1_score,
                "precision_score": metric.precision_score,
                "recall_score": metric.recall_score,
                "accuracy_score": metric.accuracy_score,
                "roc_auc_score": metric.roc_auc_score
            }
            # Remove any metric that is None (just to be safe)
            safe_metrics = {k: v for k, v in metrics_to_log.items() if v is not None}
            # Log all the safe metrics to MLflow
            mlflow.log_metrics({f"{stage}_{k}": v for k, v in safe_metrics.items()})
            # Log the model itself, along with how the inputs/outputs look (for documentation)
            signature = infer_signature(X_sample, model.predict(X_sample))

            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=f"{model_name}_{stage}_model",
                input_example=pd.DataFrame(X_sample),
                signature=signature
            )
    def evaluate(self, X_train, y_train, X_test, y_test):
        """
        The main method to either:
        1. Train models, choose the best one, and test it.
        2. Or, load a model and evaluate it.
        
        Returns:
            best_model: The model that performed best
            train_metric: Training performance
            test_metric: Testing performance
        """
        try:
            logger.info("Starting model evaluation process...")
            # Load the column names from the data transformation step
            feature_columns = load_object(self.data_transformation_artifact.feature_columns_file_path)
            # If the input is not a DataFrame, convert it to one with proper column names
            if not isinstance(X_train, pd.DataFrame):
                X_train = pd.DataFrame(X_train, columns=feature_columns)

            if not isinstance(X_test, pd.DataFrame):
                X_test = pd.DataFrame(X_test, columns=feature_columns)
            # If we are loading a saved model
            if self.mode == "load_and_evaluate":
                logger.info(f"Loading pre-trained model from {self.model_path}")
                best_model = joblib.load(self.model_path)
                best_model_name = type(best_model).__name__  # Get the model's class name
            else:
                logger.info("Training and evaluating models...")


                # Train and test all models, collect scores
                model_report, trained_models = evaluate_models(
                    X_train, y_train, X_test, y_test,
                    models=self.models, param=self.param_grid,
                    skip_training=False
                )

                # Find the model with the highest test accuracy
                best_model_name = max(model_report, key=lambda name: model_report[name]["accuracy"])
                best_model = trained_models[best_model_name]

                logger.info(f"Best model selected: {best_model_name} with accuracy {model_report[best_model_name]['accuracy']:.4f}")
            
            logger.info("Generating predictions and computing metrics...")
            # Predict outputs
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)
            # If model supports probabilities, get those too
            y_train_proba = best_model.predict_proba(X_train)[:, 1] if hasattr(best_model, "predict_proba") else None
            y_test_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else None
            # Calculate metrics like accuracy, precision, recall, etc.
            train_metric = get_classification_score(y_train, y_train_pred, y_train_proba)
            test_metric = get_classification_score(y_test, y_test_pred, y_test_proba)

            logger.info("Logging to MLflow...")
            # Log the model and metrics to MLflow
            if self.mode == "train_and_evaluate":
                self._track_with_mlflow(best_model, train_metric, best_model_name, stage="train", X_sample=X_train[:5], y_sample=y_train[:5])
                self._track_with_mlflow(best_model, test_metric, best_model_name, stage="test", X_sample=X_test[:5], y_sample=y_test[:5])

            logger.info("Model evaluation completed.")
            logger.info("Model evaluation completed.")
            logger.info(f"Best model evaluation completed.\nTrain Accuracy: {train_metric.accuracy_score}\nTest Accuracy: {test_metric.accuracy_score}")

            return best_model, train_metric, test_metric  # Return final results
            
        except Exception as e:
                # If anything goes wrong, log the error and raise a custom exception
                logger.error("Error during model evaluation", exc_info=True)
                raise NetworkSecurityException(e, sys)




