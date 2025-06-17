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
import dagshub
import yaml

# Initialize DagsHub tracking
dagshub.init(repo_owner='iyan-coder', repo_name='networksecurity', mlflow=True)

class ModelEvaluator:
    """
    Handles:
    - Training multiple models and selecting the best one.
    - Or loading a saved model and evaluating its performance.
    """

    def __init__(self, data_transformation_artifact=None, model_path=None, mode="train_and_evaluate"):
        logger.info("Initializing ModelEvaluator...")

        # Save initialization arguments
        self.data_transformation_artifact = data_transformation_artifact
        self.model_path = model_path
        self.mode = mode

        # If we're training, load model definitions and hyperparameters
        if self.mode == "train_and_evaluate":
            self.models, self.param_grid = load_model_config(MODEL_CONFIG_FILE_NAME)
            logger.info("Loaded model configurations and parameter grid.")

        # If we're just loading a pre-trained model for evaluation
        elif self.mode == "load_and_evaluate":
            if not model_path or not os.path.exists(model_path):
                raise NetworkSecurityException(f"Model path is invalid or does not exist: {model_path}", sys)
            logger.info(f"ModelEvaluator set to load mode. Model will be loaded from {model_path}.")

    def _track_with_mlflow(self, model, metric, model_name: str, stage: str, X_sample, y_sample):
        """
        Logs a model and its metrics to MLflow for experiment tracking.

        Args:
            model: Trained model object
            metric: Object with evaluation scores (accuracy, precision, etc.)
            model_name: Name of the model
            stage: "train" or "test"
            X_sample: Example input data (small sample)
            y_sample: Example target values
        """
        with mlflow.start_run(nested=True):  # Start a nested MLflow run
            mlflow.set_tag("stage", stage)  # e.g., train or test
            mlflow.set_tag("model_name", model_name)
            mlflow.log_param("model_name", model_name)

            # Dictionary of metrics to log
            metrics_to_log = {
                "f1_score": metric.f1_score,
                "precision_score": metric.precision_score,
                "recall_score": metric.recall_score,
                "accuracy_score": metric.accuracy_score,
                "roc_auc_score": metric.roc_auc_score
            }

            # Remove any None values (e.g., roc_auc for multiclass)
            safe_metrics = {k: v for k, v in metrics_to_log.items() if v is not None}

            # Log each metric with stage prefix (e.g., train_accuracy)
            mlflow.log_metrics({f"{stage}_{k}": v for k, v in safe_metrics.items()})

            # Infer model signature (input/output schema)
            signature = infer_signature(X_sample, model.predict(X_sample))

            # Log the trained model
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=f"{model_name}_{stage}_model",
                input_example=pd.DataFrame(X_sample),
                signature=signature
            )

    def evaluate(self, X_train, y_train, X_test, y_test):
        """
        Evaluates the model(s) using training and test sets.

        Returns:
            best_model: Trained or loaded model
            train_metric: Metrics on training data
            test_metric: Metrics on test data
        """
        try:
            logger.info("Starting model evaluation process...")

            # Load feature column names if training mode
            feature_columns = None
            if self.mode == "train_and_evaluate":
                feature_columns = load_object(self.data_transformation_artifact.feature_columns_file_path)

            # Convert to DataFrame with proper columns if needed
            if not isinstance(X_train, pd.DataFrame):
                if feature_columns:
                    X_train = pd.DataFrame(X_train, columns=feature_columns)
                else:
                    X_train = pd.DataFrame(X_train)

            if not isinstance(X_test, pd.DataFrame):
                if feature_columns:
                    X_test = pd.DataFrame(X_test, columns=feature_columns)
                else:
                    X_test = pd.DataFrame(X_test)

            # === LOAD MODE ===
            if self.mode == "load_and_evaluate":
                logger.info(f"Loading pre-trained model from {self.model_path}")
                best_model = joblib.load(self.model_path)  # Load the saved model
                best_model_name = type(best_model).__name__  # Get model's class name

            # === TRAIN MODE ===
            else:
                logger.info("Training and evaluating models...")
                with open(MODEL_CONFIG_FILE_NAME, "r") as yaml_file:
                    model_yaml = yaml.safe_load(yaml_file)

                    # Log all hyperparameter spaces
                for model_name, model_data in model_yaml.get("models", {}).items():
                    for param_name, values in model_data.get("params", {}).items():
                        mlflow.log_param(f"{model_name}_{param_name}_space", str(values))


                # Train all models and collect scores
                model_report, trained_models = evaluate_models(
                    X_train, y_train, X_test, y_test,
                    models=self.models, param=self.param_grid,
                    skip_training=False
                )

                # Select the model with the highest test accuracy
                best_model_name = max(model_report, key=lambda name: model_report[name]["accuracy"])
                best_model = trained_models[best_model_name]
                logger.info(f"Best model selected: {best_model_name} with accuracy {model_report[best_model_name]['accuracy']:.4f}")

            # Make predictions
            logger.info("Generating predictions and computing metrics...")
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            # If model supports probability prediction, get them
            y_train_proba = best_model.predict_proba(X_train)[:, 1] if hasattr(best_model, "predict_proba") else None
            y_test_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else None

            # Compute evaluation metrics
            train_metric = get_classification_score(y_train, y_train_pred, y_train_proba)
            test_metric = get_classification_score(y_test, y_test_pred, y_test_proba)

            # Log to MLflow if training mode
            logger.info("Logging to MLflow...")
            if self.mode == "train_and_evaluate":
                self._track_with_mlflow(best_model, train_metric, best_model_name, stage="train", X_sample=X_train[:5], y_sample=y_train[:5])
                self._track_with_mlflow(best_model, test_metric, best_model_name, stage="test", X_sample=X_test[:5], y_sample=y_test[:5])

            return best_model, train_metric, test_metric

        except Exception as e:
            # Catch any error and raise custom exception
            logger.error("Error during model evaluation", exc_info=True)
            raise NetworkSecurityException(e, sys)
