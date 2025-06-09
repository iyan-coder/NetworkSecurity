import os
import sys
import mlflow
import mlflow.sklearn
import pandas as pd

from networksecurity.logger.logger import logger
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score
from networksecurity.utils.ml_utils.model.model_factory import load_model_config
from networksecurity.utils.main_utils.utils import evaluate_models
from networksecurity.constant.training_pipeline import MODEL_CONFIG_FILE_NAME

from mlflow.models.signature import infer_signature
import joblib

class ModelEvaluator:
    """
    Responsible for evaluating models. Can either:
    1. Train and evaluate multiple models (default)
    2. Load a saved model and only evaluate (skip training)
    """

    def __init__(self, data_transformation_artifact, model_path=None, mode="train_and_evaluate"):
        logger.info("Initializing ModelEvaluator...")
        self.data_transformation_artifact = data_transformation_artifact
        self.mode = mode
        self.model_path = model_path

        if self.mode == "train_and_evaluate":
            self.models, self.param_grid = load_model_config(MODEL_CONFIG_FILE_NAME)
            logger.info("Loaded model configurations and parameter grid.")
        elif self.mode == "load_and_evaluate":
            if not model_path or not os.path.exists(model_path):
                raise NetworkSecurityException(f"Model path is invalid or does not exist: {model_path}", sys)
            logger.info(f"ModelEvaluator set to load mode. Model will be loaded from {model_path}.")

    def _track_with_mlflow(self, model, metric, model_name: str, stage: str, X_sample, y_sample):
        with mlflow.start_run(nested=True):
            mlflow.set_tag("stage", stage)
            mlflow.set_tag("model_name", model_name)
            mlflow.log_param("model_name", model_name)

            mlflow.log_metrics({
                "f1_score": metric.f1_score,
                "precision_score": metric.precision_score,
                "recall_score": metric.recall_score,
                "accuracy_score": metric.accuracy_score,
                "roc_auc_score": metric.roc_auc_score
            })

            signature = infer_signature(X_sample, model.predict(X_sample))

            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=f"{model_name}_{stage}_model",
                input_example=pd.DataFrame(X_sample),
                signature=signature
            )

    def evaluate(self, X_train, y_train, X_test, y_test):
        try:
            logger.info("Starting model evaluation process...")

            if self.mode == "load_and_evaluate":
                logger.info(f"Loading pre-trained model from {self.model_path}")
                best_model = joblib.load(self.model_path)
                best_model_name = type(best_model).__name__

            else:
                logger.info("Training and evaluating models...")
                model_report, trained_models = evaluate_models(
                    X_train, y_train, X_test, y_test,
                    models=self.models, param=self.param_grid,
                    skip_training=False
                )

                best_model_name = max(model_report, key=lambda name: model_report[name]["accuracy"])
                best_model = trained_models[best_model_name]
                logger.info(f"Best model selected: {best_model_name} with accuracy {model_report[best_model_name]['accuracy']:.4f}")

            logger.info("Generating predictions and computing metrics...")
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            y_train_proba = best_model.predict_proba(X_train)[:, 1] if hasattr(best_model, "predict_proba") else None
            y_test_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else None

            train_metric = get_classification_score(y_train, y_train_pred, y_train_proba)
            test_metric = get_classification_score(y_test, y_test_pred, y_test_proba)

            logger.info("Logging to MLflow...")
            if self.mode == "train_and_evaluate":
                self._track_with_mlflow(best_model, train_metric, best_model_name, stage="train", X_sample=X_train[:5], y_sample=y_train[:5])

            self._track_with_mlflow(best_model, test_metric, best_model_name, stage="test", X_sample=X_test[:5], y_sample=y_test[:5])

            logger.info("Model evaluation completed.")
            return best_model, train_metric, test_metric

        except Exception as e:
            logger.error("Error during model evaluation", exc_info=True)
            raise NetworkSecurityException(e, sys)
