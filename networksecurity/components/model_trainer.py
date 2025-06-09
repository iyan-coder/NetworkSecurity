import os
import sys
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logger.logger import logger
from networksecurity.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.entity.config_entity import ModelTrainerConfig
from networksecurity.utils.main_utils.utils import save_object, load_object, load_numpy_array_data, evaluate_models
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score
from networksecurity.utils.ml_utils.model.model_factory import load_model_config
from networksecurity.constant.training_pipeline import MODEL_CONFIG_FILE_NAME
from sklearn.metrics import roc_auc_score, accuracy_score  # Added here for completeness

class ModelTrainer:
    """
    ModelTrainer handles the complete process of training machine learning models:
    1. Loading models and hyperparameters configuration.
    2. Training multiple candidate models and evaluating their performance.
    3. Selecting the best performing model based on evaluation metrics.
    4. Saving the final model along with any preprocessing pipeline for later deployment.

    This modular design allows easy extension to include more models, hyperparameter tuning, or metric types.
    
    Real-world use case:
    In a fraud detection system (networksecurity domain), multiple models (Random Forest, XGBoost, SVM, etc.) might be 
    evaluated on transformed data to select the best classifier that minimizes false negatives and maximizes overall accuracy.
    This class automates that model selection workflow.
    """

    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
        data_transformation_artifact: DataTransformationArtifact
    ):
        """
        Initialize ModelTrainer with configuration and artifacts from data transformation stage.

        Args:
            model_trainer_config (ModelTrainerConfig): Contains file paths and model training parameters.
            data_transformation_artifact (DataTransformationArtifact): Contains paths to transformed training/testing data and preprocessor.
        """
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact

            # Load model definitions and hyperparameter grids from config file (e.g., models: RandomForest, params: n_estimators, max_depth)
            self.models, self.param_grid = load_model_config(MODEL_CONFIG_FILE_NAME)

        except Exception as e:
            logger.error("Error during ModelTrainer initialization")
            raise NetworkSecurityException(e, sys)

    def _train_and_evaluate_models(self, X_train, y_train, X_test, y_test):
        """
        Train multiple models defined in the config with hyperparameter tuning,
        and evaluate their performance on the test set.

        Args:
            X_train (np.ndarray): Training features.
            y_train (np.ndarray): Training labels.
            X_test (np.ndarray): Testing features.
            y_test (np.ndarray): Testing labels.

        Returns:
            model_report (dict): Dictionary with model names as keys and their evaluation scores.
            models (dict): Dictionary of instantiated model objects.
        """
        try:
            # evaluate_models will internally perform grid search or default training for each model
            model_report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=self.models,
                param=self.param_grid
            )
            return model_report, self.models
        except Exception as e:
            logger.error("Model training and evaluation failed.")
            raise NetworkSecurityException(e, sys)

    def _select_best_model(self, model_report, models):
        """
        Select the best model based on accuracy metric from model_report.

        Args:
            model_report (dict): Performance metrics for each model.
            models (dict): Candidate models dictionary.

        Returns:
            best_model_name (str): Name of the best performing model.
            best_model (model object): Instantiated best model object.
        """
        try:
            # Extract accuracy scores from each model's evaluation results
            accuracies = {model_name: info["accuracy"] for model_name, info in model_report.items()}

            # Select the model with the highest accuracy score
            best_model_name = max(accuracies, key=accuracies.get)
            best_model = models[best_model_name]
            best_model_score = accuracies[best_model_name]

            logger.info(f"Best model selected: {best_model_name} with accuracy: {best_model_score}")

            return best_model_name, best_model
        except Exception as e:
            logger.error("Error during best model selection.")
            raise NetworkSecurityException(e, sys)

    def _save_model(self, model, preprocessor):
        """
        Save the best model wrapped with the preprocessor pipeline.

        Args:
            model (model object): The trained ML model.
            preprocessor (object): Data preprocessing pipeline object (e.g., scaler, encoder).
        """
        try:
            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path, exist_ok=True)

            # Wrap the model with preprocessor for consistent input transformation during inference
            network_model = NetworkModel(preprocessor=preprocessor, model=model)

            # Save the combined model object to disk (pickle, joblib, etc.)
            save_object(self.model_trainer_config.trained_model_file_path, obj=network_model)

            logger.info(f"Model saved successfully at: {self.model_trainer_config.trained_model_file_path}")
        except Exception as e:
            logger.error("Failed to save the final model.")
            raise NetworkSecurityException(e, sys)


    def _print_comparison_summary(self, model_name: str, comparison: list, max_examples: int = 5):
        """
        Log a summary of prediction comparisons between true labels and model predictions.

        Args:
            model_name (str): Name of the trained model.
            comparison (list): List of dicts with keys 'y_true' and 'y_pred' for each test sample.
            max_examples (int): Max number of examples to log for correct and incorrect predictions.
        """
        total_samples = len(comparison)
        correct_preds = [c for c in comparison if c['y_true'] == c['y_pred']]
        errors = [c for c in comparison if c['y_true'] != c['y_pred']]

        logger.info(f"Model: {model_name} - Prediction Comparison Summary")
        logger.info(f"Total samples: {total_samples}")
        logger.info(f"Correct predictions: {len(correct_preds)}")
        logger.info(f"Misclassified samples: {len(errors)}")

        logger.info(f"Examples of Correct Predictions (up to {max_examples}):")
        for example in correct_preds[:max_examples]:
            logger.info(f"  True: {example['y_true']}  Predicted: {example['y_pred']}")

        logger.info(f"Examples of Misclassifications (up to {max_examples}):")
        for example in errors[:max_examples]:
            logger.info(f"  True: {example['y_true']}  Predicted: {example['y_pred']}")

        logger.info("="*50)

    def train_model(self, X_train, y_train, X_test, y_test):
        """
        Execute full model training pipeline:
        - Train and evaluate all candidate models.
        - Select the best model.
        - Evaluate final model on train and test sets.
        - Save the best model with preprocessor.
        - Log detailed prediction comparisons for the test set.

        Args:
            X_train (np.ndarray): Training features.
            y_train (np.ndarray): Training labels.
            X_test (np.ndarray): Testing features.
            y_test (np.ndarray): Testing labels.

        Returns:
            ModelTrainerArtifact: Artifact summarizing training outcome and metrics.
        """
        try:
            logger.info("Training and evaluating models...")
            model_report, models = self._train_and_evaluate_models(X_train, y_train, X_test, y_test)

            best_model_name, best_model = self._select_best_model(model_report, models)

            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            if hasattr(best_model, "predict_proba"):
                y_train_proba = best_model.predict_proba(X_train)[:, 1]
                y_test_proba = best_model.predict_proba(X_test)[:, 1]
            else:
                y_train_proba = None
                y_test_proba = None

            classification_train_metric = get_classification_score(y_true=y_train, y_pred=y_train_pred, y_proba=y_train_proba)
            classification_test_metric = get_classification_score(y_true=y_test, y_pred=y_test_pred, y_proba=y_test_proba)

            # Generate list comparing true vs predicted labels for the test set
            comparison = [
                {"y_true": true_label, "y_pred": pred_label}
                for true_label, pred_label in zip(y_test.tolist(), y_test_pred.tolist())
            ]

            # Log comparison summary to help understand model predictions on test set
            self._print_comparison_summary(best_model_name, comparison)

            preprocessor = load_object(self.data_transformation_artifact.transformed_object_file_path)

            self._save_model(best_model, preprocessor)

            return ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=classification_train_metric,
                test_metric_artifact=classification_test_metric
            )
        except Exception as e:
            logger.error("Training pipeline failed.")
            raise NetworkSecurityException(e, sys)


    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        """
        Load transformed train and test data, and start the model training pipeline.

        Returns:
            ModelTrainerArtifact: Result artifact from the training process.
        """
        logger.info("Entered initiate_model_trainer method of ModelTrainer class")
        try:
            # Load numpy arrays containing features + target from transformation artifact
            train_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_test_file_path)

            # Separate features and target variables (last column assumed to be target)
            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            # Call training pipeline
            model_trainer_artifact = self.train_model(X_train, y_train, X_test, y_test)
            logger.info("Model trainer pipeline completed successfully.\n"
            f"ModelTrainerArtifact: {model_trainer_artifact}")
            return model_trainer_artifact

        except Exception as e:
            logger.error("Model Trainer process failed.", exc_info=True)
            raise NetworkSecurityException(e, sys)
