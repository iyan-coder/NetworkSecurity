import os, sys
from networksecurity.logger.logger import logger
from networksecurity.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from networksecurity.entity.config_entity import ModelTrainerConfig
from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.utils.main_utils.utils import save_object, load_object, load_numpy_array_data
from networksecurity.components.model_evaluator import ModelEvaluator
from networksecurity.exception.exception import NetworkSecurityException

class ModelTrainer:
    """
    ModelTrainer is responsible for:
    - Loading the transformed training and testing data.
    - Using ModelEvaluator to find the best model based on metrics.
    - Saving the best model wrapped with the preprocessor.
    - Returning a ModelTrainerArtifact with file path and evaluation metrics.
    """

    def __init__(self, model_trainer_config: ModelTrainerConfig,
                 data_transformation_artifact: DataTransformationArtifact):
        """
        Constructor for ModelTrainer.

        Args:
            model_trainer_config (ModelTrainerConfig): Config containing path to save the model.
            data_transformation_artifact (DataTransformationArtifact): Contains paths to transformed data.
        """
        logger.info("Initializing ModelTrainer...")
        self.model_trainer_config = model_trainer_config
        self.data_transformation_artifact = data_transformation_artifact

    def _save_model(self, model, preprocessor):
        """
        Internal method to save the trained model along with its preprocessor.

        Args:
            model: The trained ML model.
            preprocessor: The preprocessor (e.g., StandardScaler, ColumnTransformer) used in training.
        """
        try:
            logger.info("Saving the trained model with preprocessor...")

            # Ensure model directory exists
            model_dir = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir, exist_ok=True)

            # Wrap the model with its preprocessor
            network_model = NetworkModel(preprocessor=preprocessor, model=model)

            # Save the wrapped model object
            save_object(self.model_trainer_config.trained_model_file_path, obj=network_model)
            logger.info(f"Model saved successfully at: {self.model_trainer_config.trained_model_file_path}")

        except Exception as e:
            logger.error("Failed to save model.", exc_info=True)
            raise NetworkSecurityException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        """
        Main function to:
        - Load transformed data.
        - Train and evaluate multiple models using ModelEvaluator.
        - Save the best performing model.
        - Return training artifacts.

        Returns:
            ModelTrainerArtifact: Contains model path and performance metrics.
        """
        try:
            logger.info("Loading transformed train and test data...")

            # Load transformed train and test data from disk (saved as .npy arrays)
            train_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_test_file_path)

            # Separate features (X) and labels (y)
            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]
            logger.info("Successfully split features and labels from transformed arrays.")

            # Initialize the evaluator to find the best model
            logger.info("Evaluating models to find the best one...")
            evaluator = ModelEvaluator(self.data_transformation_artifact)
            best_model, train_metric, test_metric = evaluator.evaluate(X_train, y_train, X_test, y_test)

            # Load the preprocessor (used during data transformation)
            logger.info("Loading data preprocessor used during transformation...")
            preprocessor = load_object(self.data_transformation_artifact.transformed_object_file_path)

            # Save the best model (wrapped with preprocessor)
            logger.info("Saving the final trained model...")
            self._save_model(best_model, preprocessor)

            # Log final training completion
            logger.info("Model training completed successfully.")

            # Return the artifact with model path and performance metrics
            return ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=train_metric,
                test_metric_artifact=test_metric
            )

        except Exception as e:
            logger.error("Failed to initiate model trainer.", exc_info=True)
            raise NetworkSecurityException(e, sys)
