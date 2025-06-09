from networksecurity.constant.training_pipeline import SAVED_MODEL_DIR, MODEL_FILE_NAME
import os
import sys
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logger.logger import logger

class NetworkModel:
    """
    A wrapper class that combines data preprocessing and the machine learning model
    into a single entity to streamline the inference (prediction) pipeline.

    Real-world intuition:
    In production environments, raw input data must be transformed (scaled, encoded, etc.)
    before being fed to the trained model. This class ensures that the same preprocessing
    steps applied during training are consistently applied at inference time,
    preventing data mismatch errors and improving reliability.
    """

    def __init__(self, preprocessor, model):
        """
        Initialize the NetworkModel with a preprocessor and trained model.

        Args:
            preprocessor: A fitted preprocessing object (e.g., scaler, encoder, pipeline)
            model: The trained machine learning model object
        
        Raises:
            NetworkSecurityException: If initialization fails due to any error
        """
        try:
            self.preprocessor = preprocessor
            self.model = model
        except Exception as e:
            logger.error("Failed to initiate NetworkModel")
            raise NetworkSecurityException(e, sys)
        
    def predict(self, x):
        """
        Perform prediction on raw input data by first applying preprocessing,
        then using the trained model to generate predictions.

        Args:
            x: Raw input data (e.g., numpy array or DataFrame)

        Returns:
            y_hat: Model predictions corresponding to input x
        
        Raises:
            NetworkSecurityException: If prediction fails due to any error
        """
        try:
            # Apply the preprocessing pipeline to transform raw input features
            x_transform = self.preprocessor.transform(x)

            # Predict labels or values using the trained model on transformed data
            y_hat = self.model.predict(x_transform)
            return y_hat
        except Exception as e:
            logger.error("Error! Prediction failed")
            raise NetworkSecurityException(e, sys)
