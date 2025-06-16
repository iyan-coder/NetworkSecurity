from networksecurity.constant.training_pipeline import SAVED_MODEL_DIR, MODEL_FILE_NAME
import os
import sys
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logger.logger import logger
import pandas as pd

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

    def __init__(self, preprocessor, model, feature_columns):
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
            self.feature_columns = feature_columns  
        except Exception as e:
            logger.error("Failed to initiate NetworkModel")
            raise NetworkSecurityException(e, sys)
        
    def predict(self, x):
        """
        Transforms input and returns model predictions.
        """
        try:
            if not isinstance(x, pd.DataFrame):
                x = pd.DataFrame(x, columns=self.feature_columns)
            else:
                x = x[self.feature_columns]

            # Transform input
            x_transformed = self.preprocessor.transform(x)

            #Convert back to DataFrame to match training structure
            x_transformed_df = pd.DataFrame(x_transformed, columns=self.model.feature_names_in_)

            # Predict using DataFrame input to avoid the warning
            y_hat = self.model.predict(x_transformed_df)

            return y_hat

        
        except Exception as e:
            logger.error("Error! Prediction failed")
            raise NetworkSecurityException(e, sys)
