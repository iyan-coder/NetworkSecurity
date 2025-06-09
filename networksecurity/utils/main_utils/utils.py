import yaml
from networksecurity.exception.exception import NetworkSecurityException
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from typing import Any, Dict
from networksecurity.logger.logger import logger
import os
import sys
import numpy as np
import pandas as pd
import dill
import pickle

# =====================
# READ YAML FILE METHOD
# =====================

def read_yaml_file(file_path: str) -> dict:
    """
    Reads a YAML file and returns its contents as a dictionary.

    Args:
        file_path (str): The path to the YAML file.

    Returns:
        dict: Parsed YAML content as a dictionary.

    Raises:
        NetworkSecurityException: If reading or parsing the YAML file fails.
    """
    try:
        # Open the YAML file in binary read mode
        with open(file_path, "rb") as yaml_file:
            # Parse and return content as dictionary
            return yaml.safe_load(yaml_file)
    
    except Exception as e:
        # Log full error traceback for debugging
        logger.error("Could not read_yaml_file", exc_info=True)
        # Raise custom exception with system info
        raise NetworkSecurityException(e, sys)

# ======================
# WRITE YAML FILE METHOD
# ======================

def write_yaml_file(file_path: str, content: dict, replace: bool = False) -> None:
    """
    Saves a dictionary to a YAML file, optionally replacing an existing file.

    Args:
        file_path (str): Path to save the YAML file.
        content (dict): Data to write to the YAML file.
        replace (bool): If True, removes file if it already exists.
    """
    try:
        # If replace is enabled and file exists, delete it
        if replace and os.path.exists(file_path):
            os.remove(file_path)

        # Ensure the directory exists before saving
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Write dictionary content to YAML file
        with open(file_path, "w") as file:
            yaml.dump(
                content,                    # Dictionary to write
                file,                       # File object
                default_flow_style=False,   # Use block YAML style (cleaner)
                sort_keys=False,            # Preserve original dictionary order
                Dumper=yaml.SafeDumper      # Use safe dumper (safe serialization)
            )

    except Exception as e:
        logger.error("Failed to write yaml file", exc_info=True)
        raise NetworkSecurityException(e, sys)

# ==========================
# SAVE NUMPY ARRAY TO DISK
# ==========================

def save_numpy_array_data(file_path: str, array: np.array):
    """
    Saves a NumPy array to a .npy file.

    Args:
        file_path (str): Where to save the .npy file.
        array (np.array): The NumPy array to save.
    """
    try:
        # Create directory if it doesn't exist
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        # Save array in binary format
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)

    except Exception as e:
        logger.error("Failed to save numpy array file", exc_info=True)
        raise NetworkSecurityException(e, sys)

# =====================
# SAVE PICKLE OBJECT
# =====================

def save_object(file_path: str, obj: object) -> None:
    """
    Saves any Python object to disk using Pickle.

    Args:
        file_path (str): Destination file path.
        obj (object): Python object to serialize and save.
    """
    try:
        logger.info("Entered the save_object method of MainUtils class")

        # Create folder if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Save the object using Pickle
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

        logger.info("Exited the save_object method of MainUtils class")
    except Exception as e:
        logger.error("Failed to save pickle file", exc_info=True)
        raise NetworkSecurityException(e, sys)


# ================
# LOAD PICKLE FILE
# ================

def load_object(file_path: str) -> Any:
    """
    Load and deserialize a Python object from a pickle file.

    Args:
        file_path (str): Path to the pickle file.

    Returns:
        Any: Deserialized Python object.

    Raises:
        NetworkSecurityException: If file does not exist or loading fails.
    """
    try:
        if not os.path.exists(file_path):
            message = f"The file: {file_path} does not exist."
            logger.error(message)
            raise FileNotFoundError(message)

        with open(file_path, "rb") as file_obj:
            logger.info(f"Loading object from file: {file_path}")
            obj = pickle.load(file_obj)
            logger.info(f"Successfully loaded object from file: {file_path}")
            return obj

    except Exception as e:
        logger.error(f"Failed to load object from file: {file_path}", exc_info=True)
        raise NetworkSecurityException(e, sys)


# ============================
# LOAD NUMPY ARRAY PICKLE FILE
# ============================

def load_numpy_array_data(file_path: str) -> np.ndarray:
    """
    Load a numpy array stored in a binary .npy or pickle file.

    Args:
        file_path (str): Path to the numpy array file.

    Returns:
        np.ndarray: Loaded numpy array.

    Raises:
        NetworkSecurityException: If loading fails.
    """
    try:
        logger.info(f"Loading numpy array from file: {file_path}")
        with open(file_path, "rb") as file_obj:
            array = np.load(file_obj)
        logger.info(f"Successfully loaded numpy array from file: {file_path}")
        return array

    except Exception as e:
        logger.error(f"Failed to load numpy array from file: {file_path}", exc_info=True)
        raise NetworkSecurityException(e, sys)


# ===============
# EVALUATE MODELS
# ===============

def evaluate_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    models: Dict[str, Any],
    param: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """
    Perform hyperparameter tuning using GridSearchCV on each model, train with best parameters,
    evaluate on test set, and return a report containing best params and accuracy.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        X_test (np.ndarray): Testing features.
        y_test (np.ndarray): Testing labels.
        models (Dict[str, Any]): Dictionary of model name to model instance.
        param (Dict[str, Dict[str, Any]]): Dictionary of model name to hyperparameter grid.

    Returns:
        Dict[str, Dict[str, Any]]: A report dictionary containing for each model:
            - best_params: Best hyperparameters found.
            - accuracy: Accuracy score on test data.
            - comparison: List of dictionaries with true and predicted labels for inspection.

    Raises:
        NetworkSecurityException: If model evaluation fails.
    """
    try:
        report: Dict[str, Dict[str, Any]] = {}
        logger.info("Starting model evaluation and hyperparameter tuning...")

        # Iterate over each model and its parameter grid
        for model_name, model in models.items():
            logger.info(f"Evaluating model: {model_name}")

            para = param.get(model_name, {})
            logger.debug(f"Hyperparameter grid for {model_name}: {para}")

            # Setup GridSearchCV for hyperparameter tuning with 3-fold cross-validation
            gs = GridSearchCV(model, para, cv=3)

            # Fit grid search on training data to find best hyperparameters
            gs.fit(X_train, y_train)
            logger.info(f"Best parameters for {model_name}: {gs.best_params_}")

            # Update model with best parameters and retrain on full training data
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            # Predict on test data using the tuned model
            y_test_pred = model.predict(X_test)

            # Calculate accuracy score for test predictions
            accuracy = accuracy_score(y_test, y_test_pred)
            logger.info(f"Accuracy for model {model_name}: {accuracy:.4f}")

           

            # Store evaluation results for current model in the report
            report[model_name] = {
                "best_params": gs.best_params_,
                "accuracy": accuracy,
                
            }

        logger.info("Completed model evaluation for all candidates.")
        # Return complete evaluation report of all models
        return report

    except Exception as e:
        logger.error("Failed to evaluate models", exc_info=True)
        raise NetworkSecurityException(e, sys)
