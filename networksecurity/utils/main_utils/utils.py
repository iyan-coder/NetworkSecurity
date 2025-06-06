import yaml
from networksecurity.exception.exception import NetworkSecurityException
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
