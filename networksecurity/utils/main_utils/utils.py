import yaml
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logger.logger import logger
import os
import sys
import numpy as np
import pandas as pd
import dill
import pickle

def read_yaml_file(file_path: str) ->dict:
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except:
        logger.error("Could not read_yaml_file", exc_info=True)
        raise NetworkSecurityException(e,sys)
    

# def write_yaml_file(file_path: str, content: object, replace:bool =False) -> None:
#     try:
#         if replace:
#             if os.path.exists(file_path):
#                 os.remove(file_path)
#         os.makedirs(os.path.dirname(file_path), exist_ok=True)
#         with open(file_path, "w")as file:
#             yaml.dump(content,file)


def write_yaml_file(file_path: str, content: dict) -> None:
    try:
        with open(file_path, "w") as file:
          yaml.dump(content, file, default_flow_style=False, sort_keys=False, Dumper=yaml.SafeDumper)


    except Exception as e:
        logger.error("Failed to write yaml file")
        raise NetworkSecurityException(e,sys)