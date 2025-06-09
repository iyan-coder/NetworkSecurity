import importlib
import yaml
from typing import Dict

def load_model_config(config_path: str) -> Dict:
    """
    Load model and hyperparameter configuration from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file containing model definitions.

    Returns:
        Tuple[Dict[str, object], Dict[str, dict]]: 
            - models: A dictionary mapping model names to instantiated model objects.
            - params: A dictionary mapping model names to their hyperparameter grids/dictionaries.

    How it works:
    1. Reads the YAML file which should contain a section like:
       models:
         RandomForest:
           class_path: sklearn.ensemble.RandomForestClassifier
           params:
             n_estimators: [100, 200]
             max_depth: [10, 20]
         SVM:
           class_path: sklearn.svm.SVC
           params:
             kernel: ['linear', 'rbf']

    2. For each model entry:
       - It extracts the Python module path and class name from the 'class_path' string.
       - Dynamically imports the module using importlib.
       - Instantiates the model class with default parameters.
       - Retrieves the hyperparameter grid/dictionary if present.

    This design allows:
    - Easy addition or removal of models and their hyperparameters by editing a YAML file.
    - Flexibility to use any model class available in Python packages without hardcoding imports.
    - Clean separation of configuration from code.

    Example usage:
    models, params = load_model_config('model_config.yaml')
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    models = {}
    params = {}

    for name, model_info in config['models'].items():
        # Split the full class path into module and class names
        module_path, class_name = model_info['class_path'].rsplit('.', 1)
        
        # Dynamically import the module and get the class object
        model_class = getattr(importlib.import_module(module_path), class_name)
        
        # Instantiate the model with default parameters
        models[name] = model_class()
        
        # Retrieve hyperparameters dictionary, default to empty dict if not provided
        params[name] = model_info.get('params', {})

    return models, params
