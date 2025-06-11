import importlib
import yaml
from typing import Dict, Tuple

def load_model_config(config_path: str) -> Tuple[Dict[str, object], Dict[str, dict]]:
    """
    Load models and their hyperparameter configurations from a YAML file.

    Args:
        config_path (str): Path to YAML configuration file.

    Returns:
        Tuple:
            - models: Dict of instantiated model objects.
            - params: Dict of hyperparameter search grids.
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

        models = {}
        params = {}

        if 'models' not in config:
            raise ValueError("YAML config must contain a 'models' section.")

        for name, model_info in config['models'].items():
            if 'class_path' not in model_info:
                raise ValueError(f"Missing 'class_path' for model '{name}'")

            # Split path into module and class name
            module_path, class_name = model_info['class_path'].rsplit('.', 1)

            # Import and instantiate the model
            model_class = getattr(importlib.import_module(module_path), class_name)
            models[name] = model_class()

            # Add hyperparameter grid (defaults to empty if not provided)
            params[name] = model_info.get('params', {})

        return models, params

    except Exception as e:
        raise RuntimeError(f"Failed to load model config from {config_path}: {e}")
