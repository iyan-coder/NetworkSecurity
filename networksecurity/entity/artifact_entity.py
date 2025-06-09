# -------------------------------------------------------------
# This file defines the artifact data classes for pipeline stages.
# Artifacts are structured outputs from each stage in the ML pipeline.
# They help track outputs and pass necessary data/configs downstream.
# -------------------------------------------------------------

from dataclasses import dataclass  # Provides a decorator and functions for automatically adding special methods


# ========================================================
# DataIngetionArtifact: Holds paths to split data outputs
# ========================================================
@dataclass  # Simplifies the creation of classes for storing data
class DataIngetionArtifact:
    """
    Artifact class for the data ingestion step.

    Attributes:
        trained_file_path (str): Path to the CSV file containing the training dataset.
        test_file_path (str): Path to the CSV file containing the testing dataset.
    """
    trained_file_path: str  # ✅ File path to the training dataset after split
    test_file_path: str     # ✅ File path to the testing dataset after split


# ===================================================================
# DataValidationArtifact: Holds output of the data validation process
# ===================================================================
@dataclass
class DataValidationArtifact:
    """
    Artifact class for the data validation step.

    Attributes:
        validation_status (bool): Indicates if the validation was successful (True/False).
        valid_train_file_path (str): Path to the validated training dataset.
        valid_test_file_path (str): Path to the validated testing dataset.
        invalid_train_file_path (str): Path to the invalid training dataset, if any.
        invalid_test_file_path (str): Path to the invalid testing dataset, if any.
        drift_report_file_path (str): Path to the saved data drift report (YAML).
    """
    validation_status: bool               # ✅ Was validation successful or not
    valid_train_file_path: str            # ✅ Path to valid training data
    valid_test_file_path: str             # ✅ Path to valid testing data
    invalid_train_file_path: str          # ✅ Path to invalid training data (if validation fails)
    invalid_test_file_path: str           # ✅ Path to invalid testing data (if validation fails)
    drift_report_file_path: str           # ✅ Path to YAML file storing the drift detection report



# ===========================================================================
# DataTransformationArtifact: Holds output of the data transformation process
# ===========================================================================
@dataclass
class DataTransformationArtifact:

    """
    Data class for storing artifact paths generated during the data transformation step
    of a machine learning pipeline.

    Attributes:
        transformed_object_file_path (str): Path to the serialized preprocessing object
                                            (e.g., imputer, scaler, encoder).
        transformed_train_file_path (str): Path to the preprocessed and transformed training dataset.
        transformed_test_file_path (str): Path to the preprocessed and transformed testing dataset.
    """
    # File path to the serialized preprocessing object (e.g., imputer, scaler, encoder)
    transformed_object_file_path: str

    # File path to the transformed training dataset (after preprocessing)
    transformed_train_file_path: str

    # File path to the transformed testing dataset (after preprocessing)
    transformed_test_file_path: str

# ===================================================================================
# ClassficationMetricArtifact: Data class to store key classification evaluation metrics
# ===================================================================================
@dataclass
class ClassficationMetricArtifact:
    # F1 score balances precision and recall - useful for imbalanced classification
    f1_score: float
    
    # Precision indicates how many predicted positives are actually positive
    precision_score: float
    
    # Recall indicates how many actual positives were correctly identified
    recall_score: float

    accuracy_score: float
    
    roc_auc_score: float


# =========================================================================
# ModelTrainerArtifact: Data class to encapsulate the outcome of model training
# =========================================================================
@dataclass
class ModelTrainerArtifact:
    # File path where the trained model (pickle or similar) is saved for later use
    trained_model_file_path: str
    
    # Classification metrics evaluated on the training dataset to check model fit
    train_metric_artifact: ClassficationMetricArtifact
    
    # Classification metrics evaluated on the test dataset to check generalization
    test_metric_artifact: ClassficationMetricArtifact
