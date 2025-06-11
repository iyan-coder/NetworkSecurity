# Importing necessary classes and functions
from networksecurity.entity.artifact_entity import ClassficationMetricArtifact  # Custom class to store metric results
from networksecurity.exception.exception import NetworkSecurityException  # Custom exception handler
from networksecurity.logger.logger import logger  # Logger to track events and errors
import os
import sys
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score  # Sklearn metrics
def get_classification_score(y_true, y_pred, y_proba=None) -> ClassficationMetricArtifact:
    """
    This function calculates how well a model is performing using common classification metrics.

    Args:
        y_true (np.ndarray): The actual/true labels (e.g., [0, 1, 1, 0]).
        y_pred (np.ndarray): The predicted labels from the model.
        y_proba (np.ndarray, optional): The predicted probabilities for class 1 (used for AUC).

    Returns:
        ClassficationMetricArtifact: A custom object that holds all the calculated metric scores.
    """
    try:
        # F1 Score is the balance between precision and recall.
        model_f1_score = f1_score(y_true, y_pred)

        # Recall shows how many actual positives were correctly identified.
        model_recall_score = recall_score(y_true, y_pred)

        # Precision shows how many predicted positives were correct.
        model_precision_score = precision_score(y_true, y_pred)

        # Accuracy tells us what fraction of predictions were correct.
        model_accuracy = accuracy_score(y_true, y_pred)

        # AUC (Area Under Curve) tells us how well the model separates classes (only calculated if probabilities are provided).
        model_roc_auc = roc_auc_score(y_true, y_proba) if y_proba is not None else None

        # Create an object to store all the scores in a structured way
        classification_metric = ClassficationMetricArtifact(
            f1_score=model_f1_score,
            precision_score=model_precision_score,
            recall_score=model_recall_score,
            accuracy_score=model_accuracy,
            roc_auc_score=model_roc_auc
        )

        # Return the final result object
        return classification_metric

    except Exception as e:
        # Log the error and raise a custom exception if anything goes wrong
        logger.error("Failed to get classification metrics", exc_info=True)
        raise NetworkSecurityException(e, sys)

