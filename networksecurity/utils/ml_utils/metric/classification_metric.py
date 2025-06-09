from networksecurity.entity.artifact_entity import ClassficationMetricArtifact
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logger.logger import logger
import os
import sys
from sklearn.metrics import f1_score, precision_score, recall_score,roc_auc_score, accuracy_score

def get_classification_score(y_true, y_pred, y_proba=None) -> ClassficationMetricArtifact:
    """
    Calculate classification metrics including f1, precision, recall, accuracy, and ROC AUC.

    Args:
        y_true (np.ndarray): True class labels.
        y_pred (np.ndarray): Predicted class labels.
        y_proba (np.ndarray, optional): Predicted probabilities for positive class.

    Returns:
        ClassficationMetricArtifact: Object containing computed metrics.
    """
    try:
        model_f1_score = f1_score(y_true, y_pred)
        model_recall_score = recall_score(y_true, y_pred)
        model_precision_score = precision_score(y_true, y_pred)
        model_accuracy = accuracy_score(y_true, y_pred)

        # Compute ROC AUC only if probability scores are provided
        model_roc_auc = roc_auc_score(y_true, y_proba) if y_proba is not None else None

        classification_metric = ClassficationMetricArtifact(
            f1_score=model_f1_score,
            precision_score=model_precision_score,
            recall_score=model_recall_score,
            accuracy_score=model_accuracy,
            roc_auc_score=model_roc_auc
        )

        return classification_metric

    except Exception as e:
        logger.error("Failed to get classification metrics", exc_info=True)
        raise NetworkSecurityException(e, sys)
