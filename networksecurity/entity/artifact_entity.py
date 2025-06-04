# ----------------------------------------------
# This file defines the artifact object for data ingestion.
# An artifact is an output/result from a pipeline stage that
# can be passed to the next stage. This promotes modularity.
# ----------------------------------------------

from dataclasses import dataclass  # Automatically generates __init__, __repr__, __eq__, etc.

@dataclass  # Declares a simple class to hold structured data without writing boilerplate code
class DataIngetionArtifact:
    """
    DataIngetionArtifact holds the output of the data ingestion process.
    
    Attributes:
        trained_file_path (str): File path to the training dataset saved after the train-test split.
        test_file_path (str): File path to the testing dataset saved after the train-test split.
    """

    trained_file_path: str  # ✅ Path to the CSV file containing the training data
    test_file_path: str     # ✅ Path to the CSV file containing the testing data
