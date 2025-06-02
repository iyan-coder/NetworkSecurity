# sys gives access to system-specific parameters and functions (used for getting exception traceback info)
import sys

# Optional: traceback module helps if you want to extract/print more complex stack traces (not used directly here)
import traceback

# Import logger
from networksecurity.logger.logger import logger

# Define a custom exception by extending the built-in Exception class
class NetworkSecurityException(Exception):
    """
    Custom exception class for ML systems related to network security.

    It provides detailed error reporting with file name, line number,
    and the actual error message, making it easier to debug complex pipelines.
    """

    def __init__(self, error_message: Exception, error_detail: sys):
        """
        Constructor for the NetworkSecurityException.

        Args:
            error_message (Exception): The original exception that was raised.
            error_detail (sys): The sys module (so we can use sys.exc_info()).
        """
        # Call base Exception class constructor with string version of the original error
        super().__init__(str(error_message))

        # Create a detailed error message using helper method
        self.error_message = self.get_detailed_error_message(error_message, error_detail)

    def get_detailed_error_message(self, error: Exception, error_detail: sys) -> str:
        """
        Builds a detailed error message including file name and line number.

        Args:
            error (Exception): The actual error object raised.
            error_detail (sys): The sys module to access traceback info.

        Returns:
            str: A descriptive error message showing where and what went wrong.
        """
        # Extract exception type, exception object, and traceback
        _, _, exc_tb = error_detail.exc_info()

        # Get the file name where the exception occurred
        file_name = exc_tb.tb_frame.f_code.co_filename

        # Get the line number where the exception occurred
        line_number = exc_tb.tb_lineno

        # Return a well-formatted error string
        return f"[ERROR] {error} | File: {file_name}, Line: {line_number}"

    def __str__(self) -> str:
        """
        When you print the exception or convert it to string, show the detailed error message.
        """
        return self.error_message

# if __name__ == "__main__":
#     try:
#         logger.info("Enter the try block")
#         x = 1 / 0  # Force an error
#     except Exception as e:
#         logger.error("Exception occurred", exc_info=True)
#         raise NetworkSecurityException(e, sys)
        