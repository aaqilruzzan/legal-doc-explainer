"""
Utility functions for the Legal Document Explainer.
"""
import os
import logging
from werkzeug.utils import secure_filename
import json
from typing import Dict, Any

logger = logging.getLogger(__name__)


def validate_pdf_file(file):
    """
    Validate uploaded PDF file.
    
    Args:
        file: Uploaded file object
        
    Returns:
        tuple: (is_valid, error_message)
    """
    if not file:
        return False, "No file provided"
    
    if file.filename == '':
        return False, "No file selected"
    
    if not file.filename.lower().endswith('.pdf'):
        return False, "Only PDF files are supported"
    
    return True, ""


def safe_file_cleanup(filepath):
    """
    Safely remove a file with error handling.
    
    Args:
        filepath: Path to file to remove
        
    Returns:
        bool: True if successfully removed or file doesn't exist
    """
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
            logger.info(f"Cleaned up file: {filepath}")
        return True
    except Exception as e:
        logger.warning(f"Failed to cleanup file {filepath}: {str(e)}")
        return False


def get_secure_filename(original_filename):
    """
    Get a secure filename for upload.
    
    Args:
        original_filename: Original filename from upload
        
    Returns:
        str: Secure filename
    """
    return secure_filename(original_filename)


def log_error_and_return(error_msg, status_code=500):
    """
    Log an error and return a formatted error response.
    
    Args:
        error_msg: Error message to log and return
        status_code: HTTP status code
        
    Returns:
        tuple: (error_dict, status_code)
    """
    logger.error(error_msg)
    return {"error": error_msg}, status_code


def parse_json_response(response_content: str) -> Dict[str, Any]:
    """Parse JSON response from LLM, handling common formatting issues."""
    try:
        cleaned_json = response_content.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(cleaned_json)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {str(e)}")
        return {"error": f"Failed to parse response: {str(e)}"}

def create_error_response(error_message: str) -> Dict[str, Any]:
    """Create a standardized error response."""
    return {
        "summary": "Could not generate a summary for this document.",
        "key_data_points": {"error": "Failed to extract key data points."},
        "important_contract_terms": {"error": "Failed to extract important contract terms."},
        "legal_terms_glossary": {"error": "Failed to generate legal terms glossary."}
    }

def safe_execute(func, *args, **kwargs):
    """Execute function with error handling and logging."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.error(f"Error executing {func.__name__}: {str(e)}")
        raise
