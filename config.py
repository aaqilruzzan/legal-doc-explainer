"""
Simple configuration for the Legal Document Explainer.
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Configuration class for the Legal Document Explainer."""
    
    # API Configuration
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

    # Model Settings
    EXTRACTION_MODEL = "gpt-4o-mini"
    ANALYSIS_MODEL = "gpt-4o-mini"
    EXTRACTION_TEMPERATURE = 0
    ANALYSIS_TEMPERATURE = 0.3

    # Document Processing
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 150
    K_DOCUMENTS = 5
    MAX_SUMMARY_LENGTH = 8000

    # File Upload
    UPLOAD_FOLDER = 'uploads'
    MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

    # API Settings
    API_HOST = "0.0.0.0"
    API_PORT = 5001
    API_DEBUG = True
    API_VERSION = "1.0.0"


# Backward compatibility - keep module-level variables
OPENAI_API_KEY = Config.OPENAI_API_KEY
EXTRACTION_MODEL = Config.EXTRACTION_MODEL
ANALYSIS_MODEL = Config.ANALYSIS_MODEL
EXTRACTION_TEMPERATURE = Config.EXTRACTION_TEMPERATURE
ANALYSIS_TEMPERATURE = Config.ANALYSIS_TEMPERATURE
CHUNK_SIZE = Config.CHUNK_SIZE
CHUNK_OVERLAP = Config.CHUNK_OVERLAP
RETRIEVAL_K = Config.K_DOCUMENTS
UPLOAD_FOLDER = Config.UPLOAD_FOLDER
MAX_FILE_SIZE = Config.MAX_FILE_SIZE
API_HOST = Config.API_HOST
API_PORT = Config.API_PORT
API_DEBUG = Config.API_DEBUG
API_VERSION = Config.API_VERSION
