"""
Main entry point for the refactored Legal Document Explainer application.
"""
import sys
import os

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.app import main

if __name__ == '__main__':
    main()
