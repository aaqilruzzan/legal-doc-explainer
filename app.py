import os
import logging
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Import configuration and processing functions
from config import UPLOAD_FOLDER, API_HOST, API_PORT, API_DEBUG, API_VERSION, MAX_FILE_SIZE
from ai_processor import setup_and_process_document, ask_question, analyze_clauses_and_risks

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- FLASK APP SETUP ---
app = Flask(__name__)
CORS(app)  # Cross-Origin Resource Sharing configuration for frontend compatibility

# File upload directory configuration
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

logger.info("Flask application initialized")

# --- API ENDPOINTS ---

@app.route('/ping', methods=['GET'])
def ping():
    """
    Health check endpoint to verify the server is running.
    Returns server status and timestamp.
    """
    try:
        return jsonify({
            "status": "ok",
            "message": "Legal Document Explainer API is running",
            "timestamp": datetime.now().isoformat(),
            "version": API_VERSION
        }), 200
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({"error": "Health check failed"}), 500

@app.route('/analyze', methods=['POST'])
def analyze_document():
    """
    Handles PDF document upload, processing, and result delivery.
    """
    filepath = None
    
    try:
        # Validate file upload
        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        if not file or not file.filename.endswith('.pdf'):
            return jsonify({"error": "Invalid file type, please upload a PDF"}), 400

        # Save file securely
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        logger.info(f"Processing document: {filename}")
        
        # Execute document processing pipeline
        analysis_results = setup_and_process_document(filepath)
        
        # Check for processing errors
        if "error" in analysis_results:
            return jsonify(analysis_results), 500
            
        logger.info(f"Document processed successfully: {filename}")
        return jsonify(analysis_results), 200
        
    except Exception as e:
        # Error handling for processing failures
        logger.error(f"Document analysis failed: {str(e)}")
        return jsonify({"error": f"An error occurred during analysis: {str(e)}"}), 500
        
    finally:
        # Temporary file cleanup
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
                logger.info(f"Cleaned up temporary file: {filepath}")
            except Exception as e:
                logger.warning(f"Failed to cleanup file {filepath}: {str(e)}")

@app.route('/highlights', methods=['POST'])
def get_highlights():
    """
    Analyzes key clauses and risks for a previously processed document.
    Requires the namespace from a previous /analyze call.
    """
    try:
        data = request.get_json()
        namespace = data.get('namespace') if data else None

        if not namespace:
            return jsonify({"error": "Missing 'namespace' in request body"}), 400

        logger.info(f"Analyzing highlights for namespace: {namespace}")
        highlights_results = analyze_clauses_and_risks(namespace)
        
        # Check for analysis errors
        if "error" in highlights_results:
            return jsonify(highlights_results), 404
            
        logger.info(f"Highlights analysis completed for: {namespace}")
        return jsonify(highlights_results), 200
        
    except Exception as e:
        logger.error(f"Highlights analysis failed: {str(e)}")
        return jsonify({"error": f"An error occurred while analyzing highlights: {str(e)}"}), 500

@app.route('/ask', methods=['POST'])
def ask_follow_up_question():
    """
    Handles follow-up questions for processed documents.
    """
    try:
        data = request.get_json()
        query = data.get('query') if data else None
        namespace = data.get('namespace') if data else None

        if not query or not namespace:
            return jsonify({"error": "Missing 'query' or 'namespace' in request body"}), 400

        logger.info(f"Processing question for namespace: {namespace}")
        answer = ask_question(query, namespace)
        
        # Check for errors in answer
        if answer.startswith("Error:"):
            return jsonify({"error": answer}), 404
            
        logger.info("Question processed successfully")
        return jsonify({"answer": answer}), 200
        
    except Exception as e:
        logger.error(f"Question processing failed: {str(e)}")
        return jsonify({"error": f"An error occurred while processing the question: {str(e)}"}), 500

# --- RUN THE APP ---
if __name__ == '__main__':
    # Application server configuration
    logger.info(f"Starting Legal Document Explainer API on {API_HOST}:{API_PORT}")
    app.run(host=API_HOST, port=API_PORT, debug=API_DEBUG)
