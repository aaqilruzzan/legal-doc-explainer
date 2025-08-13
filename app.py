import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Import document processing functions from the analysis module
from ai_processor import setup_and_process_document, ask_question

# --- FLASK APP SETUP ---
app = Flask(__name__)
CORS(app)  # Cross-Origin Resource Sharing configuration for frontend compatibility

# File upload directory configuration
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- API ENDPOINTS ---

@app.route('/ping', methods=['GET'])
def ping():
    """
    Health check endpoint to verify the server is running.
    Returns server status and timestamp.
    """
    from datetime import datetime
    return jsonify({
        "status": "ok",
        "message": "Legal Document Explainer API is running",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }), 200

@app.route('/analyze', methods=['POST'])
def analyze_document():
    """
    Handles PDF document upload, processing, and result delivery.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if file and file.filename.endswith('.pdf'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # Execute document processing pipeline
            analysis_results = setup_and_process_document(filepath)
            return jsonify(analysis_results), 200
        except Exception as e:
            # Error handling for processing failures
            return jsonify({"error": f"An error occurred during analysis: {str(e)}"}), 500
        finally:
            # Temporary file cleanup
            if os.path.exists(filepath):
                os.remove(filepath)
    else:
        return jsonify({"error": "Invalid file type, please upload a PDF"}), 400

@app.route('/ask', methods=['POST'])
def ask_follow_up_question():
    """
    Handles follow-up questions for processed documents.
    """
    data = request.get_json()
    query = data.get('query')
    namespace = data.get('namespace')  # Document namespace from analysis result

    if not query or not namespace:
        return jsonify({"error": "Missing 'query' or 'namespace' in request body"}), 400

    try:
        answer = ask_question(query, namespace)
        return jsonify({"answer": answer}), 200
    except Exception as e:
        return jsonify({"error": f"An error occurred while processing the question: {str(e)}"}), 500

# --- RUN THE APP ---
if __name__ == '__main__':
    # Application server configuration
    app.run(host='0.0.0.0', port=5001, debug=True)
