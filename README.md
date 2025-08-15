# Legal Document Explainer

A Flask-based web application that uses AI to analyze legal documents, extract key information, and provide intelligent question-answering capabilities.

## 🎯 Features

- **PDF Document Analysis**: Upload and analyze legal documents (contracts, agreements, etc.)
- **Key Information Extraction**: Automatically extract parties, dates, financial terms, and deadlines
- **Clause Analysis**: Identify and analyze critical clauses (termination, liability, renewal, etc.)
- **Risk Assessment**: Evaluate potential risks and provide recommendations
- **Interactive Q&A**: Ask questions about uploaded documents and get contextual answers
- **Plain Language Summaries**: Convert complex legal language into easy-to-understand summaries

## 🏗️ Architecture

```
legal-doc-explainer/
├── app.py              # Flask web application
├── ai_processor.py     # AI analysis and document processing
├── config.py           # Configuration settings
├── utils.py            # Utility functions
├── requirements.txt    # Python dependencies
├── .env               # Environment variables (create this)
└── uploads/           # Temporary file storage (auto-created)
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Virtual environment (recommended)

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/aaqilruzzan/legal-doc-explainer.git
   cd legal-doc-explainer
   ```

2. **Create and activate virtual environment**

   ```bash
   python -m venv venv

   # On Windows
   venv\Scripts\activate

   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**

   Create a `.env` file in the root directory:

   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ```

5. **Run the application**

   ```bash
   python app.py
   ```

6. **Access the API**

   The application will be available at `http://localhost:5001`

## 📡 API Endpoints

### Health Check

```http
GET /ping
```

Returns server status and version information.

### Document Analysis

```http
POST /analyze
Content-Type: multipart/form-data

file: [PDF file]
```

Uploads and analyzes a PDF document, returning summary and key data points.

### Clause Analysis

```http
POST /highlights
Content-Type: application/json

{
  "namespace": "document_filename.pdf"
}
```

Analyzes key clauses and risks for a previously processed document.

### Question Answering

```http
POST /ask
Content-Type: application/json

{
  "query": "What is the termination notice period?",
  "namespace": "document_filename.pdf"
}
```

Ask questions about a processed document and get contextual answers.

## 🔧 Configuration

The application can be configured through `config.py`:

```python
# Model Settings
EXTRACTION_MODEL = "gpt-4o-mini"
ANALYSIS_MODEL = "gpt-4o-mini"

# Document Processing
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# File Upload
UPLOAD_FOLDER = 'uploads'
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

# API Settings
API_HOST = "0.0.0.0"
API_PORT = 5001
API_DEBUG = True
```

## 📝 Usage Examples

### Analyzing a Document

```bash
curl -X POST \
  http://localhost:5001/analyze \
  -F "file=@contract.pdf"
```

### Getting Clause Analysis

```bash
curl -X POST \
  http://localhost:5001/highlights \
  -H "Content-Type: application/json" \
  -d '{"namespace": "contract.pdf"}'
```

### Asking Questions

```bash
curl -X POST \
  http://localhost:5001/ask \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the payment terms?",
    "namespace": "contract.pdf"
  }'
```

## 🧠 How It Works

1. **Document Upload**: PDF files are uploaded via the `/analyze` endpoint
2. **Text Extraction**: PyMuPDF extracts text content from the PDF
3. **Chunking**: Documents are split into manageable chunks for processing
4. **Vector Storage**: Text chunks are converted to embeddings and stored in FAISS
5. **AI Analysis**: OpenAI models analyze the document and extract key information
6. **Structured Output**: Results are returned in structured JSON format
7. **Q&A**: Users can ask follow-up questions using the stored embeddings

## 🔍 Document Processing Pipeline

```
PDF Upload → Text Extraction → Chunking → Embeddings → Vector Store
                                                            ↓
AI Analysis ← Retrieval ← Query Processing ← User Questions
```

## 🛡️ Security Features

- **File Validation**: Only PDF files are accepted
- **Secure Filenames**: Uploaded files are sanitized to prevent path traversal
- **Temporary Storage**: Files are automatically deleted after processing
- **Size Limits**: Maximum file size of 16MB
- **No Permanent Storage**: Documents are not stored long-term

## 🎛️ Environment Variables

| Variable         | Required | Description                          |
| ---------------- | -------- | ------------------------------------ |
| `OPENAI_API_KEY` | Yes      | Your OpenAI API key                  |
| `FLASK_ENV`      | No       | Environment (development/production) |

## 📊 Response Formats

### Analysis Response

```json
{
  "summary": {
    "summary": "Plain language summary...",
    "key_data_points": {
      "parties_involved": [{"name": "ABC Corp", "role": "Provider"}],
      "contract_period": {"start_date": "2024-01-01", "end_date": "2025-01-01"},
      "financial_terms": ["$5,000 monthly fee"],
      "key_deadlines": ["30-day notice required"]
    },
    "important_contract_terms": {...},
    "legal_terms_glossary": {...}
  },
  "namespace": "document.pdf"
}
```

### Clause Analysis Response

```json
{
  "termination": {
    "clause": {
      "heading": "Termination Clause",
      "description": "Contract can be terminated with 30 days notice"
    },
    "recommendation": "Review termination conditions carefully",
    "risk": "medium",
    "confidence": "high"
  },
  "financial": {...},
  "liability": {...}
}
```

## 🚨 Error Handling

The API returns structured error responses:

```json
{
  "error": "Error description",
  "details": "Additional error details (if available)"
}
```

Common HTTP status codes:

- `200`: Success
- `400`: Bad Request (invalid input)
- `404`: Not Found (document not processed)
- `500`: Internal Server Error

## 🧪 Testing

Test the API endpoints:

```bash
# Health check
curl http://localhost:5001/ping

# Upload a test document
curl -X POST -F "file=@test_contract.pdf" http://localhost:5001/analyze
```

## 📋 Dependencies

Key dependencies include:

- **Flask**: Web framework
- **LangChain**: Document processing and AI chains
- **OpenAI**: AI models for analysis
- **FAISS**: Vector similarity search
- **PyMuPDF**: PDF text extraction
- **Pydantic**: Data validation

## 🔄 Development Workflow

1. **Make changes** to the code
2. **Test locally** using the provided endpoints
3. **Check logs** for any errors or warnings
4. **Validate responses** match expected format

## 📈 Performance Considerations

- **File Size**: Limited to 16MB for performance
- **Processing Time**: Varies with document size and complexity
- **Memory Usage**: Documents are processed in chunks to manage memory
- **Concurrent Requests**: Flask development server handles one request at a time

## 🔐 Production Deployment

For production deployment:

1. Set `API_DEBUG = False` in config
2. Use a production WSGI server (e.g., Gunicorn)
3. Set up proper logging
4. Configure reverse proxy (e.g., Nginx)
5. Implement rate limiting
6. Add authentication if needed

## 🐛 Troubleshooting

### Common Issues

1. **"OPENAI_API_KEY not found"**

   - Ensure `.env` file exists with valid API key

2. **"Import errors"**

   - Activate virtual environment: `venv\Scripts\activate`
   - Install dependencies: `pip install -r requirements.txt`

3. **"File upload failed"**

   - Check file is PDF format
   - Ensure file size is under 16MB

4. **"Processing timeout"**
   - Large documents may take longer
   - Check OpenAI API rate limits

### Debug Mode

Enable detailed logging by setting `API_DEBUG = True` in config.py.

## 📞 Support

For issues or questions:

1. Check the troubleshooting section above
2. Review the logs for error details
3. Ensure all dependencies are properly installed
4. Verify your OpenAI API key is valid

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 🔮 Future Enhancements

- Multi-language support
- Batch document processing
- Document comparison features
- Advanced risk scoring
- Integration with legal databases
- Document template generation
