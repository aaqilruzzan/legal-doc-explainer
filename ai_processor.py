import os
import json
import pinecone
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- INITIALIZATION ---
# Configuration for API keys and environment variables
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENV = os.environ.get("PINECONE_ENV")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Initialize Pinecone
pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENV
)

# Define a unique name for your Pinecone index
INDEX_NAME = "codestorm-legal-docs-v1"

# Initialize LLMs and Embeddings
# Using different models for different tasks can be cost-effective.
# gpt-3.5-turbo is fast and cheap for extraction.
# gpt-4 is better for nuanced analysis.
extraction_llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name="gpt-3.5-turbo", temperature=0)
analysis_llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name="gpt-4", temperature=0.3)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# --- HELPER FUNCTIONS FOR ANALYSIS ---

def get_simple_summary(full_document_text):
    """Creates a simple, easy-to-understand summary of the entire document."""
    # Create a prompt for document summarization
    prompt = f"""Provide a clear, easy-to-understand summary of the following legal document. The summary should:
    1. Use plain, simple language.
    2. Avoid legal jargon.
    3. Include clear headings for different sections.
    4. Focus on the most important aspects of the agreement.

    Document Text:
    ---
    {full_document_text}
    ---
    """
    return analysis_llm.predict(prompt)

def highlight_key_clauses(retriever):
    """Uses a retriever to find and extract specific, important clause types."""
    # Define a checklist of important clause types to look for
    CLAUSE_CHECKLIST = [
        "Penalties for late payment or breach of contract",
        "Conditions for contract termination (by either party)",
        "Terms related to auto-renewal of the contract",
        "Confidentiality obligations",
        "Limitations of liability",
        "Governing law and jurisdiction for disputes"
    ]
    highlighted_clauses = {}
    qa_chain = RetrievalQA.from_chain_type(llm=extraction_llm, chain_type="stuff", retriever=retriever)

    # Process each clause type in the checklist
    for clause_topic in CLAUSE_CHECKLIST:
        prompt = f"Extract the exact clause or sentences from the document that specifically address '{clause_topic}'. If not found, respond with 'Not Found'."
        response = qa_chain.run(prompt)
        if "not found" not in response.lower():
            highlighted_clauses[clause_topic] = response
    return highlighted_clauses

def analyze_for_red_flags(clauses_text):
    """Takes a string of extracted clauses and analyzes them for risks."""
    # Handle case where no clauses were found
    if not clauses_text:
        return "No specific clauses were identified for risk analysis."
    
    # Create a prompt for risk analysis
    prompt = f"""Based ONLY on the following clauses, summarize potential red flags or one-sided terms. Focus on terms that may be disadvantageous to the person signing.

    Extracted Clauses:
    ---
    {clauses_text}
    ---
    """
    return analysis_llm.predict(prompt)

def get_final_assessment(full_document_text, red_flags_summary):
    """Performs a meta-analysis to generate a confidence score and recommendation."""
    # Create a prompt for final assessment
    prompt = f"""Based on the complexity of the original document and the severity of identified red flags, provide a final assessment.
    Return ONLY a valid JSON object with keys: "complexity_score" (1-10), "confidence_score" (0-100), "recommend_lawyer" (boolean), "recommendation_reason" (string).

    Original Document Snippet:
    ---
    {full_document_text[:4000]}
    ---
    Summary of Red Flags:
    ---
    {red_flags_summary}
    ---
    """
    assessment_json_string = analysis_llm.predict(prompt)
    try:
        return json.loads(assessment_json_string)
    except json.JSONDecodeError:
        return {"error": "Failed to parse the assessment."}

# --- MAIN PUBLIC FUNCTIONS ---

def setup_and_process_document(filepath):
    """
    Orchestrates the entire document processing and analysis workflow.
    This is the main function to be called by the Flask app.
    """
    # 1. Load and Chunk Document
    loader = PyMuPDFLoader(filepath)
    documents = loader.load()
    full_text = " ".join([doc.page_content for doc in documents])
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(documents)

    # 2. Create or Update Pinecone Index
    # Using the filepath as a namespace allows us to keep multiple docs in one index
    doc_namespace = os.path.basename(filepath)
    if INDEX_NAME not in pinecone.list_indexes():
        pinecone.create_index(name=INDEX_NAME, dimension=1536, metric='cosine')

    Pinecone.from_documents(chunks, embeddings, index_name=INDEX_NAME, namespace=doc_namespace)
    vector_store = Pinecone.from_existing_index(index_name=INDEX_NAME, embedding=embeddings, namespace=doc_namespace)
    retriever = vector_store.as_retriever()

    # 3. Run the Analysis Chain
    summary = get_simple_summary(full_text)
    highlighted_clauses = highlight_key_clauses(retriever)
    clauses_text_for_analysis = "\n\n".join(f"Topic: {k}\nClause: {v}" for k, v in highlighted_clauses.items())
    red_flags = analyze_for_red_flags(clauses_text_for_analysis)
    final_assessment = get_final_assessment(full_text, red_flags)

    # 4. Compile and Return Results
    return {
        "summary": summary,
        "namespace": doc_namespace,
        "highlighted_clauses": highlighted_clauses,
        "red_flags_summary": red_flags,
        "final_assessment": final_assessment
    }

def ask_question(query, namespace):
    """Handles follow-up Q&A for a specific, already-processed document."""
    # Validate namespace parameter
    if not namespace:
        return "Error: No document namespace provided."

    # Create retriever for the specified document
    vector_store = Pinecone.from_existing_index(index_name=INDEX_NAME, embedding=embeddings, namespace=namespace)
    retriever = vector_store.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=extraction_llm, chain_type="stuff", retriever=retriever)

    return qa_chain.run(query)
