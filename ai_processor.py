import os
import json
import time
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- INITIALIZATION ---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

extraction_llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-4o-mini", temperature=0)
analysis_llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-4o-mini", temperature=0.3)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# In-memory dictionary to store FAISS indexes for the current session
vector_stores = {}

# --- HELPER FUNCTIONS FOR ANALYSIS ---

def get_simple_summary(full_document_text):
    """Creates a simple, easy-to-understand summary of the entire document."""
    prompt = f"""Provide a clear, easy-to-understand summary of the following legal document. The summary should:
    1. Use plain, simple language and avoid legal jargon.
    2. Include clear headings for different sections (e.g., "Key Responsibilities").
    3. Focus on the most important aspects of the agreement.

    Document Text: --- {full_document_text} ---
    """
    return analysis_llm.invoke(prompt).content

def highlight_key_clauses(retriever):
    """Uses a retriever to find and extract specific, important clause types."""
    CLAUSE_CHECKLIST = [
        "Penalties for late payment or breach of contract",
        "Conditions for contract termination (by either party)",
        "Terms related to auto-renewal of the contract",
        "Confidentiality obligations",
        "Limitations of liability",
        "Governing law and jurisdiction for disputes"
    ]
    highlighted_clauses = {}
    
    for clause_topic in CLAUSE_CHECKLIST:
        print(f"INFO: Retrieving context for: {clause_topic}")
        retrieved_docs = retriever.invoke(clause_topic)
        
        if not retrieved_docs:
            print(f"WARN: No context found for '{clause_topic}'. Skipping.")
            highlighted_clauses[clause_topic] = "Not Found"
            continue

        context_text = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])

        prompt = f"""
        Based ONLY on the following context, extract the exact, verbatim clause or sentences that specifically address '{clause_topic}'.
        If no relevant clause is found within this context, respond with ONLY the text 'Not Found'.

        Context: --- {context_text} ---
        """
        
        print(f"INFO: Extracting clause for: {clause_topic}")
        response_content = extraction_llm.invoke(prompt).content
        result = response_content.strip()

        if "not found" not in result.lower():
            highlighted_clauses[clause_topic] = result
        else:
            highlighted_clauses[clause_topic] = "Not Found"
            
    return highlighted_clauses

def analyze_for_red_flags(clauses_text):
    """Takes a string of extracted clauses and analyzes them for risks."""
    if not clauses_text:
        return "No specific clauses were identified for risk analysis."
    prompt = f"""Based ONLY on the following clauses, summarize potential red flags or one-sided terms. Focus on terms disadvantageous to the person signing.

    Extracted Clauses: --- {clauses_text} ---
    """
    return analysis_llm.invoke(prompt).content

def get_final_assessment(full_document_text, red_flags_summary):
    """Performs a meta-analysis to generate a confidence score and recommendation."""
    prompt = f"""Based on the document's complexity and identified red flags, provide a final assessment.
    Return ONLY a valid JSON object with keys: "complexity_score" (1-10), "confidence_score" (0-100), "recommend_lawyer" (boolean), "recommendation_reason" (string).

    Document Snippet: --- {full_document_text[:4000]} ---
    Summary of Red Flags: --- {red_flags_summary} ---
    """
    assessment_json_string = analysis_llm.invoke(prompt).content
    try:
        return json.loads(assessment_json_string)
    except json.JSONDecodeError:
        return {"error": "Failed to parse the assessment."}

# --- MAIN PUBLIC FUNCTIONS ---

def setup_and_process_document(filepath):
    """Orchestrates the entire document processing and analysis workflow using FAISS."""
    # 1. Load and Chunk Document
    loader = PyMuPDFLoader(filepath)
    documents = loader.load()
    full_text = " ".join([doc.page_content for doc in documents])
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(documents)
    doc_namespace = os.path.basename(filepath)

    # 2. Create FAISS index in memory
    print(f"INFO: Creating FAISS index for {doc_namespace}...")
    try:
        vector_store = FAISS.from_documents(chunks, embeddings)
        vector_stores[doc_namespace] = vector_store # Store the index in our global dictionary
        print(f"SUCCESS: FAISS index created and stored for {doc_namespace}.")
    except Exception as e:
        print(f"ERROR: Failed to create FAISS index: {e}")
        return {"error": "Could not process the document."}

    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    # 3. Run the Full Analysis Chain
    summary = get_simple_summary(full_text)
    highlighted_clauses = highlight_key_clauses(retriever)
    
    found_clauses = {k: v for k, v in highlighted_clauses.items() if v != "Not Found"}
    clauses_text_for_analysis = "\n\n".join(f"Topic: {k}\nClause: {v}" for k, v in found_clauses.items())
    
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
    if not namespace:
        return "Error: No document namespace provided."

    # Retrieve the correct FAISS index from our in-memory store
    vector_store = vector_stores.get(namespace)
    if not vector_store:
        return "Error: Document has not been processed or session has expired."

    retriever = vector_store.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=extraction_llm, chain_type="stuff", retriever=retriever)

    response = qa_chain.invoke(query)
    return response.get('result', 'No answer found.')
