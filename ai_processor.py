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

def get_summary_and_key_data(full_document_text):
    """
    Extracts a summary, key data points, important contract terms, and a glossary in a single LLM call.
    """
    prompt = f"""
    Analyze the following legal document and provide four things in a single JSON object:
    1. A summary, generated according to the specific instructions provided.
    2. A structured extraction of key data points.
    3. A structured extraction of important contract terms.
    4. A glossary of important legal terms found in the document.

    Return ONLY a valid JSON object with four top-level keys: "summary", "key_data_points", "important_contract_terms", and "legal_terms_glossary". Do not add any text before or after the JSON.

    INSTRUCTIONS FOR THE "summary" VALUE:
    Provide a clear, easy-to-understand summary of the document. The summary MUST:
    - Use plain, simple language and avoid legal jargon.
    - Include clear headings for different sections and number them (e.g., "1. Key Responsibilities"). Do not number sub-headings or points.
    
    Example for the "summary" value:
    "summary": "1. What This Agreement Is About\\nThis is a service agreement where ABC Company agrees to provide marketing services to XYZ Corporation for a period of one year.\\n\\n2. Key Responsibilities\\nAs the Client, you are responsible for providing necessary materials and feedback in a timely manner. Payments are due on the first of each month."

    INSTRUCTIONS FOR THE "key_data_points" VALUE:
    This should be an object with the following structure. Be as specific as possible. If a value is not found, use an empty string "" or an empty list [].
    {{
      "parties_involved": [ {{ "name": "Name", "role": "Role" }} ],
      "contract_period": {{
        "start_date": "Extract the specific start date in YYYY-MM-DD format or leave blank if not found.",
        "end_date": "Extract the specific end date in YYYY-MM-DD format or leave blank if not found.",
        "term_description": "Provide a specific description, e.g., 'A 2-year initial term with an option for one 12-month renewal.'"
      }},
      "financial_terms": [
        "Extract specific amounts, fees, and percentages. e.g., '$5,000 monthly subscription fee', '1.5% late fee on overdue payments'"
      ],
      "key_deadlines": [ "e.g., 30-day termination notice" ]
    }}

    INSTRUCTIONS FOR THE "important_contract_terms" VALUE:
    This should be an object containing key-value pairs for high-level contract terms. If a term is not found, use an empty string "".
    Example:
    {{
      "Service Scope": "Consulting services as defined in Exhibit A",
      "Confidentiality": "Standard NDA provisions apply",
      "Governing Law": "Delaware State Law",
      "Intellectual Property": "Work product ownership defined"
    }}

    INSTRUCTIONS FOR THE "legal_terms_glossary" VALUE:
    This should be an object containing key-value pairs. The key should be 5 complex legal terms found in the document, and the value should be its simple definition.
    Example:
    {{
        "Force Majeure": "Unforeseeable circumstances that prevent a party from fulfilling a contract.",
        "Indemnification": "Protection against financial loss, typically through compensation."
    }}

    Document Text to Analyze:
    ---
    {full_document_text}
    ---
    """
    response_content = analysis_llm.invoke(prompt).content
    try:
        cleaned_json = response_content.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(cleaned_json)
    except json.JSONDecodeError:
        print("ERROR: Failed to parse summary and key data points JSON from LLM.")
        return {
            "summary": "Could not generate a summary for this document.",
            "key_data_points": {"error": "Failed to extract key data points."},
            "important_contract_terms": {"error": "Failed to extract important contract terms."},
            "legal_terms_glossary": {"error": "Failed to generate legal terms glossary."}
        }



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
    summary = get_summary_and_key_data(full_text)
    # highlighted_clauses = highlight_key_clauses(retriever)
    
    # found_clauses = {k: v for k, v in highlighted_clauses.items() if v != "Not Found"}
    # clauses_text_for_analysis = "\n\n".join(f"Topic: {k}\nClause: {v}" for k, v in found_clauses.items())
    
    # red_flags = analyze_for_red_flags(clauses_text_for_analysis)
    # final_assessment = get_final_assessment(full_text, red_flags)

    # 4. Compile and Return Results
    return {
        "summary": summary,
       
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
