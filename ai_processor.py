import os
import json
import pinecone
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import Pinecone as PineconeVectorStore # Aliased to avoid name conflicts
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- INITIALIZATION ---
# Load all keys and settings from environment variables
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
INDEX_NAME = "codestorm-legal-docs-v1"

# Initialize the Pinecone client. It automatically uses the PINECONE_API_KEY from the environment.
pc = pinecone.Pinecone()

# Initialize LLMs and Embeddings
extraction_llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-4o-mini", temperature=0)
analysis_llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-4o-mini", temperature=0.3)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

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
        "Confidentiality obligations", "Limitations of liability",
        "Governing law and jurisdiction for disputes"
    ]
    highlighted_clauses = {}
    qa_chain = RetrievalQA.from_chain_type(llm=extraction_llm, chain_type="stuff", retriever=retriever)
    for clause_topic in CLAUSE_CHECKLIST:
        prompt = f"Extract the exact clause from the document that specifically addresses '{clause_topic}'. If not found, respond with 'Not Found'."
        response = qa_chain.invoke(prompt)
        # Access the result correctly from the dictionary and always store the result
        result = response.get('result', '').strip()
        highlighted_clauses[clause_topic] = result
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
    """Orchestrates the entire document processing and analysis workflow."""
    # 1. Load and Chunk Document
    loader = PyMuPDFLoader(filepath)
    documents = loader.load()
    full_text = " ".join([doc.page_content for doc in documents])
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(documents)

    # 2. Create or Update Pinecone Index
    doc_namespace = os.path.basename(filepath)
    # The new way to check if an index exists
    if INDEX_NAME not in pc.list_indexes().names():
        from pinecone import ServerlessSpec
        pc.create_index(
            name=INDEX_NAME,
            dimension=1536, # Standard for OpenAI embeddings
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )

    # Use the aliased PineconeVectorStore class
    vector_store = PineconeVectorStore.from_documents(chunks, embeddings, index_name=INDEX_NAME, namespace=doc_namespace)
    retriever = vector_store.as_retriever()

    # 3. Run the Full Analysis Chain
    summary = get_simple_summary(full_text)
    highlighted_clauses = highlight_key_clauses(retriever)
    # clauses_text_for_analysis = "\n\n".join(f"Topic: {k}\nClause: {v}" for k, v in highlighted_clauses.items())
    # red_flags = analyze_for_red_flags(clauses_text_for_analysis)
    # final_assessment = get_final_assessment(full_text, red_flags)

    # 4. Compile and Return Results
    return {
        "summary": summary,
        "namespace": doc_namespace,
        "highlighted_clauses": highlighted_clauses,
        # "red_flags_summary": red_flags,
        # "final_assessment": final_assessment
    }

def ask_question(query, namespace):
    """Handles follow-up Q&A for a specific, already-processed document."""
    if not namespace:
        return "Error: No document namespace provided."

    # Use the aliased PineconeVectorStore class
    vector_store = PineconeVectorStore.from_existing_index(index_name=INDEX_NAME, embedding=embeddings, namespace=namespace)
    retriever = vector_store.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=extraction_llm, chain_type="stuff", retriever=retriever)

    # Use .invoke for the new LangChain versions and access the result
    response = qa_chain.invoke(query)
    return response.get('result', 'No answer found.')
