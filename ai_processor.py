import os
import json
import logging
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

# Import configuration
from config import (
    OPENAI_API_KEY, EXTRACTION_MODEL, ANALYSIS_MODEL, 
    EXTRACTION_TEMPERATURE, ANALYSIS_TEMPERATURE, 
    CHUNK_SIZE, CHUNK_OVERLAP, RETRIEVAL_K
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the nested "clause" object
class ClauseDetail(BaseModel):
    heading: str = Field(description="A concise, descriptive title for the specific clause found in the context.")
    description: str = Field(description="A summary of the key terms found (max 30 words).")

# Define the main structure for the analysis output
class ClauseAnalysis(BaseModel):
    clause: ClauseDetail
    recommendation: str = Field(description="A brief, actionable recommendation (max 20 words).")
    risk: Literal["critical", "high", "medium", "low", ""] = Field(description="The assessed risk level.")
    confidence: Literal["high", "medium", "low", ""] = Field(description="The confidence level of the analysis.")

# Load environment variables from .env file
# load_dotenv()  # Now handled in config.py

# --- INITIALIZATION ---
# OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")  # Now in config.py

extraction_llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model=EXTRACTION_MODEL, temperature=EXTRACTION_TEMPERATURE)
analysis_llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model=ANALYSIS_MODEL, temperature=ANALYSIS_TEMPERATURE)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# In-memory dictionary to store FAISS indexes for the current session
vector_stores = {}

# Define clause topics for analysis
CLAUSE_TOPICS = {
    "termination": "Contract Termination",
    "financial": "Payment Terms and Financial Details",
    "liability": "Limitation of Liability and Indemnification",
    "renewal": "Auto-Renewal Clause",
    "service": "Service Level Agreement (SLA) or Scope of Work"
}

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
      "parties_involved": [ {{ "name": "Name", "role": "Role (use simple role names like 'Provider', 'Customer', 'Contractor' without additional descriptive text in parentheses)" }} ],
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
    
    try:
        logger.info("Generating document summary and key data")
        response_content = analysis_llm.invoke(prompt).content
        cleaned_json = response_content.strip().replace("```json", "").replace("```", "").strip()
        result = json.loads(cleaned_json)
        logger.info("Successfully generated summary and key data")
        return result
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse summary JSON: {str(e)}")
        return {
            "summary": "Could not generate a summary for this document.",
            "key_data_points": {"error": "Failed to extract key data points."},
            "important_contract_terms": {"error": "Failed to extract important contract terms."},
            "legal_terms_glossary": {"error": "Failed to generate legal terms glossary."}
        }
    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}")
        return {
            "summary": "Could not generate a summary for this document.",
            "key_data_points": {"error": "Failed to extract key data points."},
            "important_contract_terms": {"error": "Failed to extract important contract terms."},
            "legal_terms_glossary": {"error": "Failed to generate legal terms glossary."}
        }


def analyze_clauses_and_risks(namespace):
    """
    Retrieves and analyzes key clauses using a robust, structured output method.
    """
    if namespace not in vector_stores:
        logger.error(f"Namespace '{namespace}' not found")
        return {"error": f"Document namespace '{namespace}' has not been processed."}

    try:
        vector_store = vector_stores[namespace]
        retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVAL_K})

        # Setup for Structured Output
        json_llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY, 
            model=ANALYSIS_MODEL, 
            temperature=ANALYSIS_TEMPERATURE,
            model_kwargs={"response_format": {"type": "json_object"}}
        )

        parser = PydanticOutputParser(pydantic_object=ClauseAnalysis)

        prompt_template = PromptTemplate(
            template="""You are an expert contract analysis AI.
Analyze the context below related to the topic: {topic_query}.
Based ONLY on the context, generate a structured analysis.

{format_instructions}

CONTEXT:
---
{context_text}
---
""",
            input_variables=["topic_query", "context_text"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        chain = prompt_template | json_llm | parser

        analysis_results = {}
        for json_key, topic_query in CLAUSE_TOPICS.items():
            try:
                logger.info(f"Analyzing topic: {topic_query}")
                
                retrieved_docs = retriever.invoke(topic_query)
                context_text = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])

                response_object = chain.invoke({
                    "topic_query": topic_query,
                    "context_text": context_text
                })
                
                analysis_results[json_key] = response_object.dict()
                logger.info(f"Successfully analyzed topic: {topic_query}")
                
            except Exception as e:
                logger.error(f"Failed to analyze topic '{topic_query}': {str(e)}")
                analysis_results[json_key] = {
                    "error": f"Failed to generate AI analysis for topic: {topic_query}"
                }
                
        return analysis_results
        
    except Exception as e:
        logger.error(f"Clause analysis failed for namespace '{namespace}': {str(e)}")
        return {"error": f"Clause analysis failed: {str(e)}"}


def get_final_assessment(full_document_text, red_flags_summary):
    """Performs a meta-analysis to generate a confidence score and recommendation."""
    prompt = f"""Based on the document's complexity and identified red flags, provide a final assessment.
    Return ONLY a valid JSON object with keys: "complexity_score" (1-10), "confidence_score" (0-100), "recommend_lawyer" (boolean), "recommendation_reason" (string).

    Document Snippet: --- {full_document_text[:4000]} ---
    Summary of Red Flags: --- {red_flags_summary} ---
    """
    
    try:
        logger.info("Generating final assessment")
        assessment_json_string = analysis_llm.invoke(prompt).content
        result = json.loads(assessment_json_string)
        logger.info("Successfully generated final assessment")
        return result
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse assessment JSON: {str(e)}")
        return {"error": "Failed to parse the assessment."}
    except Exception as e:
        logger.error(f"Error generating assessment: {str(e)}")
        return {"error": f"Failed to generate assessment: {str(e)}"}

# --- MAIN PUBLIC FUNCTIONS ---

def setup_and_process_document(filepath):
    """Orchestrates the entire document processing and analysis workflow using FAISS."""
    try:
        logger.info(f"Starting document processing for: {filepath}")
        
        loader = PyMuPDFLoader(filepath)
        documents = loader.load()
        full_text = " ".join([doc.page_content for doc in documents])
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, 
            chunk_overlap=CHUNK_OVERLAP
        )
        chunks = text_splitter.split_documents(documents)
        doc_namespace = os.path.basename(filepath)

        logger.info(f"Creating FAISS index for {doc_namespace}...")
        vector_store = FAISS.from_documents(chunks, embeddings)
        vector_stores[doc_namespace] = vector_store
        logger.info(f"Successfully created FAISS index for {doc_namespace}")

        summary_data = get_summary_and_key_data(full_text)
        
        result = {
            "summary": summary_data,
            "namespace": doc_namespace
        }
        
        logger.info(f"Document processing completed successfully for: {doc_namespace}")
        return result
        
    except Exception as e:
        logger.error(f"Document processing failed: {str(e)}")
        return {"error": f"Document processing failed: {str(e)}"}

def ask_question(query, namespace):
    """Handles follow-up Q&A for a specific, already-processed document."""
    if not namespace:
        return "Error: No document namespace provided."

    vector_store = vector_stores.get(namespace)
    if not vector_store:
        return "Error: Document has not been processed or session has expired."

    try:
        retriever = vector_store.as_retriever()

        prompt_template = """
        You are a helpful assistant analyzing a document. Use the following pieces of context to answer the question at the end.
        Your answer should be based ONLY on the provided context.

        If the answer is found in the context, provide a clear and concise answer.
        If the answer is not found in the context, explicitly state "The provided document does not contain specific information on this topic."

        Context:
        {context}

        Question: {question}

        Answer:
        """

        QA_PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=extraction_llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": QA_PROMPT}
        )

        logger.info(f"Processing question: {query[:50]}...")
        response = qa_chain.invoke({"query": query})
        answer = response.get('result', 'No answer found.')
        logger.info("Successfully processed question")
        return answer
        
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        return f"Error processing question: {str(e)}"