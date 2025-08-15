import os
import logging
from typing import Dict, Any, Optional
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser

from config import (
    OPENAI_API_KEY, EXTRACTION_MODEL, ANALYSIS_MODEL, 
    EXTRACTION_TEMPERATURE, ANALYSIS_TEMPERATURE, 
    CHUNK_SIZE, CHUNK_OVERLAP, RETRIEVAL_K
)
from constants import (
    CLAUSE_TOPICS, SUMMARY_PROMPT_TEMPLATE, 
    QA_PROMPT_TEMPLATE, CLAUSE_ANALYSIS_TEMPLATE
)
from utils import parse_json_response, create_error_response, safe_execute
from schemas import ClauseDetail, ClauseAnalysis

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize AI models
extraction_llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY, 
    model=EXTRACTION_MODEL, 
    temperature=EXTRACTION_TEMPERATURE
)
analysis_llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY, 
    model=ANALYSIS_MODEL, 
    temperature=ANALYSIS_TEMPERATURE
)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# In-memory storage
vector_stores: Dict[str, FAISS] = {}

class DocumentProcessor:
    """Handles document loading and vector store creation."""
    
    @staticmethod
    def load_and_split_document(filepath: str) -> tuple[str, list]:
        """Load PDF and split into chunks."""
        loader = PyMuPDFLoader(filepath)
        documents = loader.load()
        full_text = " ".join([doc.page_content for doc in documents])
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, 
            chunk_overlap=CHUNK_OVERLAP
        )
        chunks = text_splitter.split_documents(documents)
        
        return full_text, chunks
    
    @staticmethod
    def create_vector_store(chunks: list, namespace: str) -> FAISS:
        """Create and store FAISS vector store."""
        vector_store = FAISS.from_documents(chunks, embeddings)
        vector_stores[namespace] = vector_store
        logger.info(f"Successfully created FAISS index for {namespace}")
        return vector_store

class AIAnalyzer:
    """Handles AI-powered analysis tasks."""
    
    @staticmethod
    def get_summary_and_key_data(full_document_text: str) -> Dict[str, Any]:
        """Extract summary and key data from document."""
        prompt = SUMMARY_PROMPT_TEMPLATE.format(document_text=full_document_text)
        
        try:
            logger.info("Generating document summary and key data")
            response_content = analysis_llm.invoke(prompt).content
            result = parse_json_response(response_content)
            logger.info("Successfully generated summary and key data")
            return result
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return create_error_response(str(e))
    
    @staticmethod
    def analyze_single_clause(retriever, topic_query: str) -> Dict[str, Any]:
        """Analyze a single clause topic."""
        try:
            json_llm = ChatOpenAI(
                openai_api_key=OPENAI_API_KEY, 
                model=ANALYSIS_MODEL, 
                temperature=ANALYSIS_TEMPERATURE,
                model_kwargs={"response_format": {"type": "json_object"}}
            )
            
            parser = PydanticOutputParser(pydantic_object=ClauseAnalysis)
            prompt_template = PromptTemplate(
                template=CLAUSE_ANALYSIS_TEMPLATE,
                input_variables=["topic_query", "context_text"],
                partial_variables={"format_instructions": parser.get_format_instructions()},
            )
            
            chain = prompt_template | json_llm | parser
            
            retrieved_docs = retriever.invoke(topic_query)
            context_text = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
            
            response_object = chain.invoke({
                "topic_query": topic_query,
                "context_text": context_text
            })
            
            return response_object.dict()
        except Exception as e:
            logger.error(f"Failed to analyze topic '{topic_query}': {str(e)}")
            return {"error": f"Failed to generate AI analysis for topic: {topic_query}"}

class QuestionAnswering:
    """Handles Q&A functionality."""
    
    @staticmethod
    def ask_question(query: str, namespace: str) -> str:
        """Answer question about processed document."""
        if not namespace:
            return "Error: No document namespace provided."

        vector_store = vector_stores.get(namespace)
        if not vector_store:
            return "Error: Document has not been processed or session has expired."

        try:
            retriever = vector_store.as_retriever()
            
            QA_PROMPT = PromptTemplate(
                template=QA_PROMPT_TEMPLATE, 
                input_variables=["context", "question"]
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

# Main public functions (keeping original API)
def setup_and_process_document(filepath: str) -> Dict[str, Any]:
    """Orchestrates the entire document processing and analysis workflow."""
    try:
        logger.info(f"Starting document processing for: {filepath}")
        
        # Process document
        full_text, chunks = DocumentProcessor.load_and_split_document(filepath)
        doc_namespace = os.path.basename(filepath)
        
        # Create vector store
        DocumentProcessor.create_vector_store(chunks, doc_namespace)
        
        # Generate summary
        summary_data = AIAnalyzer.get_summary_and_key_data(full_text)
        
        result = {
            "summary": summary_data,
            "namespace": doc_namespace
        }
        
        logger.info(f"Document processing completed successfully for: {doc_namespace}")
        return result
        
    except Exception as e:
        logger.error(f"Document processing failed: {str(e)}")
        return {"error": f"Document processing failed: {str(e)}"}

def analyze_clauses_and_risks(namespace: str) -> Dict[str, Any]:
    """Analyze key clauses and risks for processed document."""
    if namespace not in vector_stores:
        logger.error(f"Namespace '{namespace}' not found")
        return {"error": f"Document namespace '{namespace}' has not been processed."}

    try:
        vector_store = vector_stores[namespace]
        retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVAL_K})

        analysis_results = {}
        for json_key, topic_query in CLAUSE_TOPICS.items():
            logger.info(f"Analyzing topic: {topic_query}")
            analysis_results[json_key] = AIAnalyzer.analyze_single_clause(retriever, topic_query)
                
        return analysis_results
        
    except Exception as e:
        logger.error(f"Clause analysis failed for namespace '{namespace}': {str(e)}")
        return {"error": f"Clause analysis failed: {str(e)}"}

def ask_question(query: str, namespace: str) -> str:
    """Handle follow-up Q&A for processed document."""
    return QuestionAnswering.ask_question(query, namespace)
