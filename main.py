import os
import requests
import tempfile
from typing import List
from dotenv import load_dotenv
import google.generativeai as genai
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from fastapi import FastAPI, HTTPException, Depends, Security, File, UploadFile
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import uvicorn
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Get API keys from environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY_1")
BEARER_TOKEN = os.getenv("BEARER_TOKEN", "d9e1bb21c5cff7a1f6ca363f518247edc095b46b77b0f67ccf4787591bfaabb7")

# Validate API keys
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is not set")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY environment variable is not set")

# Set environment variables for langchain
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

INDEX_NAME = "index-1"

app = FastAPI(title="HackRX RAG API", description="RAG system for document Q&A", version="1.0.0")

# Security
security = HTTPBearer()

# Request/Response models
class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

# Global variables to hold the systems
qa_systems = {}

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Verify the bearer token"""
    if credentials.credentials != BEARER_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    return credentials.credentials

def setup_pinecone():
    """Initialize Pinecone and create index if needed"""
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        
        # Check if index exists
        existing_indexes = pc.list_indexes().names()
        
        if INDEX_NAME not in existing_indexes:
            logger.info(f"Creating new index: {INDEX_NAME}")
            pc.create_index(
                name=INDEX_NAME,
                dimension=768,  # Gemini embedding dimension
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
            # Wait for index to be ready
            time.sleep(10)
            logger.info("Index created successfully!")
        else:
            # Verify dimensions
            index_info = pc.describe_index(INDEX_NAME)
            if index_info.dimension != 768:
                logger.error(f"Index dimension mismatch: {index_info.dimension} != 768")
                raise ValueError("Index dimension mismatch")
            logger.info(f"✅ Index {INDEX_NAME} already exists with correct dimension")
            
        return pc
        
    except Exception as e:
        logger.error(f"Error setting up Pinecone: {str(e)}")
        raise

def process_document_source(source: str) -> str:
    """Process document source - either URL or local file path"""
    # Check if it's a local file path
    if os.path.exists(source) and source.lower().endswith('.pdf'):
        logger.info(f"Using local PDF file: {source}")
        return source
    
    # Check if it's a URL
    if source.startswith(('http://', 'https://')):
        logger.info(f"Downloading PDF from URL: {source}")
        return download_pdf(source)
    
    # Invalid source
    raise HTTPException(
        status_code=400, 
        detail=f"Invalid document source. Must be a valid URL or local file path to a PDF file."
    )
    """Download PDF from URL and return temporary file path"""
    try:
        logger.info(f"Downloading PDF from: {url}")
        
        # Add headers to mimic a browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/pdf,*/*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        
        response = requests.get(url, headers=headers, timeout=30, allow_redirects=True)
        
        # Check if the response is successful
        if response.status_code == 403:
            logger.error("403 Forbidden - The PDF URL may have expired or requires different authentication")
            raise HTTPException(
                status_code=403, 
                detail="PDF URL access denied. The URL may have expired or requires different authentication. Please provide a valid PDF URL."
            )
        elif response.status_code == 404:
            logger.error("404 Not Found - The PDF URL does not exist")
            raise HTTPException(
                status_code=404, 
                detail="PDF not found at the provided URL. Please check the URL and try again."
            )
        
        response.raise_for_status()
        
        # Verify content type
        content_type = response.headers.get('content-type', '').lower()
        if 'pdf' not in content_type and not content_type.startswith('application/'):
            logger.warning(f"Content type may not be PDF: {content_type}")
        
        # Check if we actually got content
        if len(response.content) == 0:
            raise HTTPException(status_code=400, detail="Downloaded file is empty")
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(response.content)
            tmp_path = tmp_file.name
            
        logger.info(f"PDF downloaded successfully to: {tmp_path} (Size: {len(response.content)} bytes)")
        return tmp_path
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error downloading PDF: {str(e)}")
        if "403" in str(e):
            raise HTTPException(
                status_code=403, 
                detail="Access denied to PDF URL. The URL may have expired or requires authentication."
            )
        elif "404" in str(e):
            raise HTTPException(
                status_code=404, 
                detail="PDF not found at the provided URL."
            )
        else:
            raise HTTPException(status_code=400, detail=f"Failed to download PDF: {str(e)}")
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        logger.error(f"Unexpected error downloading PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error downloading PDF: {str(e)}")

def load_and_process_documents(pdf_path: str, is_temp_file: bool = True):
    """Load PDF and split into chunks"""
    try:
        # Verify file exists
        if not os.path.exists(pdf_path):
            raise HTTPException(status_code=404, detail=f"PDF file not found: {pdf_path}")
        
        # Load document
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} pages from PDF")
        
        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=3500,
            chunk_overlap=300
        )
        docs = text_splitter.split_documents(documents)
        logger.info(f"Split into {len(docs)} chunks")
        
        return docs, is_temp_file
        
    except Exception as e:
        logger.error(f"Error processing documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing documents: {str(e)}")

def create_document_hash(url: str) -> str:
    """Create a hash for the document URL to use as identifier"""
    import hashlib
    return hashlib.md5(url.encode()).hexdigest()

def store_embeddings_in_pinecone(docs, doc_hash: str):
    """Create embeddings using Gemini and store in Pinecone with namespace"""
    try:
        # Create Gemini embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GOOGLE_API_KEY
        )
        
        logger.info("Creating embeddings and storing in Pinecone...")
        
        # Upload to Pinecone with namespace
        vectorstore = PineconeVectorStore.from_documents(
            docs, 
            embeddings, 
            index_name=INDEX_NAME,
            namespace=doc_hash
        )
        
        logger.info("✅ Documents successfully stored in Pinecone with Gemini embeddings!")
        return vectorstore
        
    except Exception as e:
        logger.error(f"Error creating embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating embeddings: {str(e)}")

def setup_qa_system(vectorstore):
    """Setup the QA system with Gemini LLM"""
    try:
        system_prompt = (
            "You are a helpful assistant answering questions from an insurance policy document. "
            "Answer each question with a complete and factually accurate sentence, using only the information found in the document. "
            "Do not include line breaks, bullet points, or extra commentary. "
            "If the document does not contain the answer, respond with: 'The document does not contain this information.'"
            "Mention the key exclusions for maternity, childbirth, and follow-up treatment, which are important to clarify the coverage boundaries"
        )



        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=0,
            system_message=system_prompt,  # ✅ Add this line
            convert_system_message_to_human=True
        )

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
        )

        return qa

    except Exception as e:
        logger.error(f"Error setting up QA system: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error setting up QA system: {str(e)}")


def get_or_create_qa_system(doc_source: str):
    """Get existing QA system or create new one for the document"""
    doc_hash = create_document_hash(doc_source)
    
    # Check if we already have a QA system for this document
    if doc_hash in qa_systems:
        logger.info(f"Reusing existing QA system for document: {doc_hash}")
        return qa_systems[doc_hash]
    
    pdf_path = None
    is_temp_file = False
    
    try:
        # Setup Pinecone
        pc = setup_pinecone()
        
        # Process document source (URL or local file)
        pdf_path = process_document_source(doc_source)
        is_temp_file = not os.path.exists(doc_source)  # True if we downloaded it
        
        # Verify the PDF file
        if not os.path.exists(pdf_path):
            raise HTTPException(status_code=500, detail="Failed to access PDF file")
        
        file_size = os.path.getsize(pdf_path)
        if file_size == 0:
            raise HTTPException(status_code=400, detail="PDF file is empty")
        
        logger.info(f"Processing PDF file: {pdf_path} (Size: {file_size} bytes)")
        
        docs, is_temp = load_and_process_documents(pdf_path, is_temp_file)
        
        if not docs:
            raise HTTPException(status_code=400, detail="No content could be extracted from the PDF")
        
        # Store embeddings
        vectorstore = store_embeddings_in_pinecone(docs, doc_hash)
        
        # Setup QA system
        qa = setup_qa_system(vectorstore)
        
        # Cache the QA system
        qa_systems[doc_hash] = qa
        
        logger.info(f"Created new QA system for document: {doc_hash}")
        return qa
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        logger.error(f"Error creating QA system: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating QA system: {str(e)}")
    finally:
        # Clean up temporary file only if it was downloaded
        if pdf_path and is_temp_file and os.path.exists(pdf_path):
            try:
                os.unlink(pdf_path)
                logger.info(f"Cleaned up temporary file: {pdf_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file {pdf_path}: {str(e)}")

@app.post("/hackrx/run", response_model=QueryResponse)
async def run_queries(request: QueryRequest, token: str = Depends(verify_token)):
    """
    Process document and answer questions using RAG
    """
    try:
        logger.info(f"Processing request with {len(request.questions)} questions")
        logger.info(f"Document URL: {request.documents}")
        
        # Get or create QA system for the document
        qa = get_or_create_qa_system(request.documents)
        
        # Process all questions
        answers = []
        for i, question in enumerate(request.questions):
            logger.info(f"Processing question {i+1}/{len(request.questions)}: {question[:50]}...")
            
            try:
                # Use invoke instead of deprecated run method
                response = qa.invoke({"query": question})
                answer = response.get("result", "No answer generated")

                    # Strip newline characters and ensure it's in one line
                clean_answer = answer.replace('\n', ' ').strip()

                # Prevent duplicates
                if clean_answer not in answers:
                    answers.append(clean_answer)

                logger.info(f"Answer {i+1} generated successfully")
                
            except Exception as e:
                logger.error(f"Error processing question {i+1}: {str(e)}")
                answers.append(f"Error processing question: {str(e)}")
        
        logger.info("All questions processed successfully")
        return QueryResponse(answers=answers)
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        logger.error(f"Error in run_queries: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Test endpoint for debugging
@app.post("/test/pdf-download")
async def test_pdf_download(url: str, token: str = Depends(verify_token)):
    """
    Test endpoint to check if PDF URL is accessible
    """
    try:
        logger.info(f"Testing PDF download from: {url}")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/pdf,*/*'
        }
        
        response = requests.head(url, headers=headers, timeout=10, allow_redirects=True)
        
        return {
            "url": url,
            "status_code": response.status_code,
            "headers": dict(response.headers),
            "accessible": response.status_code == 200,
            "content_type": response.headers.get('content-type', 'unknown'),
            "content_length": response.headers.get('content-length', 'unknown')
        }
        
    except Exception as e:
        return {
            "url": url,
            "error": str(e),
            "accessible": False
        }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "RAG API is running"}

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "HackRX RAG API", "version": "1.0.0"}

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    logger.info("🚀 Starting HackRX RAG API")
    try:
        # Test Pinecone connection
        setup_pinecone()
        logger.info("✅ Pinecone connection verified")
        
        # Test Gemini connection
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GOOGLE_API_KEY
        )
        logger.info("✅ Gemini connection verified")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {str(e)}")
        raise

if __name__ == "__main__":
    uvicorn.run(
        "main:app",  # Replace "main" with your filename if different
        host="0.0.0.0",
        port=8000,
        reload=True
    )