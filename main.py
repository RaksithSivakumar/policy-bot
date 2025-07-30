import os
import requests
import tempfile
from typing import List
from dotenv import load_dotenv
import google.generativeai as genai
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain.schema import Document
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, validator
import uvicorn
import time
import logging
from email.parser import Parser

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

# CHANGE 1: Add root_path for URL prefix support
app = FastAPI(
    title="HackRX RAG API", 
    description="Single endpoint RAG system for PDF, DOCX, and Email documents", 
    version="1.0.0",
    root_path="/api/v1"  # This handles the /api/v1 prefix
)

# Security
security = HTTPBearer()

# Supported document types
SUPPORTED_EXTENSIONS = {'.pdf', '.docx', '.doc', '.eml', '.msg', '.txt'}

# Request/Response models
class QueryRequest(BaseModel):
    documents: str
    questions: List[str]
    
    @validator('documents')
    def validate_documents(cls, v):
        if not v.strip():
            raise ValueError('documents cannot be empty')
        return v
    
    @validator('questions')
    def validate_questions(cls, v):
        if not v:
            raise ValueError('questions list cannot be empty')
        if len(v) > 20:  # Limit questions per request
            raise ValueError('maximum 20 questions per request')
        return v

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
            logger.info(f"✅ Index {INDEX_NAME} already exists")
            
        return pc
        
    except Exception as e:
        logger.error(f"Error setting up Pinecone: {str(e)}")
        raise

def detect_document_type(file_path: str, content_type: str = None) -> str:
    """Detect document type from file extension and content type"""
    # Get file extension
    _, ext = os.path.splitext(file_path.lower())
    
    # Check by extension first
    if ext == '.pdf':
        return 'pdf'
    elif ext in ['.docx', '.doc']:
        return 'docx'
    elif ext in ['.eml', '.msg']:
        return 'email'
    elif ext == '.txt':
        # Check if it's an email in text format
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(1000)  # Read first 1000 chars
                if any(header in content.lower() for header in ['from:', 'to:', 'subject:', 'date:']):
                    return 'email'
        except:
            pass
        return 'txt'
    
    # Check by content type if extension is unclear
    if content_type:
        if 'pdf' in content_type.lower():
            return 'pdf'
        elif 'word' in content_type.lower() or 'document' in content_type.lower():
            return 'docx'
        elif 'email' in content_type.lower() or 'message' in content_type.lower():
            return 'email'
    
    # Default fallback
    logger.warning(f"Could not determine document type for {file_path}, defaulting to text")
    return 'txt'

def download_document(url: str) -> str:
    """Download document from URL and return temporary file path"""
    try:
        logger.info(f"Downloading document from: {url}")
        
        # Add headers to mimic a browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/msword,message/rfc822,*/*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        
        response = requests.get(url, headers=headers, timeout=30, allow_redirects=True)
        
        # Check if the response is successful
        if response.status_code == 403:
            logger.error("403 Forbidden - The document URL may have expired or requires different authentication")
            raise HTTPException(
                status_code=403, 
                detail="Document URL access denied. The URL may have expired or requires different authentication."
            )
        elif response.status_code == 404:
            logger.error("404 Not Found - The document URL does not exist")
            raise HTTPException(
                status_code=404, 
                detail="Document not found at the provided URL. Please check the URL and try again."
            )
        
        response.raise_for_status()
        
        # Get content type
        content_type = response.headers.get('content-type', '').lower()
        logger.info(f"Content type: {content_type}")
        
        # Check if we actually got content
        if len(response.content) == 0:
            raise HTTPException(status_code=400, detail="Downloaded file is empty")
        
        # Determine file extension based on content type
        file_ext = '.bin'  # default
        if 'pdf' in content_type:
            file_ext = '.pdf'
        elif 'word' in content_type or 'officedocument' in content_type:
            file_ext = '.docx'
        elif 'msword' in content_type:
            file_ext = '.doc'
        elif 'email' in content_type or 'message' in content_type:
            file_ext = '.eml'
        elif 'text' in content_type:
            file_ext = '.txt'
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            tmp_file.write(response.content)
            tmp_path = tmp_file.name
            
        logger.info(f"Document downloaded successfully to: {tmp_path} (Size: {len(response.content)} bytes)")
        return tmp_path
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error downloading document: {str(e)}")
        if "403" in str(e):
            raise HTTPException(
                status_code=403, 
                detail="Access denied to document URL. The URL may have expired or requires authentication."
            )
        elif "404" in str(e):
            raise HTTPException(
                status_code=404, 
                detail="Document not found at the provided URL."
            )
        else:
            raise HTTPException(status_code=400, detail=f"Failed to download document: {str(e)}")
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        logger.error(f"Unexpected error downloading document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error downloading document: {str(e)}")

def parse_email_content(file_path: str) -> List[Document]:
    """Parse email content and extract meaningful text"""
    try:
        documents = []
        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Try to parse as email
        if 'From:' in content or 'To:' in content or 'Subject:' in content:
            # Parse email headers and body
            parser = Parser()
            try:
                msg = parser.parsestr(content)
                
                # Extract email metadata
                subject = msg.get('Subject', 'No Subject')
                from_addr = msg.get('From', 'Unknown Sender')
                to_addr = msg.get('To', 'Unknown Recipient')
                date = msg.get('Date', 'Unknown Date')
                
                # Extract body
                body = ""
                if msg.is_multipart():
                    for part in msg.walk():
                        if part.get_content_type() == "text/plain":
                            try:
                                body += part.get_payload(decode=True).decode('utf-8', errors='ignore')
                            except:
                                body += str(part.get_payload())
                else:
                    try:
                        body = msg.get_payload(decode=True).decode('utf-8', errors='ignore')
                    except:
                        body = str(msg.get_payload())
                
                # Create structured document
                email_text = f"""
                Subject: {subject}
                From: {from_addr}
                To: {to_addr}
                Date: {date}
                
                Content:
                {body}
                """
                
                documents.append(Document(
                    page_content=email_text.strip(),
                    metadata={
                        "source": file_path,
                        "type": "email",
                        "subject": subject,
                        "from": from_addr,
                        "to": to_addr,
                        "date": date
                    }
                ))
                
            except Exception as e:
                logger.warning(f"Failed to parse as email, treating as plain text: {str(e)}")
                # Fallback to plain text
                documents.append(Document(
                    page_content=content,
                    metadata={"source": file_path, "type": "text"}
                ))
        else:
            # Treat as plain text
            documents.append(Document(
                page_content=content,
                metadata={"source": file_path, "type": "text"}
            ))
        
        return documents
        
    except Exception as e:
        logger.error(f"Error parsing email content: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error parsing email content: {str(e)}")

def process_document_source(source: str) -> str:
    """Process document source - either URL or local file path"""
    # Check if it's a local file path
    if os.path.exists(source):
        # Validate file extension
        _, ext = os.path.splitext(source.lower())
        if ext not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: {ext}. Supported types: {', '.join(SUPPORTED_EXTENSIONS)}"
            )
        logger.info(f"Using local file: {source}")
        return source
    
    # Check if it's a URL
    if source.startswith(('http://', 'https://')):
        logger.info(f"Downloading document from URL: {source}")
        return download_document(source)
    
    # Invalid source
    raise HTTPException(
        status_code=400, 
        detail=f"Invalid document source. Must be a valid URL or local file path to a supported document type: {', '.join(SUPPORTED_EXTENSIONS)}"
    )

def load_and_process_documents(file_path: str, is_temp_file: bool = True):
    """Load document based on type and split into chunks"""
    try:
        # Verify file exists
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"Document file not found: {file_path}")
        
        # Detect document type
        doc_type = detect_document_type(file_path)
        logger.info(f"Detected document type: {doc_type}")
        
        # Load document based on type
        documents = []
        
        if doc_type == 'pdf':
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            logger.info(f"Loaded {len(documents)} pages from PDF")
            
        elif doc_type == 'docx':
            try:
                loader = Docx2txtLoader(file_path)
                documents = loader.load()
                logger.info(f"Loaded DOCX document with {len(documents)} sections")
            except Exception as e:
                logger.error(f"Error loading DOCX: {str(e)}")
                # Fallback: try to read as text
                try:
                    with open(file_path, 'rb') as f:
                        content = f.read().decode('utf-8', errors='ignore')
                    documents = [Document(page_content=content, metadata={"source": file_path, "type": "docx"})]
                except Exception as fallback_e:
                    logger.error(f"Fallback also failed: {str(fallback_e)}")
                    raise HTTPException(status_code=500, detail=f"Could not process DOCX file: {str(e)}")
                
        elif doc_type == 'email':
            documents = parse_email_content(file_path)
            logger.info(f"Loaded email document with {len(documents)} sections")
            
        else:  # text or unknown
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            documents = [Document(page_content=content, metadata={"source": file_path, "type": "text"})]
            logger.info("Loaded as plain text document")
        
        if not documents:
            raise HTTPException(status_code=400, detail="No content could be extracted from the document")
        
        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=3500,
            chunk_overlap=300,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        docs = text_splitter.split_documents(documents)
        logger.info(f"Split into {len(docs)} chunks")
        
        return docs, is_temp_file
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
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
            "You are a highly accurate and professional assistant answering questions strictly using information from uploaded documents such as PDFs, Word files, or emails. "
            "Respond to each question with a clear, concise, factually correct, and well-structured full sentence. "
            "Avoid using bullet points, line breaks, or formatting unless explicitly present in the original document. "
            "Each answer should be informative and natural, suitable for direct use in policy explanations or customer communication. "
            "Use formal language with high confidence, and maintain consistency in terminology. "
            "When answering questions about insurance policies, always include exclusions, conditions, and specific limits when applicable. "
            "If a question cannot be answered based on the document, respond exactly with: 'The document does not contain this information.' "
            "Ensure your response style resembles professional policy documentation or FAQ tone with an accuracy range of 75–85%."
        )

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=0,
            system_message=system_prompt,
            convert_system_message_to_human=True
        )

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 5})  # Increased k for better context
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
    
    file_path = None
    is_temp_file = False
    
    try:
        # Setup Pinecone
        pc = setup_pinecone()
        
        # Process document source (URL or local file)
        file_path = process_document_source(doc_source)
        is_temp_file = not os.path.exists(doc_source)  # True if we downloaded it
        
        # Verify the document file
        if not os.path.exists(file_path):
            raise HTTPException(status_code=500, detail="Failed to access document file")
        
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            raise HTTPException(status_code=400, detail="Document file is empty")
        
        logger.info(f"Processing document file: {file_path} (Size: {file_size} bytes)")
        
        docs, is_temp = load_and_process_documents(file_path, is_temp_file)
        
        if not docs:
            raise HTTPException(status_code=400, detail="No content could be extracted from the document")
        
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
        if file_path and is_temp_file and os.path.exists(file_path):
            try:
                os.unlink(file_path)
                logger.info(f"Cleaned up temporary file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file {file_path}: {str(e)}")

@app.post("/hackrx/run", response_model=QueryResponse)
async def run_queries(request: QueryRequest, token: str = Depends(verify_token)):
    """
    Process multi-format documents (PDF, DOCX, Email) and answer questions using RAG
    
    Supported formats:
    - PDF documents (.pdf)
    - Word documents (.docx, .doc)
    - Email files (.eml, .msg, or text files with email headers)
    - Plain text files (.txt)
    
    The API automatically detects the document type and processes it accordingly.
    """
    try:
        logger.info(f"🚀 Processing request with {len(request.questions)} questions")
        logger.info(f"Document source: {request.documents[:100]}...")
        
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
                clean_answer = answer.replace('\n', ' ').replace('\r', ' ').strip()
                
                # Remove extra spaces
                clean_answer = ' '.join(clean_answer.split())

                answers.append(clean_answer)
                logger.info(f"Answer {i+1} generated successfully")
                
            except Exception as e:
                logger.error(f"Error processing question {i+1}: {str(e)}")
                answers.append("Error processing this question.")
        
        logger.info("✅ All questions processed successfully")
        return QueryResponse(answers=answers)
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        logger.error(f"❌ Error in run_queries: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# CHANGE 2: Update health endpoint to reflect new URL structure
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "message": "HackRX RAG API is running",
        "supported_formats": list(SUPPORTED_EXTENSIONS),
        "endpoint": "/api/v1/hackrx/run",  # Updated to show full path
        "public_url": "https://my-webhooks-endpoint.com/api/v1/hackrx/run"
    }

# CHANGE 3: Update root endpoint to reflect new URL structure
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "HackRX RAG API - Single Endpoint", 
        "version": "1.0.0",
        "main_endpoint": "/api/v1/hackrx/run",  # Updated to show full path
        "supported_formats": list(SUPPORTED_EXTENSIONS),
        "description": "Upload PDF, DOCX, or Email documents and ask questions about them",
        "public_url": "https://my-webhooks-endpoint.com/api/v1/hackrx/run"
    }

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
        
        logger.info(f"✅ Supported document formats: {', '.join(SUPPORTED_EXTENSIONS)}")
        logger.info("✅ Ready to process documents at /api/v1/hackrx/run")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {str(e)}")
        raise

# CHANGE 4: Update uvicorn configuration for deployment
if __name__ == "__main__":
    # Get port from environment variable (useful for deployment platforms)
    port = int(os.getenv("PORT", 8000))
    
    uvicorn.run(
        "main:app",  # Replace "main" with your filename if different
        host="0.0.0.0",
        port=port,
        reload=False  # Set to False for production
    )