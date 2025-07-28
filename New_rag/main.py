import os
import shutil
import pickle
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from utils.chunking import chunk_pdf
from utils.gemini_embed import embed_texts, ask_gemini
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "doc-index")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

TEMP_DIR = "temp"
DOC_STORE = "vector_store/docs.pkl"
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs("vector_store", exist_ok=True)

pc = Pinecone(api_key=PINECONE_API_KEY)

# Delete old index if exists (run once or when dimension changes)
if PINECONE_INDEX_NAME in pc.list_indexes().names():
    pc.delete_index(PINECONE_INDEX_NAME)
    print(f"Deleted old index: {PINECONE_INDEX_NAME}")

# Create new index with embedding dimension 768
print(f"Creating Pinecone index: {PINECONE_INDEX_NAME}")
pc.create_index(
    name=PINECONE_INDEX_NAME,
    dimension=768,
    metric="cosine",
    spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION)
)

index = pc.Index(PINECONE_INDEX_NAME)

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🔁 Starting up app...")
    yield
    print("🔁 Shutting down app...")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_path = os.path.join(TEMP_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    chunks = chunk_pdf(file_path)
    embeddings = embed_texts(chunks)

    print(f"Chunks count: {len(chunks)}")
    print(f"Embeddings count: {len(embeddings)}")
    print(f"Embedding dimensions sample: {[len(e) for e in embeddings[:5]]}")

    filtered_vectors = []
    filtered_chunks = {}

    expected_dim = 768

    for i, (emb, chunk) in enumerate(zip(embeddings, chunks)):
        if emb and len(emb) == expected_dim:
            vec_id = f"{file.filename}_{i}"
            filtered_vectors.append({
                "id": vec_id,
                "values": emb,
                "metadata": {"text": chunk}
            })
            filtered_chunks[vec_id] = chunk
        else:
            print(f"Skipping chunk {i} due to invalid embedding dimension: {len(emb) if emb else 'None'}")

    if not filtered_vectors:
        return {"error": "No valid embeddings generated from the uploaded file."}

    index.upsert(vectors=filtered_vectors)

    with open(DOC_STORE, "wb") as f:
        pickle.dump(filtered_chunks, f)

    return {"message": "PDF uploaded and indexed successfully."}

from pydantic import BaseModel

class QueryRequest(BaseModel):
    question: str

@app.post("/query")
async def query_knowledgebase(request: QueryRequest):
    question = request.question

    if not os.path.exists(DOC_STORE):
        return {"error": "No documents indexed yet."}

    # Get embedding from Gemini
    query_embedding = embed_texts([question])[0]

    if not query_embedding or not isinstance(query_embedding, list):
        return {"error": "Failed to generate valid embedding for query."}

    # Query Pinecone with Gemini-generated embedding
    result = index.query(
        vector=query_embedding,
        top_k=3,
        include_metadata=True
    )

    # Load your local document store
    with open(DOC_STORE, "rb") as f:
        chunk_dict = pickle.load(f)

    matched_chunks = []
    for match in result.matches:
        text = match.metadata.get("text") if match.metadata else chunk_dict.get(match.id, "")
        matched_chunks.append(text)

    # Combine matched chunks into context
    context = "\n".join(matched_chunks)

    # Ask Gemini to answer based on context
    answer = ask_gemini(context, question)

    return {"answer": answer}
