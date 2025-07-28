# ingest.py
import os, fitz, json, faiss, google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import numpy as np

load_dotenv(); genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

EMBED = genai.embed_content
INDEX_FILE = "faiss.index"
META_FILE  = "meta.json"

def pdf_to_chunks(path: str, chunk=800, overlap=100):
    doc = fitz.open(path)
    full = "\n".join(page.get_text() for page in doc)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk, chunk_overlap=overlap)
    return splitter.split_text(full)

def build_index(chunks):
    embs = []
    for c in chunks:
        emb = EMBED(model="models/text-embedding-004", content=c)["embedding"]
        embs.append(emb)
    embs = np.array(embs).astype("float32")
    index = faiss.IndexFlatIP(embs.shape[1])    # cosine via inner-product + normalisation
    faiss.normalize_L2(embs)
    index.add(embs)
    faiss.write_index(index, INDEX_FILE)
    json.dump(chunks, open(META_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"Indexed {len(chunks)} chunks")

if __name__ == "__main__":
    import pathlib
    ROOT = pathlib.Path(__file__).parent
    chunks = pdf_to_chunks(ROOT / "data" / "policy.pdf")
    build_index(chunks)