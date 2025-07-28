import os
import fitz
import json
import faiss
import pathlib
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai

# Load environment and configure GenAI
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Constants
ROOT = pathlib.Path(__file__).parent
INDEX_FILE = ROOT / "faiss.index"
META_FILE = ROOT / "meta.json"
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

EMBED = genai.embed_content
MODEL = genai.GenerativeModel("gemini-2.0-flash-exp")

SYSTEM_PROMPT = """
You are a universal document analysis assistant capable of processing multiple document types including:
- Insurance policies and claims
- Medical reports and records
- Resumes and CVs
- Legal contracts
- Financial documents
- Technical specifications

📥 USER QUERY:
${userQuery}

📜 DOCUMENT CONTENT:
Below are excerpts from the uploaded document(s):

${documentExcerpts}

⚙️ TASKS
1. Identify the document type and its primary purpose
2. Extract relevant structured information based on document type:
   - For insurance: age, coverage details, claims info
   - For medical: patient info, diagnoses, treatments
   - For resumes: skills, experience, education
   - For legal: parties, terms, obligations
   - For financial: amounts, dates, transactions
3. Analyze the content to answer the user's query by:
   - Finding relevant sections or clauses
   - Cross-referencing multiple pieces of information
   - Identifying any contradictions or special conditions
4. Return **only valid JSON** in this exact format:

{
  "document_type": "<identified type>",
  "key_entities": {
    "people": [],
    "organizations": [],
    "dates": [],
    "locations": [],
    "amounts": []
  },
  "decision": "<context-appropriate decision>",
  "summary": "<brief document summary>",
  "relevant_sections": [
    {
      "text": "<exact text excerpt>",
      "page": <number>,
      "confidence": 0.0-1.0
    }
  ],
  "action_items": [],
  "potential_issues": []
}

✅ RULES
- Maintain strict document neutrality (don't favor any party)
- Highlight any incomplete or ambiguous information
- For tables, convert to markdown format
- Always cite source locations
- Flag any compliance risks or inconsistencies
"""

# ------------------- Functions -------------------
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
    faiss.normalize_L2(embs)
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    faiss.write_index(index, str(INDEX_FILE))
    json.dump(chunks, open(META_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return len(chunks)

def vector_search(question: str, k=6):
    index = faiss.read_index(str(INDEX_FILE))
    with open(META_FILE, encoding="utf-8") as f:
        chunks = json.load(f)

    emb = EMBED(model="models/text-embedding-004", content=question)["embedding"]
    emb = np.array([emb]).astype("float32")
    faiss.normalize_L2(emb)
    D, I = index.search(emb, k)

    results = []
    for i in I[0]:
        if i < len(chunks):
            results.append(chunks[i])
    return results

def decide(query: str):
    context = "\n\n".join(vector_search(query))
    prompt = f"{SYSTEM_PROMPT}\n\nUser query: {query}\n\nDocument excerpts:\n{context}"
    response = MODEL.generate_content(prompt).text.strip()
    
    # Clean response
    for prefix in ["```json", "```"]:
        if response.startswith(prefix):
            response = response[len(prefix):].strip()
        if response.endswith(prefix):
            response = response[:-len(prefix)].strip()
    
    return json.loads(response)

# ------------------- Streamlit App -------------------
st.set_page_config(page_title="Universal Document Analyzer", layout="centered")
st.title("📑 Multi-Document Analysis Assistant")

# Document Upload & Indexing
with st.expander("📤 Upload Documents & Build Index"):
    uploaded_file = st.file_uploader(
        "Upload your document (PDF, DOCX, TXT supported)", 
        type=["pdf", "docx", "txt"],
        accept_multiple_files=False
    )
    
    if uploaded_file:
        save_path = DATA_DIR / uploaded_file.name
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"✅ {uploaded_file.type} saved. Now building index...")
        
        if uploaded_file.type == "application/pdf":
            chunks = pdf_to_chunks(str(save_path))
        else:
            # Add handlers for other file types here
            with open(save_path, "r", encoding="utf-8") as f:
                text = f.read()
            splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
            chunks = splitter.split_text(text)
        
        count = build_index(chunks)
        st.success(f"✅ Indexed {count} text chunks from {uploaded_file.name}")

# User Query Input
if INDEX_FILE.exists() and META_FILE.exists():
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_area("🧠 Enter your document query:", height=150)
    with col2:
        doc_type = st.selectbox(
            "Document Type",
            ["Auto-detect", "Insurance", "Medical", "Resume", "Legal", "Financial", "Other"]
        )
    
    if st.button("🔍 Analyze Document"):
        with st.spinner("Analyzing document... 💡"):
            try:
                result = decide(query)
                
                st.subheader("📋 Analysis Results")
                st.json(result)
                
                # Enhanced visualization
                with st.expander("🔍 Detailed View"):
                    if result.get("relevant_sections"):
                        st.subheader("Key Sections")
                        for section in result["relevant_sections"]:
                            with st.container(border=True):
                                st.markdown(f"**Page {section.get('page', 'N/A')}** (Confidence: {section.get('confidence', 1.0):.2f})")
                                st.text(section["text"])
                    
                    if result.get("action_items"):
                        st.subheader("Recommended Actions")
                        for action in result["action_items"]:
                            st.markdown(f"- {action}")
                
                if result.get("potential_issues"):
                    st.warning("⚠️ Potential Issues Found")
                    for issue in result["potential_issues"]:
                        st.error(f"- {issue}")
                        
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
else:
    st.warning("⚠️ Please upload a document and build the index first.")