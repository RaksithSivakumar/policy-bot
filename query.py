import os
import json
import faiss
import numpy as np
import pathlib
from dotenv import load_dotenv
import google.generativeai as genai

# Set up root path to current script folder
ROOT = pathlib.Path(__file__).parent

# Load environment variables from .env (for GOOGLE_API_KEY)
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load FAISS index from disk (make sure faiss.index exists here)
INDEX = faiss.read_index(str(ROOT / "faiss.index"))

# Load policy chunks metadata (make sure meta.json exists here)
with open(ROOT / "meta.json", encoding="utf-8") as f:
    CHUNKS = json.load(f)

# Set embedding function and generative model from Google GenAI SDK
EMBED = genai.embed_content
MODEL = genai.GenerativeModel("gemini-2.0-flash-exp")

SYSTEM_PROMPT = """
You are an intelligent insurance claims assistant.

📥 USER QUERY:
${userQuery}

📜 FULL POLICY TEXT (All 100 Pages):
Below is the complete, verbatim text of the 100-page policy document (each page as one JSON object).

${fullPolicyPages}

⚙️ TASKS
1. Extract the following structured details from the user query:
   - age (int)
   - gender ("male", "female", "other")
   - procedure (string)
   - location (string)
   - policy_age_months (int)

2. Scan the entire policy text to find clauses relevant to the extracted details (coverage, exclusions, waiting periods, reimbursement).
    - New points should start from the next line not in the same line with format specifier.
    - If you read any table format data, generate a table in markdown format. 
    - That table should be in the output.
3. Return **only valid JSON** in this exact format:



{
  "decision": "approved" | "rejected" | "needs_review",
  "amount_inr": <number or null>,
  "justification": "<brief explanation>",
  "clauses": ["<exact clause text with page reference>", "..."]
}

✅ RULES
- Quote clauses verbatim.  
- Cite the page number in parentheses.  
- If any required detail is missing, set decision to "needs_review".
"""

def vector_search(question: str, k=6):
    try:
        emb = EMBED(model="models/text-embedding-004", content=question)["embedding"]
    except Exception as e:
        print(f"Error getting embedding: {e}")
        raise

    emb = np.array([emb]).astype("float32")
    faiss.normalize_L2(emb)
    distances, idxs = INDEX.search(emb, k)

    # Defensive check: Ensure indices are valid
    results = []
    for i in idxs[0]:
        if i < len(CHUNKS):
            results.append(CHUNKS[i])
    return results

def decide(query: str):
    context = "\n\n".join(vector_search(query))
    prompt = f"{SYSTEM_PROMPT}\n\nUser query: {query}\n\nPolicy excerpts:\n{context}"
    try:
        response = MODEL.generate_content(prompt).text
    except Exception as e:
        print(f"Error generating content: {e}")
        raise

    # Clean response from code fences if present
    response = response.strip()
    if response.startswith("```json"):
        response = response[len("```json"):].strip()
    if response.endswith("```"):
        response = response[:-3].strip()

    return json.loads(response)

# ----------------- MULTI-QUERY LOOP -----------------
if __name__ == "__main__":
    print("🟢 Enter as many queries as you like. Press <Enter> on a blank line to quit.\n")
    while True:
        q = input("Query > ").strip()
        if not q:          # blank line → exit
            print("👋 Goodbye!")
            break
        try:
            result = decide(q)
            print(json.dumps(result, indent=2, ensure_ascii=False))
            print()        # blank line for readability
        except Exception as e:
            print(f"❌ Failed: {e}\n")
