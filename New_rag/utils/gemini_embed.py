import google.generativeai as genai
import os

# Use environment variable without hardcoded fallback for security
API_KEY = "AIzaSyArcAQAxzgtN92KcjAed0sJ4PHZaGSWSSI"
if not API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is required")

genai.configure(api_key=API_KEY)

# Embedding model name - no need to instantiate a separate object
EMBEDDING_MODEL = "models/text-embedding-004"

# Load chat model - use newer model
chat_model = genai.GenerativeModel("gemini-1.5-flash")

def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of texts using Gemini's embedding model.
    
    Args:
        texts: List of strings to embed
        
    Returns:
        List of embedding vectors (each vector is a list of floats)
    """
    embeddings = []
    for text in texts:
        try:
            result = genai.embed_content(
                model=EMBEDDING_MODEL,
                content=text,
                task_type="retrieval_document"
            )
            embeddings.append(result['embedding'])
        except Exception as e:
            print(f"Error embedding text: {e}")
            # You might want to handle this differently based on your needs
            embeddings.append([])
    
    return embeddings

def ask_gemini(context: str, question: str) -> str:
    """
    Ask Gemini a question based on provided context.
    
    Args:
        context: Background information for the question
        question: The question to ask
        
    Returns:
        Gemini's response as a string
    """
    prompt = f"""
You are an intelligent assistant designed to help answer insurance-related queries by analyzing relevant segments from policy documents. Your task is to understand the user's question, use the retrieved context from indexed policy documents, and provide a JSON response with your analysis and answer.

### ❓ USER QUESTION:
{question}

### 📄 RELEVANT POLICY CONTEXT:
The following content consists of top-matched segments retrieved from the insurance policy knowledge base:

{context}

---

### 🎯 OBJECTIVE

Based on the user's query and the retrieved document chunks, perform the following:

1. **Identify important query elements**, such as:
   - Age
   - Gender
   - Medical Procedure
   - City or Region
   - Policy Duration (in months)

2. **Analyze the context to find relevant information**, especially regarding:
   - Coverage
   - Waiting Periods
   - Reimbursement conditions
   - Any exclusion clauses

3. **Return ONLY valid JSON** in this exact format. Do NOT include any explanation outside the JSON block:

```json
{{
  "decision": "approved" | "rejected" | "needs_review",
  "amount_inr": <number or null>,
  "justification": "<brief explanation>",
  "clauses": [
    "<verbatim clause text (mention page number if known)>",
    "..."
  ]
}}
"""
    try:
        chat = chat_model.start_chat()
        response = chat.send_message(prompt)
        return response.text
    except Exception as e:
        return f"Error generating response: {e}"



