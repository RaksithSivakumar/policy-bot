import os
import openai
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"  # 1536 dims
CHAT_MODEL = "gpt-4o"

def embed_texts(texts: list[str]) -> list[list[float]]:
    texts = [t if t.strip() else " " for t in texts]  # avoid empty strings
    try:
        print(f"Embedding {len(texts)} texts")
        response = openai.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts
        )
        print("OpenAI embedding response:", response)
        return [d.embedding for d in response.data]
    except Exception as e:
        print(f"Embedding error: {e}")
        return [[] for _ in texts]


def ask_openai(context: str, question: str) -> str:
    prompt = f"""Answer the question based only on the following context:

Context:
{context}

Question: {question}
"""
    try:
        response = openai.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating answer: {e}"
