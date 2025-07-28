from PyPDF2 import PdfReader

def chunk_pdf(file_path, chunk_size=300):
    reader = PdfReader(file_path)
    full_text = ""

    for page in reader.pages:
        full_text += page.extract_text() + "\n"

    chunks = []
    for i in range(0, len(full_text), chunk_size):
        chunk = full_text[i:i+chunk_size]
        chunks.append(chunk.strip())

    return chunks
