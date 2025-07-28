#!/usr/bin/env python3
"""
Local PDF testing script for RAG API
This script starts a simple HTTP server to serve local PDF files
"""

import http.server
import socketserver
import threading
import time
import requests
import json
import os
from pathlib import Path

class PDFHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        super().end_headers()

def start_pdf_server(pdf_directory=".", port=8080):
    """Start a simple HTTP server to serve PDF files"""
    os.chdir(pdf_directory)
    handler = PDFHandler
    
    with socketserver.TCPServer(("", port), handler) as httpd:
        print(f"📁 Serving PDF files from {os.getcwd()} at http://localhost:{port}")
        httpd.serve_forever()

def test_rag_api_with_local_pdf(pdf_filename, questions, api_port=8000, pdf_port=8080):
    """Test the RAG API with a local PDF file"""
    
    # Construct the PDF URL
    pdf_url = f"http://localhost:{pdf_port}/{pdf_filename}"
    
    # RAG API endpoint
    api_url = f"http://localhost:{api_port}/hackrx/run"
    
    # Request payload
    payload = {
        "documents": pdf_url,
        "questions": questions
    }
    
    # Headers
    headers = {
        "Authorization": "Bearer d9e1bb21c5cff7a1f6ca363f518247edc095b46b77b0f67ccf4787591bfaabb7",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    print(f"🚀 Testing RAG API with PDF: {pdf_url}")
    print(f"📝 Questions: {len(questions)}")
    
    try:
        # Test PDF accessibility first
        pdf_response = requests.head(pdf_url, timeout=5)
        if pdf_response.status_code != 200:
            print(f"❌ PDF not accessible at {pdf_url} (Status: {pdf_response.status_code})")
            return
        
        print(f"✅ PDF accessible at {pdf_url}")
        
        # Send request to RAG API
        print("🔄 Sending request to RAG API...")
        response = requests.post(api_url, json=payload, headers=headers, timeout=300)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Success! Here are the answers:")
            print("-" * 50)
            
            for i, (question, answer) in enumerate(zip(questions, result.get("answers", [])), 1):
                print(f"Q{i}: {question}")
                print(f"A{i}: {answer}")
                print("-" * 50)
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {str(e)}")
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")

def main():
    """Main function to run the test"""
    
    # Configuration
    PDF_FILENAME = "policy.pdf"  # Change this to your PDF filename
    PDF_DIRECTORY = r"E:\Projects\Third Year\policy-bot\documents"  # Directory containing your PDF file
    PDF_PORT = 8080
    API_PORT = 8000
    
    # Sample questions
    QUESTIONS = [
        "What is this document about?",
        "Summarize the main points",
        "What are the key topics covered?"
    ]
    
    # Check if PDF file exists
    pdf_path = Path(PDF_DIRECTORY) / PDF_FILENAME
    if not pdf_path.exists():
        print(f"❌ PDF file not found: {pdf_path}")
        print(f"Please place your PDF file in {PDF_DIRECTORY} and update PDF_FILENAME in this script")
        return
    
    print(f"📄 Found PDF: {pdf_path}")
    
    # Start PDF server in a separate thread
    server_thread = threading.Thread(
        target=start_pdf_server, 
        args=(PDF_DIRECTORY, PDF_PORT),
        daemon=True
    )
    server_thread.start()
    
    # Wait a moment for server to start
    time.sleep(2)
    
    # Test the RAG API
    test_rag_api_with_local_pdf(PDF_FILENAME, QUESTIONS, API_PORT, PDF_PORT)

if __name__ == "__main__":
    print("🧪 Local PDF Testing Script for RAG API")
    print("=" * 50)
    main()