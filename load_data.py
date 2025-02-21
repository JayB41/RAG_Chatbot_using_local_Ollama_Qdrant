import os
import shutil
import fitz  
import json
import uuid
import subprocess
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

# === Directory Paths ===
# Path to the folder containing input PDFs
INPUT_FOLDER = ".../data/input/"  

# Path to the folder where processed data will be stored
PROCESSED_FOLDER = ".../data/processed/"  

# Path to the local Qdrant vector database
QDRANT_DB_PATH = ".../data/qdrant_db/qdrant_local.db"  

# === Default Embedding Model ===
# Using SentenceTransformer for generating text embeddings
STANDARD_MODEL = "all-MiniLM-L6-v2"  
embedding_model = SentenceTransformer(STANDARD_MODEL)

# === Supported Ollama Models ===
# List of supported models for text generation
MODEL_NAMES = {
    "llama3.2": "llama3.2",  
    "deepseek": "deepseek"    
}

# === Initialize Qdrant Client ===
# Establish connection to the local Qdrant vector database
client = QdrantClient(path=QDRANT_DB_PATH)
COLLECTION_NAME = "document_embeddings"  

# Create the collection if it does not exist
if not client.collection_exists(COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE)  
    )

def extract_text_from_pdf(pdf_path):
    '''
    Extracts raw text from a PDF file using PyMuPDF.
    
    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        str: Extracted text content from the PDF.
    '''
    # Open the PDF document
    doc = fitz.open(pdf_path)  
    
    # Extract text from each page and concatenate with line breaks
    extracted_text = "
".join(page.get_text("text") for page in doc)  
    return extracted_text
