import streamlit as st
import subprocess
from qdrant_client import QdrantClient
import os
import json

# === Database Configuration ===
# Path to the local Qdrant vector database
DB_PATH = ".../data/qdrant_db/qdrant_local.db"

# Name of the collection used for document embeddings
COLLECTION_NAME = "document_embeddings"  

# File path to store chat history
CHAT_HISTORY_FILE = ".../data/chat_history.json"  

def get_available_models():
    '''
    Retrieves the list of available Ollama models by executing a subprocess command.

    Returns:
        list: List of available model names as strings.
    '''
    try:
        # Run the subprocess to list available models
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        
        # If the command was successful, parse the model names
        if result.returncode == 0:
            models = [line.split()[0] for line in result.stdout.strip().split("\n")[1:] if line]
            return models
        else:
            return []
    except Exception as e:
        return []

def connect_qdrant():
    '''
    Establishes a connection to the local Qdrant vector database.

    Returns:
        QdrantClient: An instance of the Qdrant client if the connection is successful.
    '''
    try:
        client = QdrantClient(path=DB_PATH)
        return client
    except RuntimeError:
        return None

def chat_with_model(model_name, prompt):
    '''
    Communicates with the specified Ollama model using a subprocess command.

    Args:
        model_name (str): The name of the model to use for generating a response.
        prompt (str): The input text prompt for the model.

    Returns:
        str: The generated response from the model or an error message.
    '''
    try:
        # Run the subprocess to communicate with the model
        result = subprocess.run(["ollama", "run", model_name, prompt], capture_output=True, text=True)
        
        # Return the output if the command was successful, otherwise return an error message
        return result.stdout.strip() if result.returncode == 0 else "Error: Model request failed."
    except Exception:
        return "Error: Communication with Ollama failed."

def load_chat_history():
    '''
    Loads the chat history from a JSON file.

    Returns:
        dict: Dictionary containing the saved chat history.
    '''
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as file:
            return json.load(file)
    return {}

def save_chat_history(history):
    '''
    Saves the chat history to a JSON file.

    Args:
        history (dict): The chat history to be saved.
    '''
    with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=4, ensure_ascii=False)
