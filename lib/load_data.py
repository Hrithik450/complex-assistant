import sys
# import pysqlite3
# sys.modules["sqlite3"] = pysqlite3
import os
import sys
import polars as pl
import chromadb
import gdown
from functools import lru_cache
from dotenv import load_dotenv
from lib.utils import CHROMA_COLLECTION_NAME, EMAIL_JSON_PATH

# --- Environment Check ---
# This check remains the same and is the key to the solution.
IS_STREAMLIT_ENVIRONMENT = "streamlit" in sys.modules

# If in Streamlit, we'll need its special functions.
if IS_STREAMLIT_ENVIRONMENT:
    import streamlit as st

# --- Universal load_resources function ---
def _load_resources_base():
    """
    Base function that loads data and connects to ChromaDB, adapting its behavior
    based on whether it's running in Streamlit or a command-line environment.
    """
    # --- 1. Conditional Data Source Logic ---
    data_path = ""
    if IS_STREAMLIT_ENVIRONMENT:
        # --- STREAMLIT PATH: Download from Google Drive ---
        print("Streamlit environment detected. Will download data from Google Drive.")
        output_path_mails = "clean_mails.jsonl"
        if not os.path.exists(output_path_mails):
            with st.spinner("Downloading metadata from Google Drive (first-time setup)..."):
                gdown.download(id=st.secrets["EMAIL_JSONL_GDRIVE_ID"], output=output_path_mails, quiet=False)
        
        data_path = output_path_mails
    else:
        # --- COMMAND-LINE PATH: Use local file ---
        print("Command-line environment detected. Using local data file.")
        data_path = EMAIL_JSON_PATH
        if not os.path.exists(data_path):
            # Provide a clear error if the local file is missing.
            raise FileNotFoundError(f"Local data file not found at '{data_path}'. Please ensure it exists before running chatbot.py.")
        
    # --- 2. Shared Polars Loading Logic ---
    # This part is now the same for both environments, it just uses the determined data_path.
    print(f"Loading email metadata from: {data_path}")
    df = pl.read_ndjson(data_path)
    print(f"Successfully loaded {df.height} records for metadata.")

    # --- 3. Shared ChromaDB Connection Logic ---
    # This logic correctly handles secrets for both environments.
    print("Connecting to ChromaDB Vector Store...")
    try:
        if IS_STREAMLIT_ENVIRONMENT:
            client = chromadb.CloudClient(
                api_key=st.secrets["CHROMA_API_KEY"],
                tenant=st.secrets["CHROMA_TENANT"],
                database=st.secrets["CHROMA_DATABASE"]
            )
        else:
            client = chromadb.CloudClient(
                api_key=os.getenv("CHROMA_API_KEY"),
                tenant=os.getenv("CHROMA_TENANT"),
                database=os.getenv("CHROMA_DATABASE")
            )
        collection = client.get_collection(name=CHROMA_COLLECTION_NAME)
        print("Successfully connected to ChromaDB collection.")
    except Exception as e:
        print(f"FATAL ERROR: Could not connect to ChromaDB. {e}")
        collection = None
    
    return df, collection

# --- Environment-Specific Function Wrapper ---
if IS_STREAMLIT_ENVIRONMENT:
    @st.cache_resource
    def load_resources():
        return _load_resources_base()
else:
    load_dotenv()
    @lru_cache(maxsize=None)
    def load_resources():
        return _load_resources_base()

# --- Global variables that your tools will import ---
df, chroma_collection = load_resources()