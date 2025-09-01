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
        output_path = "full_mails.jsonl"
        if not os.path.exists(output_path):
            with st.spinner("Downloading metadata from Google Drive (first-time setup)..."):
                gdown.download(id=st.secrets["EMAIL_JSONL_GDRIVE_ID"], output=output_path, quiet=False)
        data_path = output_path
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
    load_resources = st.cache_resource(_load_resources_base)
else:
    load_dotenv()
    load_resources = lru_cache(maxsize=None)(_load_resources_base)

# --- Global variables that your tools will import ---
df, chroma_collection = load_resources()



# import os
# from lib.utils import CHROMA_COLLECTION_NAME, EMAIL_JSON_PATH
# import polars as pl
# import chromadb
# from dotenv import load_dotenv
# import faiss

# load_dotenv()

# EMAIL_JSONL_GDRIVE_ID = "YOUR_GOOGLE_DRIVE_FILE_ID_HERE"

# # --- NEW: Connect to ChromaDB Cloud ---
# # This replaces the old FAISS index loading.
# print("Connecting to ChromaDB Cloud...")
# try:
#     # Load credentials from environment
#     api_key = os.getenv("CHROMA_API_KEY")
#     tenant = os.getenv("CHROMA_TENANT")
#     database = os.getenv("CHROMA_DATABASE")

#     # Create the client and get the collection object
#     client = chromadb.CloudClient(api_key=api_key, tenant=tenant, database=database)
#     chroma_collection = client.get_collection(name=CHROMA_COLLECTION_NAME)
    
#     print("Successfully connected to ChromaDB collection.")
# except Exception as e:
#     print(f"FATAL ERROR: Could not connect to ChromaDB. {e}")
#     # Create a placeholder so the app doesn't crash on import, but tools will fail.
#     chroma_collection = None 

# # --- UNCHANGED: Load the email jsonl data with Polars ---
# # This logic is kept exactly as it was.
# print("Loading emails data-set for metadata...")
# df = pl.read_ndjson(EMAIL_JSON_PATH)
# df = df.with_columns(
#     pl.col("date").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False).dt.replace_time_zone("UTC").alias("date")
# )
# print(f"Successfully loaded {df.height} records for metadata.")