import os
import faiss
from lib.utils import FAISS_INDEX_PATH, METADATA_PATH

# -------------------- Check if database files exist --------------------
if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(METADATA_PATH):
    print("FATAL: Database files not found.")
    print("Please run the `process_emails.py` script first to generate the database.")
    exit()

# Load the FAISS index
print("Loading FAISS index...")
index = faiss.read_index(FAISS_INDEX_PATH)