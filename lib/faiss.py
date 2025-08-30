import os
import faiss
from lib.utils import FAISS_INDEX_PATH

# -------------------- Check if database files exist --------------------
if not os.path.exists(FAISS_INDEX_PATH):
    print("FATAL: Database files not found.")
    exit()

# Load the FAISS index
print("Loading FAISS index...")
index = faiss.read_index(FAISS_INDEX_PATH)