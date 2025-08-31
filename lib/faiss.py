import os
import faiss
from lib.utils import VECTOR_DATA_PATH

# -------------------- Check if database files exist --------------------
if not os.path.exists(VECTOR_DATA_PATH):
    print("FATAL: Database files not found.")
    exit()

# Load the FAISS index
print("Loading FAISS index...")
index = faiss.read_index(VECTOR_DATA_PATH)