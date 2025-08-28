import os

BASE_DIR = os.path.dirname(__file__)  # current file directory
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "data", "emails_faiss_oaite.bin")
METADATA_PATH = os.path.join(BASE_DIR, "data", "emails_metadata_oaite.pkl")
EMBEDDING_MODEL_NAME = "text-embedding-3-large"
AGENT_MODEL = "gpt-4o-mini" # Or another powerful model like "gpt-4-turbo"