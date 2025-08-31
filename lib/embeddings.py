from langchain_openai import OpenAIEmbeddings
from lib.utils import EMBEDDING_MODEL_NAME
from dotenv import load_dotenv
import os

# -------------------- Load environment variables from .env file --------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
print(f"Using open ai key: {OPENAI_API_KEY}")
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME, api_key=OPENAI_API_KEY)