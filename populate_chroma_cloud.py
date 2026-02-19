# --- START OF FILE: populate_chroma_cloud.py ---

import os
import pickle
import logging
import chromadb
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- I/O AND COLLECTION CONFIGURATION ---
PKL_FILE_PATH = "optimized_chunks.pkl" 
CHROMA_COLLECTION_NAME = "my_document_collection" # Name for the collection within your database

# --- MODEL AND EMBEDDING CONFIGURATION ---
EMBEDDING_MODEL_NAME = "text-embedding-3-large"
EMBEDDING_BATCH_SIZE = 250 # A safe batch size for stability, 300 is the limit

def main():
    """
    Main function to load text chunks, generate embeddings, 
    and populate a Chroma Cloud collection using the CloudClient.
    """
    load_dotenv()
    
    # --- 1. LOAD CREDENTIALS FROM .env FILE ---
    logging.info("Loading credentials from .env file...")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    chroma_api_key = os.getenv("CHROMA_API_KEY")
    chroma_tenant = os.getenv("CHROMA_TENANT")
    chroma_database = os.getenv("CHROMA_DATABASE")

    # Validate that all required credentials are set
    if not all([openai_api_key, chroma_api_key, chroma_tenant, chroma_database]):
        logging.error("FATAL: Missing one or more required environment variables.")
        logging.error("Please ensure OPENAI_API_KEY, CHROMA_API_KEY, CHROMA_TENANT, and CHROMA_DATABASE are set in your .env file.")
        return

    # --- 2. INITIALIZE CLIENTS ---
    logging.info("Initializing OpenAI and ChromaDB clients...")
    openai_client = OpenAI(api_key=openai_api_key, max_retries=5, timeout=30.0)

    try:
        # Use the CloudClient as specified by your Chroma dashboard
        chroma_client = chromadb.CloudClient(
            api_key=chroma_api_key,
            tenant=chroma_tenant,
            database=chroma_database
        )
        # Verify connection by checking the heartbeat
        chroma_client.heartbeat()
        logging.info("Successfully connected to Chroma Cloud.")
    except Exception as e:
        logging.error(f"Failed to connect to Chroma Cloud: {e}")
        return

    # --- 3. CREATE OR GET THE COLLECTION ---
    logging.info(f"Creating or getting Chroma collection: '{CHROMA_COLLECTION_NAME}'")
    collection = chroma_client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)

    # --- 4. LOAD AND PROCESS LOCAL DATA ---
    try:
        logging.info(f"Loading document chunks from '{PKL_FILE_PATH}'...")
        with open(PKL_FILE_PATH, 'rb') as f:
            documents = pickle.load(f)
        texts_to_embed = [doc.page_content for doc in documents if hasattr(doc, 'page_content') and doc.page_content]
        if not texts_to_embed:
            logging.warning("No text content could be extracted from the pickle file. Exiting.")
            return
        logging.info(f"Successfully extracted {len(texts_to_embed)} non-empty text chunks.")
    except FileNotFoundError:
        logging.error(f"FATAL: Input file not found at '{PKL_FILE_PATH}'.")
        return
    except Exception as e:
        logging.error(f"Failed to load or process pickle file: {e}")
        return

    # --- 5. BATCH PROCESS AND UPLOAD TO CHROMA CLOUD ---
    total_texts = len(texts_to_embed)
    logging.info(f"Processing {total_texts} texts in batches of {EMBEDDING_BATCH_SIZE} to push to Chroma Cloud...")

    for i in tqdm(range(0, total_texts, EMBEDDING_BATCH_SIZE), desc="Uploading to Chroma Cloud"):
        batch_texts = texts_to_embed[i:i + EMBEDDING_BATCH_SIZE]
        # Create unique string IDs for this batch
        batch_ids = [str(idx) for idx in range(i, i + len(batch_texts))]
        
        try:
            # Generate embeddings for the current batch
            response = openai_client.embeddings.create(model=EMBEDDING_MODEL_NAME, input=batch_texts)
            batch_embeddings = [item.embedding for item in response.data]

            # Add this batch directly to the Chroma Cloud collection
            collection.add(
                embeddings=batch_embeddings,
                documents=batch_texts,
                ids=batch_ids
            )
        except Exception as e:
            logging.error(f"An error occurred for batch starting at index {i}: {e}")
            logging.warning(f"Skipping this batch due to an error.")

    # --- 6. FINAL VERIFICATION ---
    final_count = collection.count()
    print("\n" + "="*50)
    print("✅ Chroma Cloud Population Complete ✅")
    print(f"Your collection '{CHROMA_COLLECTION_NAME}' now contains {final_count} items.")
    print("You can now access this data from any application, including your Streamlit app.")
    print("="*50)

if __name__ == "__main__":
    main()