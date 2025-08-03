import os
import zipfile
import pandas as pd
from dotenv import load_dotenv
import logging
import re
import shutil
import json
from tqdm import tqdm
import pickle
import numpy as np
import tiktoken
import faiss
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from unstructured.partition.auto import partition
from unstructured.partition.email import partition_email
import tiktoken

# --- CONFIGURATION ---
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- MODEL AND DB CONFIGURATION ---
FINETUNED_MODEL_PATH = os.path.join(SCRIPT_DIR, "finetuned_bge_real_estate_model")
EMBEDDING_DIMENSION = 768
COLLECTION_NAME = "real_estate_finetuned_local"

FAISS_INDEX_PATH = os.path.join(SCRIPT_DIR, f"{COLLECTION_NAME}_faiss.bin")
METADATA_PATH = os.path.join(SCRIPT_DIR, f"{COLLECTION_NAME}_metadata.pkl")
CHUNK_SIZE_TOKENS = 512
CHUNK_OVERLAP_TOKENS = 50

# --- INITIALIZATION ---
def initialize_services():
    """Load the fine-tuned local embedding model and OpenAI client."""
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if not client.api_key: raise ValueError("FATAL: OPENAI_API_KEY not found.")
    print("[*] OpenAI client initialized.")

    if not os.path.isdir(FINETUNED_MODEL_PATH):
        raise FileNotFoundError(f"FATAL: Fine-tuned model not found at '{FINETUNED_MODEL_PATH}'.")
    
    print(f"[*] Loading fine-tuned embedding model...")
    embedding_model = SentenceTransformer(FINETUNED_MODEL_PATH, device='cpu')
    print("[+] Fine-tuned model loaded.")
    return client, embedding_model

# --- DATA QUALITY CONTROL ---
def filter_and_split_chunks(raw_chunks):
    """
    Filters out junk and splits oversized chunks to ensure data quality.
    
    Args:
        raw_chunks (list[str]): A list of raw text strings extracted from a document.

    Returns:
        list[str]: A list of clean, correctly sized text chunks ready for embedding.
    """
    # Use the tokenizer that corresponds to modern OpenAI and many open-source models
    tokenizer = tiktoken.get_encoding("cl100k_base")
    
    # Define a list of common, low-value phrases to filter out.
    # This list can be expanded with more domain-specific junk phrases.
    JUNK_PHRASES = [
        "messages and calls are end-to-end encrypted",
        "this message was deleted",
        "<media omitted>",
        "created group",
        "added you",
        "you're now an admin",
        "tap to learn more"
    ]
    
    final_chunks = []
    
    for chunk in raw_chunks:
        # First, strip any leading/trailing whitespace from the chunk
        stripped_chunk = chunk.strip()
        
        # 1. Filter out junk chunks
        # Check if the chunk is too short or contains a known junk phrase.
        if len(stripped_chunk) < 25 or any(phrase in stripped_chunk.lower() for phrase in JUNK_PHRASES):
            continue # Skip this chunk entirely
            
        # 2. Split oversized chunks
        # Encode the chunk into tokens to measure its length
        tokens = tokenizer.encode(stripped_chunk)
        
        if len(tokens) > CHUNK_SIZE_TOKENS:
            # If the chunk is too long, split it into overlapping sub-chunks
            for i in range(0, len(tokens), CHUNK_SIZE_TOKENS - CHUNK_OVERLAP_TOKENS):
                sub_chunk_tokens = tokens[i:i + CHUNK_SIZE_TOKENS]
                # Decode the tokens back into a string and add to our final list
                final_chunks.append(tokenizer.decode(sub_chunk_tokens))
        else:
            # If the chunk is already a good size, add it directly
            final_chunks.append(stripped_chunk)
            
    return final_chunks

# --- INTELLIGENT PDF PARSER ---
def extract_email_metadata_from_pdf_text(client, text_sample):
    system_prompt = """
    You are a data extraction engine. Analyze the text from the first page of a PDF.
    Your task is to determine if it is an email and extract its metadata into a single, valid JSON object.
    
    RULES:
    - Your entire response MUST be ONLY a valid JSON object.
    - If the text is an email, return a JSON with "is_email": true and populate the other fields.
    - If it is NOT an email, return a JSON with only one key: {"is_email": false}.
    
    JSON Schema for Emails:
    {"is_email": true, "from": "...", "to": "...", "subject": "...", "date": "...", "summary": "...", "signature": "..."}
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": text_sample[:4000]}],
            temperature=0.0, response_format={"type": "json_object"})
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        logging.error(f"Could not analyze PDF text with LLM. Error: {e}")
        return {"is_email": False}
    

# --- SPECIALIZED FILE PARSERS (THE DEFINITIVE FIX) ---
def get_chunks_and_metadata(client, file_path):
    """Processes a file, intelligently extracts chunks and metadata."""
    file_ext = os.path.splitext(file_path)[1].lower()
    filename = os.path.basename(file_path)
    extraction_dir = os.path.join(SCRIPT_DIR, 'data', 'extracted')
    base_metadata = {'source': os.path.relpath(file_path, extraction_dir)}
    
    try:
        items_to_return = []
        file_level_metadata = base_metadata.copy()

        if file_ext == '.jsonl':
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        body_dict = data.get('body')
                        if not isinstance(body_dict, dict): continue

                        # Unpack the text from the body dictionary
                        text_content = body_dict.get('text', '')
                        if not text_content: continue
                        
                        # Extract metadata from the top-level data object
                        email_metadata = base_metadata.copy()
                        to_field = data.get('to', [])
                        cc_field = data.get('cc', [])
                        email_metadata.update({
                            "from": data.get('from', 'N/A'),
                            "to": ", ".join(to_field) if isinstance(to_field, list) else str(to_field),
                            "cc": ", ".join(cc_field) if isinstance(cc_field, list) else str(cc_field),
                            "subject": data.get('subject', 'N/A'),
                            "timestamp": data.get('timestamp', 'N/A')
                        })
                        items_to_return.append((text_content, email_metadata))
                    except (json.JSONDecodeError, AttributeError):
                        continue
            return items_to_return

        if file_ext == '.txt' and filename.lower().startswith('whatsapp chat with'):
            chat_pattern = re.compile(r"^(\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}(?:\s[ap]m)?)\s-\s([^:]+):\s(.*)", re.IGNORECASE)
            current_message_text = ""
            current_msg_meta = {}
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    match = chat_pattern.match(line)
                    if match:
                        if current_message_text:
                            items_to_return.append((current_message_text, current_msg_meta))
                        timestamp, sender, message = match.groups()
                        current_message_text = message.strip()
                        current_msg_meta = base_metadata.copy()
                        current_msg_meta['sender'] = sender.strip()
                        current_msg_meta['timestamp'] = timestamp.strip()
                    else:
                        current_message_text += "\n" + line.strip()
            if current_message_text:
                items_to_return.append((current_message_text, current_msg_meta))
            
            final_items = []
            for text, metadata in items_to_return:
                clean_chunks = filter_and_split_chunks([text])
                for clean_chunk in clean_chunks:
                    final_items.append((clean_chunk, metadata))
            return final_items

        elif file_ext == '.pdf':
            try:
                first_page_elements = partition(filename=file_path, strategy="fast", max_pages=1)
                first_page_text = "\n".join([str(el) for el in first_page_elements])
                email_metadata = extract_email_metadata_from_pdf_text(client, first_page_text)
                if email_metadata.get("is_email"):
                    print(f"    - Detected Email in PDF: {filename}")
                    file_level_metadata.update(email_metadata)
            except Exception as e:
                logging.error(f"Could not perform PDF pre-check on {filename}: {e}")
            raw_chunks = [str(el) for el in partition(filename=file_path, strategy="fast")]
        
        elif file_ext in {'.docx', '.txt'}:
            raw_chunks = [str(el) for el in partition(filename=file_path, strategy="fast")]
        elif file_ext in {'.xlsx', '.csv'}:
            df = pd.read_csv(file_path) if file_ext == '.csv' else pd.read_excel(file_path, engine='openpyxl')
            raw_chunks = [f"Row {idx+1}: {', '.join(f'{col}: {val}' for col, val in row.astype(str).items())}" for idx, row in df.iterrows()]
        else:
            return []
        
        return [(chunk, file_level_metadata) for chunk in raw_chunks]

    except Exception as e:
        logging.error(f"Could not process file {file_path}: {e}")
        return []
    
# --- UTILITIES ---
def sanitize_filename(filename):
    filename = filename.strip()
    return re.sub(r'[<>:"/\\|?*]', '_', filename).rstrip('. ')

# --- UTILITIES ---
def get_local_embeddings(model, chunks):
    try:
        return model.encode(chunks, show_progress_bar=False).tolist()
    except Exception as e:
        logging.error(f"Failed to get local embeddings: {e}")
        return []

# --- MAIN EXECUTION (Refactored for Correctness) ---
def main():
    extraction_dir = os.path.join(SCRIPT_DIR, 'data', 'extracted')
    if not os.path.isdir(extraction_dir):
        print(f"[!] FATAL: Extracted data folder not found at '{extraction_dir}'")
        return
    
    client, embedding_model = initialize_services()

    if os.path.exists(FAISS_INDEX_PATH): os.remove(FAISS_INDEX_PATH)
    if os.path.exists(METADATA_PATH): os.remove(METADATA_PATH)

    files_to_process = [os.path.join(root, file) for root, _, files in os.walk(extraction_dir) for file in files]
    print(f"[+] Found {len(files_to_process)} total files to process.")
    
    all_embeddings = []
    metadata_store = []

    pbar = tqdm(files_to_process, desc="Processing Files", unit="file")
    for file_path in pbar:
        pbar.set_postfix_str(os.path.basename(file_path))
        
        # The client is now correctly passed for PDF email detection
        items_from_file = get_chunks_and_metadata(client, file_path)
        if not items_from_file: continue
        
        # The quality filtering is now applied to the items
        final_items = []
        for text, metadata in items_from_file:
            clean_chunks = filter_and_split_chunks([text])
            for clean_chunk in clean_chunks:
                final_items.append((clean_chunk, metadata))

        if not final_items: continue
        
        texts_to_embed = [item[0] for item in final_items]
        metadatas_to_store = [item[1] for item in final_items]
        
        # The correct embedding_model is now passed
        embeddings = get_local_embeddings(embedding_model, texts_to_embed)
        if not embeddings: continue

        for i, embedding in enumerate(embeddings):
            all_embeddings.append(embedding)
            metadata = metadatas_to_store[i]
            metadata['original_text'] = texts_to_embed[i]
            metadata_store.append(metadata)

    if not all_embeddings:
        print("\n[!] No text chunks were extracted from any files.")
        return

    print(f"\n[*] Extracted a total of {len(all_embeddings)} high-quality text chunks.")
    print(f"[*] Building FAISS index with dimension {EMBEDDING_DIMENSION}...")

    np_embeddings = np.array(all_embeddings).astype('float32')
    index = faiss.IndexFlatL2(EMBEDDING_DIMENSION)
    index.add(np_embeddings)
    print(f"[+] FAISS index built successfully. Total vectors: {index.ntotal}")

    print(f"[*] Saving FAISS index to '{FAISS_INDEX_PATH}'...")
    faiss.write_index(index, FAISS_INDEX_PATH)
    
    print(f"[*] Saving metadata to '{METADATA_PATH}'...")
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata_store, f)

    print(f"\n[+] --- SCRIPT COMPLETE ---")
    print(f"[*] Your new, expert database files ('{os.path.basename(FAISS_INDEX_PATH)}' and '{os.path.basename(METADATA_PATH)}') are ready.")

if __name__ == "__main__":
    main()