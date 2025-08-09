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
from datetime import datetime

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

# --- UNIFIED DATE PARSER ---
class UnifiedDataParser:
    @staticmethod
    def parse_flexible_date(date_string: str):
        if not isinstance(date_string, str): return None
        formats_to_try = [
            # Formats from JSONL files
            '%Y-%m-%dT%H:%M:%SZ',          # e.g., "2019-11-20T13:36:36Z"
            
            # Formats from PDF email headers
            '%A, %d %B, %Y %I.%M %p',      # e.g., "Tuesday, 8 July, 2025 6.15 PM"
            
            # Formats from PDF email reply chains
            '%a, %b %d, %Y at %I:%M %p',   # e.g., "On Mon, Jul 7, 2025 at 9:37 PM"
            '%a, %d %b %Y at %I:%M %p',   # e.g., "On Tue, 8 Jul 2025 at 6:07 PM"
            '%a, %d %b, %Y, %I:%M %p',    # e.g., "On Fri, 13 Dec, 2024, 6:05 pm"

            # Formats from WhatsApp
            '%m/%d/%y, %I:%M\u202f%p',    # Handles '9/18/23, 2:10 PM' with special space
            '%m/%d/%y, %H:%M',           # Handles '3/13/24, 16:09'
            '%d/%m/%y, %H:%M',           # Handles '12/6/22, 09:46'
            
            # Generic formats
            '%Y-%m-%dT%H:%M:%S', 
            '%Y-%m-%d',
            '%a, %d %b %Y %H:%M:%S %z'
        ]
        for fmt in formats_to_try:
            try:
                dt_object = datetime.strptime(date_string.strip(), fmt)
                return dt_object.strftime('%Y-%m-%d %H:%M:%S')
            except (ValueError, TypeError):
                continue
        return None

# --- DATA QUALITY CONTROL ---
def filter_and_split_chunks(raw_chunks):
    """Filters out junk and splits oversized chunks."""
    tokenizer = tiktoken.get_encoding("cl100k_base")
    JUNK_PHRASES = [
        "messages and calls are end-to-end encrypted", "this message was deleted",
        "<media omitted>", "created group", "added you", "you're now an admin",
        "tap to learn more", "sent from my iphone", "kind regards", "best regards"
    ]
    final_chunks = []
    for chunk in raw_chunks:
        stripped_chunk = chunk.strip()
        if len(stripped_chunk) < 25 or any(phrase in stripped_chunk.lower() for phrase in JUNK_PHRASES):
            continue
        tokens = tokenizer.encode(stripped_chunk)
        if len(tokens) > CHUNK_SIZE_TOKENS:
            for i in range(0, len(tokens), CHUNK_SIZE_TOKENS - CHUNK_OVERLAP_TOKENS):
                sub_chunk_tokens = tokens[i:i + CHUNK_SIZE_TOKENS]
                final_chunks.append(tokenizer.decode(sub_chunk_tokens))
        else:
            final_chunks.append(stripped_chunk)
    return final_chunks

# --- INTELLIGENT EMAIL BODY CHUNKER ---
def intelligently_chunk_email_body(body_text):
    """
    Separates the main content of an email from the signature and reply chain,
    then chunks only the main content to avoid noise.
    """
    if not isinstance(body_text, str):
        return [], None

    signature_keywords = [r'thanks,', r'thank you,', r'best regards,', r'sincerely,', r'kind regards,']
    reply_keywords = [r'On\s.+wrote:', r'From:', r'Sent:', r'Date:', r'Subject:', r'To:']
    
    main_content = body_text
    signature = None

    for keyword in signature_keywords:
        parts = re.split(f'(\n\s*{keyword}.*)', main_content, 1, re.IGNORECASE)
        if len(parts) > 1:
            main_content = parts[0]
            signature = parts[1].strip()
            break

    for keyword in reply_keywords:
        parts = re.split(f'(\n\s*{keyword}.*)', main_content, 1, re.IGNORECASE)
        if len(parts) > 1:
            main_content = parts[0]
            break

    main_content_chunks = filter_and_split_chunks([main_content.strip()])
    
    return main_content_chunks, signature

# --- PDF METADATA EXTRACTOR ---
def extract_email_metadata_from_pdf_text(client, text_sample):
    """Updated to extract more fields: cc, bcc, summary, signature."""
    system_prompt = """
    You are a data extraction engine. Analyze the text from a PDF.
    Determine if it is an email and extract its metadata into a single, valid JSON object.
    
    RULES:
    - Your entire response MUST be ONLY a valid JSON object.
    - If the text is an email, return a JSON with "is_email": true and populate all other fields.
    - If a field like 'cc' or 'bcc' is not present, omit it or set its value to an empty string.
    - If it is NOT an email, return a JSON with only one key: {"is_email": false}.
    
    JSON Schema for Emails:
    {"is_email": true, "from": "...", "to": "...", "cc": "...", "bcc": "...", "subject": "...", "date": "...", "summary": "A concise one-sentence summary.", "signature": "..."}
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

# --- MASTER FILE PARSER ---
def get_items_from_file(client, file_path):
    """
    Master function to process a single file. It determines the file type,
    extracts all relevant metadata, and returns a list of (text, metadata) tuples.
    """
    file_ext = os.path.splitext(file_path)[1].lower()
    filename = os.path.basename(file_path)
    base_metadata = {'source': file_path}
    
    try:
        items_to_return = []

        if file_ext == '.jsonl':
            base_metadata['file_type'] = 'email'
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        body_text = data.get('body', {}).get('text', '')
                        if not body_text: continue
                        
                        email_metadata = base_metadata.copy()
                        
                        # --- THIS IS THE CHANGE: Add email and thread IDs ---
                        email_metadata.update({
                            "id": data.get('id', 'N/A'),
                            "threadId": data.get('threadId', 'N/A'),
                            "from": data.get('from', 'N/A'),
                            "to": data.get('to', 'N/A'),
                            "cc": data.get('cc', 'N/A'),
                            "bcc": data.get('bcc', 'N/A'),
                            "subject": data.get('subject', 'N/A'),
                            "parsed_date": UnifiedDataParser.parse_flexible_date(data.get('date'))
                        })
                        
                        main_chunks, signature = intelligently_chunk_email_body(body_text)
                        if signature:
                            email_metadata['signature'] = signature
                        
                        for chunk in main_chunks:
                            items_to_return.append((chunk, email_metadata))
                    except (json.JSONDecodeError, AttributeError):
                        continue
            return items_to_return

        if file_ext == '.txt' and (filename.lower().startswith('whatsapp chat with') or 'whatsapp' in filename.lower()):
            base_metadata['file_type'] = 'whatsapp'
            # This regex is now more robust to handle different date formats and system messages
            chat_pattern = re.compile(
                r"^(\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}(?:\u202f[ap]m)?)\s-\s([^:]+):\s(.*)", 
                re.IGNORECASE | re.DOTALL
            )
            system_message_pattern = re.compile(r"(\bcreated group\b|\badded you\b|\bchanged the subject\b)")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split the file by the date pattern to handle multi-line messages correctly
            # The regex `(?=...)` is a positive lookahead, which splits the text without consuming the delimiter
            messages = re.split(r'(?=\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2})', content)
            
            for message_block in messages:
                if not message_block.strip(): continue
                match = chat_pattern.match(message_block)
                if match:
                    timestamp, sender, message = match.groups()
                    
                    # Filter out system messages and junk
                    if system_message_pattern.search(message):
                        continue
                    
                    msg_metadata = base_metadata.copy()
                    msg_metadata.update({
                        'sender': sender.strip(),
                        'parsed_date': UnifiedDataParser.parse_flexible_date(timestamp.strip())
                    })
                    items_to_return.append((message.strip(), msg_metadata))
            return items_to_return

        # Generic Handlers
        raw_chunks = []
        if file_ext == '.pdf':
            base_metadata['file_type'] = 'pdf'
            try:
                elements = partition(filename=file_path, strategy="fast")
                raw_chunks = [str(el) for el in elements]
                email_meta = extract_email_metadata_from_pdf_text(client, "\n".join(raw_chunks[:5]))
                if email_meta.get("is_email"):
                    print(f"    - Detected Email in PDF: {filename}")
                    base_metadata['file_type'] = 'email'
                    base_metadata.update(email_meta)
                    base_metadata['parsed_date'] = UnifiedDataParser.parse_flexible_date(email_meta.get('date'))
            except Exception as e:
                logging.error(f"Could not partition PDF {filename}: {e}")
        
        elif file_ext == '.docx':
            base_metadata['file_type'] = 'docx'
            raw_chunks = [str(el) for el in partition(filename=file_path, strategy="fast")]
        elif file_ext in {'.xlsx', '.csv'}:
            base_metadata['file_type'] = 'spreadsheet'
            df = pd.read_csv(file_path) if file_ext == '.csv' else pd.read_excel(file_path, engine='openpyxl')
            raw_chunks = [f"Row {idx+1}: {', '.join(f'{col}: {val}' for col, val in row.astype(str).items())}" for idx, row in df.iterrows()]
        else:
            return []
        
        return [(chunk, base_metadata) for chunk in raw_chunks]

    except Exception as e:
        logging.error(f"Could not process file {file_path}: {e}")
        return []

# --- UTILITIES ---
def get_local_embeddings(model, chunks):
    try:
        return model.encode(chunks, show_progress_bar=False).tolist()
    except Exception as e:
        logging.error(f"Failed to get local embeddings: {e}")
        return []

# --- MAIN EXECUTION ---
def main():
    data_dir = os.path.join(SCRIPT_DIR, 'data')
    if not os.path.isdir(data_dir):
        print(f"[!] FATAL: Data folder not found at '{data_dir}'")
        return
    
    client, embedding_model = initialize_services()

    if os.path.exists(FAISS_INDEX_PATH): os.remove(FAISS_INDEX_PATH)
    if os.path.exists(METADATA_PATH): os.remove(METADATA_PATH)

    files_to_process = [os.path.join(root, file) for root, _, files in os.walk(data_dir) for file in files]
    print(f"[+] Found {len(files_to_process)} total files to process.")
    
    all_embeddings = []
    metadata_store = []

    pbar = tqdm(files_to_process, desc="Processing Files", unit="file")
    for file_path in pbar:
        pbar.set_postfix_str(os.path.basename(file_path))
        
        raw_items = get_items_from_file(client, file_path)
        if not raw_items: continue
        
        final_items = []
        for text, metadata in raw_items:
            if metadata.get('file_type') != 'email':
                clean_chunks = filter_and_split_chunks([text])
                for chunk in clean_chunks:
                    final_items.append((chunk, metadata))
            else:
                final_items.append((text, metadata))

        if not final_items: continue
        
        texts_to_embed = [item[0] for item in final_items]
        metadatas_to_store = [item[1] for item in final_items]
        
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
    print(f"[*] Your new, expert database files are ready.")

if __name__ == "__main__":
    main()