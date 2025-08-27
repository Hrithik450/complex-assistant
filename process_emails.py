import os
import json
import logging
import pandas as pd
from tqdm.asyncio import tqdm
from dotenv import load_dotenv
import pickle
import asyncio
import aiohttp
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError
import time

# --- FIX: Import missing libraries ---
import numpy as np
import faiss

# --- Import the necessary components from your existing, final process_data.py ---
from process_data import (
    initialize_services,
    UnifiedDataParser,
    intelligently_chunk_email_body,
    extract_email_metadata_from_pdf_text,
    get_local_embeddings,
    EMBEDDING_DIMENSION
)

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')

# --- NEW: Define dedicated output files for the email-only knowledge base ---
EMAIL_FAISS_INDEX_PATH = os.path.join(SCRIPT_DIR, "emails_faiss.bin")
EMAIL_METADATA_PATH = os.path.join(SCRIPT_DIR, "emails_metadata.pkl")

# --- PERFORMANCE TUNING ---
CONCURRENT_REQUESTS = 10
SUMMARIZATION_MODEL = "gpt-4o-mini"
PROCESSING_BATCH_SIZE = 5000

# --- Asynchronous Summarization Function (Unchanged) ---
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30)
)
async def generate_summary(session, api_key, email_body: str) -> str:
    """Uses an LLM to generate a one-sentence summary of an email body."""
    if not email_body or not isinstance(email_body, str) or len(email_body) < 50:
        return "No summary available."

    # Use only the first 4000 characters to avoid excessive token usage
    truncated_body = email_body[:4000]

    system_prompt = "You are a summarization engine. Your task is to create a concise, one-sentence summary of the following email content. Focus on the main topic and any key actions or decisions."
    
    async with session.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": SUMMARIZATION_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": truncated_body}
            ],
            "temperature": 0.0,
        }
    ) as response:
        response.raise_for_status()
        response_json = await response.json()
        return response_json["choices"][0]["message"]["content"].strip()

# --- THIS IS THE FIX: Refactored Parsers for Better Progress Tracking ---

async def process_jsonl_line(session, api_key, line, base_metadata):
    """Processes a single line from a JSONL file."""
    try:
        data = json.loads(line)
        body_text = data.get('body', {}).get('text', '') or data.get('snippet', '')
        if not body_text: return []
        
        summary = await generate_summary(session, api_key, body_text)
        
        email_metadata = base_metadata.copy()
        email_metadata.update({
            "id": data.get('id', 'N/A'), "threadId": data.get('threadId', 'N/A'),
            "from": data.get('from', 'N/A'), "to": data.get('to', 'N/A'),
            "cc": data.get('cc', 'N/A'), "bcc": data.get('bcc', 'N/A'),
            "subject": data.get('subject', 'N/A'),
            "parsed_date": UnifiedDataParser.parse_flexible_date(data.get('date')),
            "summary": summary
        })
        
        main_chunks, signature = intelligently_chunk_email_body(body_text)
        if signature: email_metadata['signature'] = signature
        
        return [(chunk, email_metadata) for chunk in main_chunks]
    except (json.JSONDecodeError, AttributeError, RetryError) as e:
        logging.warning(f"Skipping line due to error: {e}")
        return []

async def process_pdf_file(api_key, file_path):
    """Processes a single PDF file."""
    from unstructured.partition.auto import partition
    base_metadata = {'source': file_path}
    try:
        elements = partition(filename=file_path, strategy="fast")
        raw_chunks = [str(el) for el in elements]
        email_meta = extract_email_metadata_from_pdf_text(OpenAI(api_key=api_key), "\n".join(raw_chunks[:5]))
        if email_meta.get("is_email"):
            logging.info(f"    - Identified Email in PDF: {os.path.basename(file_path)}")
            base_metadata['file_type'] = 'email'
            base_metadata.update(email_meta)
            base_metadata['parsed_date'] = UnifiedDataParser.parse_flexible_date(email_meta.get('date'))
            return [(chunk, base_metadata) for chunk in raw_chunks]
    except Exception as e:
        logging.error(f"Could not partition PDF {os.path.basename(file_path)}: {e}")
    return []

async def main():
    """Main async function to orchestrate the email processing."""
    load_dotenv()
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logging.error("FATAL: OPENAI_API_KEY not found in .env file.")
        return

    _, embedding_model = initialize_services()

    if os.path.exists(EMAIL_FAISS_INDEX_PATH): os.remove(EMAIL_FAISS_INDEX_PATH)
    if os.path.exists(EMAIL_METADATA_PATH): os.remove(EMAIL_METADATA_PATH)

    # --- THIS IS THE FIX: Phase 1 - Pre-computation and Task Creation ---
    logging.info("[+] Phase 1: Calculating total workload...")
    files_to_process = [os.path.join(root, file) for root, _, files in os.walk(DATA_DIR) for file in files]
    
    tasks_to_run = []
    for file_path in files_to_process:
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext == '.jsonl':
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    tasks_to_run.append(("jsonl_line", line, {'source': file_path, 'file_type': 'email'}))
        elif file_ext == '.pdf':
            tasks_to_run.append(("pdf_file", file_path, {}))

    logging.info(f"[+] Total tasks to process (emails + PDFs): {len(tasks_to_run)}")
    
    all_embeddings = []
    metadata_store = []
    
    # --- Phase 2: Asynchronous Execution with Detailed Progress ---
    async with aiohttp.ClientSession() as session:
        pbar = tqdm(total=len(tasks_to_run), desc="Processing Emails")
        
        async def run_task(task_info):
            task_type, data, metadata = task_info
            if task_type == "jsonl_line":
                return await process_jsonl_line(session, api_key, data, metadata)
            elif task_type == "pdf_file":
                return await process_pdf_file(api_key, data)
        
        # Create a limited number of concurrent workers
        semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
        async def worker(task):
            async with semaphore:
                result = await run_task(task)
                pbar.update(1)
                return result

        results = await asyncio.gather(*(worker(task) for task in tasks_to_run))

    # --- Process all collected results ---
    for email_items in results:
        if not email_items: continue
        
        texts_to_embed = [item[0] for item in email_items]
        metadatas_to_store = [item[1] for item in email_items]
        
        embeddings = get_local_embeddings(embedding_model, texts_to_embed)
        if not embeddings: continue

        for i, embedding in enumerate(embeddings):
            metadata = metadatas_to_store[i]
            metadata['original_text'] = texts_to_embed[i]
            metadata_store.append(metadata)
            all_embeddings.append(embedding)

    if not all_embeddings:
        print("\n[!] No email content was extracted from any files.")
        return

    print(f"\n[*] Extracted a total of {len(all_embeddings)} high-quality email text chunks.")
    print(f"[*] Building email-only FAISS index...")

    np_embeddings = np.array(all_embeddings).astype('float32')
    index = faiss.IndexFlatL2(EMBEDDING_DIMENSION)
    index.add(np_embeddings)
    print(f"[+] FAISS index built successfully. Total vectors: {index.ntotal}")

    print(f"[*] Saving email FAISS index to '{EMAIL_FAISS_INDEX_PATH}'...")
    faiss.write_index(index, EMAIL_FAISS_INDEX_PATH)
    
    print(f"[*] Saving email metadata to '{EMAIL_METADATA_PATH}'...")
    with open(EMAIL_METADATA_PATH, "wb") as f:
        pickle.dump(metadata_store, f)

    print("\n" + "="*80)
    print("               ✅ EMAIL PROCESSING COMPLETE ✅")
    print("="*80)
    print("\nYour new, email-only database with summaries is ready.")
    print("="*80)

if __name__ == "__main__":
    asyncio.run(main())