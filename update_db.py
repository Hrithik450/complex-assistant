import os
import pickle
import faiss
import numpy as np
from tqdm import tqdm
import logging

# --- IMPORTANT: Reuse the exact same processing logic ---
# This ensures consistency between a full build and an update.
from process_data import (
    initialize_services,
    get_items_from_file,
    get_local_embeddings,
    filter_and_split_chunks,
    FAISS_INDEX_PATH,
    METADATA_PATH,
    EMBEDDING_DIMENSION
)

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')

def update_database():
    """
    Intelligently updates the existing FAISS index and metadata store.
    - Processes only new files.
    - Removes data for deleted files.
    """
    # 1. Load Existing Database
    # --------------------------------------------------
    logging.info("[*] Loading existing database...")
    if not os.path.exists(METADATA_PATH) or not os.path.exists(FAISS_INDEX_PATH):
        logging.error("[!] FATAL: No existing database found. Run 'process_data.py' to create one first.")
        return

    try:
        faiss_index = faiss.read_index(FAISS_INDEX_PATH)
        with open(METADATA_PATH, "rb") as f:
            metadata_store = pickle.load(f)
        logging.info(f"[+] Successfully loaded database with {faiss_index.ntotal} vectors.")
    except Exception as e:
        logging.error(f"[!] FATAL: Could not load database files. Error: {e}")
        return

    # 2. Scan for File Changes
    # --------------------------------------------------
    logging.info("[*] Scanning for file changes...")
    processed_files = set(item['source'] for item in metadata_store)
    current_files = set(os.path.join(root, file) for root, _, files in os.walk(DATA_DIR) for file in files)

    new_files = sorted(list(current_files - processed_files))
    deleted_files = sorted(list(processed_files - current_files))

    if not new_files and not deleted_files:
        logging.info("[+] Database is already up-to-date. No changes needed.")
        return

    logging.info(f"[*] Found {len(new_files)} new file(s) and {len(deleted_files)} deleted file(s).")

    # 3. Process Deletions
    # --------------------------------------------------
    if deleted_files:
        logging.info(f"[*] Removing data for {len(deleted_files)} deleted file(s)...")
        
        ids_to_remove = [i for i, meta in enumerate(metadata_store) if meta['source'] in deleted_files]
        
        if ids_to_remove:
            # FAISS requires a NumPy array of int64 for removal
            faiss_index.remove_ids(np.array(ids_to_remove, dtype=np.int64))
            
            # Rebuild the metadata store, excluding the deleted items
            ids_to_remove_set = set(ids_to_remove)
            metadata_store = [meta for i, meta in enumerate(metadata_store) if i not in ids_to_remove_set]
            
            logging.info(f"[+] Removed {len(ids_to_remove)} vectors from the database.")
        else:
            logging.warning("[!] Mismatch: Deleted files were identified but no corresponding vectors were found to remove.")

    # 4. Process Additions
    # --------------------------------------------------
    if new_files:
        logging.info(f"[*] Processing {len(new_files)} new file(s)...")
        client, embedding_model = initialize_services()
        
        new_embeddings = []
        new_metadata = []

        pbar = tqdm(new_files, desc="Processing New Files", unit="file")
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
                new_embeddings.append(embedding)
                metadata = metadatas_to_store[i]
                metadata['original_text'] = texts_to_embed[i]
                new_metadata.append(metadata)

        if new_embeddings:
            logging.info(f"[*] Adding {len(new_embeddings)} new vectors to the index...")
            np_embeddings = np.array(new_embeddings).astype('float32')
            faiss_index.add(np_embeddings)
            metadata_store.extend(new_metadata)
            logging.info("[+] New data successfully added.")

    # 5. Save Updated Database
    # --------------------------------------------------
    logging.info("[*] Saving updated database files...")
    faiss.write_index(faiss_index, FAISS_INDEX_PATH)
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata_store, f)

    logging.info("\n[+] --- UPDATE COMPLETE ---")
    logging.info(f"[*] Your database is now up-to-date with {faiss_index.ntotal} total vectors.")


if __name__ == "__main__":
    update_database()