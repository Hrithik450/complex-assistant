import os
import pickle
import numpy as np
import faiss

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# These names must match the final collection names used in your process_data.py script
# COLLECTION_NAME = "real_estate_finetuned_local"
# FAISS_INDEX_PATH = os.path.join(SCRIPT_DIR, "finetuned_faiss_index.bin")
# METADATA_PATH = os.path.join(SCRIPT_DIR, "real_estate_finetuned_local_metadata.pkl")
FAISS_INDEX_PATH = os.path.join(SCRIPT_DIR, "real_estate_finetuned_local_faiss.bin")
METADATA_PATH = os.path.join(SCRIPT_DIR, "real_estate_finetuned_local_metadata.pkl")

def print_record_details(record_index, metadata, vector):
    """Helper function to print the details of a single record in a formatted way."""
    print("-" * 50)
    print(f"Displaying Record at Index: {record_index}")
    
    print("\n--- METADATA ---")
    if not metadata:
        print("  [!] This record has NO METADATA.")
    else:
        for key, value in metadata.items():
            # Truncate very long text for readability
            if key == 'original_text' and isinstance(value, str) and len(value) > 250:
                print(f"  - {key:<20}: '{value[:250].replace(chr(10), ' ')}...'")
            else:
                print(f"  - {key:<20}: {value}")

    print("\n--- EMBEDDING ---")
    if vector is not None and vector.size > 0:
        print(f"  - Shape:               {vector.shape}")
        print(f"  - Data Type:           {vector.dtype}")
        # Show a snippet of the embedding vector
        snippet = f"[{vector[0]:.4f}, {vector[1]:.4f}, ..., {vector[-2]:.4f}, {vector[-1]:.4f}]"
        print(f"  - Vector Snippet:      {snippet}")
    else:
        print("  [!] Could not retrieve a valid vector for this index.")
    
    print("-" * 50)


def inspect_database():
    """Loads and inspects the FAISS index and metadata store to find examples of each file type."""
    print(f"[*] Attempting to load database files for collection...")
    
    try:
        faiss_index = faiss.read_index(FAISS_INDEX_PATH)
        with open(METADATA_PATH, "rb") as f:
            metadata_store = pickle.load(f)
        print(f"[+] Successfully loaded FAISS index and metadata store.")
    except FileNotFoundError:
        print(f"\n[!] FATAL: Database files not found. Please run the final 'process_data.py' script first to create them.")
        return
    except Exception as e:
        print(f"\n[!] FATAL: Could not load the database files. Error: {e}")
        return

    total_items = len(metadata_store)
    if total_items == 0:
        print("\n[!] The database is EMPTY.")
        return
        
    print(f"\n[*] The database contains {total_items} total items (vectors).")
    print("[*] Searching for one example of each major document type...")

    # --- Find examples of each type (THE FIX: Added JSONL Email) ---
    types_to_find = {
        "JSONL Email": None,
        "WhatsApp Chat": None,
        "Email in PDF": None,
        "Generic PDF": None,
        "DOCX Document": None,
        "Spreadsheet": None
    }
    
    for i, metadata in enumerate(metadata_store):
        # Stop searching if all types have been found
        if all(v is not None for v in types_to_find.values()):
            break
            
        source = metadata.get('source', '').lower()
        filename = os.path.basename(source)
        
        # Check for JSONL emails first
        if types_to_find["JSONL Email"] is None and source.endswith('.jsonl'):
            types_to_find["JSONL Email"] = i
            continue

        if types_to_find["WhatsApp Chat"] is None and filename.startswith('whatsapp chat with'):
            types_to_find["WhatsApp Chat"] = i
            continue
            
        if types_to_find["Email in PDF"] is None and source.endswith('.pdf') and metadata.get('is_email'):
            types_to_find["Email in PDF"] = i
            continue

        if types_to_find["Generic PDF"] is None and source.endswith('.pdf') and not metadata.get('is_email'):
            types_to_find["Generic PDF"] = i
            continue
            
        if types_to_find["DOCX Document"] is None and source.endswith('.docx'):
            types_to_find["DOCX Document"] = i
            continue

        if types_to_find["Spreadsheet"] is None and (source.endswith('.xlsx') or source.endswith('.csv')):
            types_to_find["Spreadsheet"] = i
            continue

    # --- Print the report ---
    print("\n" + "="*80)
    print("                  DATABASE INSPECTION REPORT")
    print("="*80)

    for doc_type, record_index in types_to_find.items():
        print(f"\n\n## Example for: {doc_type}")
        if record_index is not None:
            metadata = metadata_store[record_index]
            # FAISS can reconstruct a vector from its index
            vector = faiss_index.reconstruct(record_index)
            print_record_details(record_index, metadata, vector)
        else:
            print("  - No example of this document type was found in the database.")
            
    print("\n" + "="*80)

if __name__ == "__main__":
    inspect_database()