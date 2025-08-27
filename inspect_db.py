import os
import pickle
import pandas as pd
from collections import Counter

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
COLLECTION_NAME = "real_estate_finetuned_local"
# METADATA_PATH = os.path.join(SCRIPT_DIR, f"{COLLECTION_NAME}_metadata.pkl")
# FAISS_INDEX_PATH = os.path.join(SCRIPT_DIR, f"{COLLECTION_NAME}_faiss.bin")

METADATA_PATH = os.path.join(SCRIPT_DIR, "emails_metadata.pkl")
FAISS_INDEX_PATH = os.path.join(SCRIPT_DIR, "emails_faiss.bin")



def inspect_database():
    """
    Loads the processed database files and generates a comprehensive
    report on its contents, including the head of the DataFrame to show its structure.
    """
    print("[*] Attempting to load database files...")
    if not os.path.exists(METADATA_PATH) or not os.path.exists(FAISS_INDEX_PATH):
        print(f"[!] FATAL: Database files not found. Please run 'process_data.py' first.")
        return

    with open(METADATA_PATH, "rb") as f:
        metadata_store = pickle.load(f)
    
    df = pd.DataFrame(metadata_store)
    print(f"[+] Successfully loaded metadata store with {len(df)} total records.\n")

    # --- Main Report ---
    print("\n" + "="*80)
    print("                  DATABASE INSPECTION REPORT")
    print("="*80)

    # --- NEW: SECTION 0 - DATAFRAME HEAD ---
    # This section prints the first 5 rows to show all columns and the overall structure.
    print("\n## 0. Metadata DataFrame Head (First 5 Rows)\n")
    if not df.empty:
        # Use to_string() to ensure all columns are displayed in the console
        print(df.head().to_string())
    else:
        print("  - The metadata DataFrame is empty.")

    # 1. File Type Distribution
    print("\n\n## 1. File Type Distribution\n")
    if 'file_type' in df.columns:
        file_type_counts = df['file_type'].value_counts()
        print(file_type_counts.to_string())
    else:
        print("  - 'file_type' column not found.")

    # 2. Date Parsing Validation
    print("\n\n## 2. Date Parsing Validation\n")
    if 'parsed_date' in df.columns:
        valid_dates = df['parsed_date'].notna().sum()
        total_records = len(df)
        success_rate = (valid_dates / total_records) * 100 if total_records > 0 else 0
        print(f"  - Records with a valid 'parsed_date': {valid_dates} / {total_records} ({success_rate:.2f}%)")
    else:
        print("  - 'parsed_date' column not found.")

    # 3. Special Report: PDF-Sourced Emails
    print("\n\n## 3. Special Report: PDF-Sourced Emails\n")
    if 'file_type' in df.columns and 'source' in df.columns:
        pdf_email_mask = (df['file_type'] == 'email') & (df['source'].str.endswith('.pdf', na=False))
        pdf_emails_df = df[pdf_email_mask]

        if not pdf_emails_df.empty:
            example = pdf_emails_df.iloc[0]
            print("  - Found at least one PDF that was successfully identified as an email.")
            print("  - Displaying the first example found:\n")
            
            print("-" * 50)
            print(f"### Example for: PDF-SOURCED EMAIL\n")
            
            print("--- METADATA (LLM EXTRACTED) ---")
            print(f"  - source      : {example.get('source', 'N/A')}")
            print(f"  - is_email    : {example.get('is_email', 'N/A')}")
            print(f"  - from        : {example.get('from', 'N/A')}")
            print(f"  - to          : {example.get('to', 'N/A')}")
            print(f"  - subject     : {example.get('subject', 'N/A')}")
            print(f"  - parsed_date : {example.get('parsed_date', 'N/A')}")
            print(f"  - summary     : {example.get('summary', 'N/A')}")

            print("\n--- TEXT CHUNK ---")
            print(f"  - {example.get('original_text', 'N/A')[:200]}...")
            print("-" * 50 + "\n")
        else:
            print("  - No PDFs were successfully identified as emails in the current database.")
    else:
        print("  - 'file_type' or 'source' column not found, cannot generate this report.")


    # 4. General Examples by File Type
    print("\n\n## 4. General Examples by File Type\n")
    if 'file_type' in df.columns:
        unique_types = [t for t in df['file_type'].unique() if t is not None]
        
        for f_type in unique_types:
            example = df[df['file_type'] == f_type].iloc[0]
            
            print("-" * 50)
            print(f"### Example for: {str(f_type).upper()}\n")
            
            print("--- METADATA ---")
            if f_type == 'email' and not example['source'].endswith('.pdf'):
                print(f"  - source      : {example.get('source', 'N/A')}")
                print(f"  - id          : {example.get('id', 'N/A')}")
                print(f"  - threadId    : {example.get('threadId', 'N/A')}")
                print(f"  - from        : {example.get('from', 'N/A')}")
                print(f"  - subject     : {example.get('subject', 'N/A')}")
                print(f"  - parsed_date : {example.get('parsed_date', 'N/A')}")
            elif f_type == 'whatsapp':
                print(f"  - source      : {example.get('source', 'N/A')}")
                print(f"  - sender      : {example.get('sender', 'N/A')}")
                print(f"  - parsed_date : {example.get('parsed_date', 'N/A')}")
            elif not (f_type == 'email' and example['source'].endswith('.pdf')):
                print(f"  - source      : {example.get('source', 'N/A')}")
                print(f"  - file_type   : {example.get('file_type', 'N/A')}")

            print("\n--- TEXT CHUNK ---")
            print(f"  - {example.get('original_text', 'N/A')[:200]}...")
            print("-" * 50 + "\n")
    else:
        print("  - 'file_type' column not found, cannot generate examples.")

    print("="*80)
    print("                  END OF REPORT")
    print("="*80)


if __name__ == "__main__":
    inspect_database()