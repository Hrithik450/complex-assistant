import os
import argparse
import logging
from collections import defaultdict
import pandas as pd
from unstructured.partition.auto import partition

# --- CONFIGURATION ---
# Configure logging to be minimal for this script
logging.basicConfig(level=logging.ERROR) # Set to ERROR to hide unstructured's INFO messages

# Define file extension categories
TEXT_EXTENSIONS = {'.txt', '.md', '.jsonl', '.json', '.csv', '.py', '.html', '.eml'}
DOCUMENT_EXTENSIONS = {'.pdf', '.docx', '.pptx', '.xlsx'}
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}

# --- HELPER FUNCTIONS ---
def format_size(size_in_bytes):
    """Formats a size in bytes to a human-readable string (KB, MB, GB)."""
    if size_in_bytes is None:
        return "0 B"
    if size_in_bytes < 1024:
        return f"{size_in_bytes} B"
    elif size_in_bytes < 1024**2:
        return f"{size_in_bytes/1024:.2f} KB"
    elif size_in_bytes < 1024**3:
        return f"{size_in_bytes/1024**2:.2f} MB"
    else:
        return f"{size_in_bytes/1024**3:.2f} GB"

def estimate_embeddings(total_text_bytes):
    """
    Estimates the number of vectors and Pinecone storage size based on total text content.
    """
    # --- Assumptions for Estimation ---
    # Avg characters per token for English text.
    CHARS_PER_TOKEN = 4
    # Number of tokens per vector chunk. A common choice.
    TOKENS_PER_CHUNK = 256
    
    # Vector size for text-embedding-3-large (3072 dimensions * 4 bytes/dimension)
    VECTOR_BYTES = 3072 * 4
    # Metadata size (storing the original text chunk)
    METADATA_BYTES_PER_CHUNK = TOKENS_PER_CHUNK * CHARS_PER_TOKEN

    # --- Calculations ---
    if total_text_bytes == 0:
        return 0, 0, 0, 0
    
    estimated_tokens = total_text_bytes / CHARS_PER_TOKEN
    estimated_vectors = estimated_tokens / TOKENS_PER_CHUNK
    
    # Calculate storage
    total_vector_storage = estimated_vectors * VECTOR_BYTES
    total_metadata_storage = estimated_vectors * METADATA_BYTES_PER_CHUNK
    total_pinecone_storage = total_vector_storage + total_metadata_storage
    
    return estimated_vectors, total_vector_storage, total_metadata_storage, total_pinecone_storage

# --- MAIN ANALYSIS FUNCTION ---
def analyze_directory(directory_path):
    """
    Walks through a directory, categorizes files, and estimates text content.
    """
    file_stats = defaultdict(lambda: {'count': 0, 'size': 0})
    total_text_bytes = 0

    print(f"[*] Starting analysis of directory: {directory_path}\n")

    for root, _, files in os.walk(directory_path):
        for filename in files:
            try:
                file_path = os.path.join(root, filename)
                file_size = os.path.getsize(file_path)
                file_ext = os.path.splitext(filename)[1].lower()

                category = 'Other'
                if file_ext in TEXT_EXTENSIONS:
                    category = 'Text-based'
                    # For pure text files, file size is the text content size
                    total_text_bytes += file_size
                elif file_ext in DOCUMENT_EXTENSIONS:
                    category = 'Documents'
                    print(f"  - Processing document: {filename}...")
                    try:
                        # Extract actual text to get an accurate size
                        elements = partition(filename=file_path)
                        text_content = "\n".join([str(el) for el in elements])
                        # Use utf-8 encoding to get byte count
                        total_text_bytes += len(text_content.encode('utf-8'))
                    except Exception as e:
                        print(f"    [!] Warning: Could not extract text from {filename}. Reason: {e}")
                elif file_ext in IMAGE_EXTENSIONS:
                    category = 'Images'
                
                file_stats[category]['count'] += 1
                file_stats[category]['size'] += file_size

            except FileNotFoundError:
                print(f"    [!] Warning: Could not access file {filename}. Skipping.")
                continue

    return file_stats, total_text_bytes

# --- SCRIPT ENTRY POINT ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze a directory to estimate file types, sizes, and vector embedding storage."
    )
    parser.add_argument(
        "directory",
        type=str,
        help="The path to the directory you want to analyze (e.g., 'data/extracted')."
    )
    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print(f"[!] Error: The specified path '{args.directory}' is not a valid directory.")
    else:
        # 1. Analyze the directory
        stats, text_bytes = analyze_directory(args.directory)
        
        # 2. Get the embedding estimates
        vectors, vec_storage, meta_storage, total_storage = estimate_embeddings(text_bytes)

        # 3. Print the report
        print("\n" + "="*50)
        print("          Analysis and Estimation Report")
        print("="*50)

        print("\n--- File Type Analysis ---")
        total_files = 0
        total_size = 0
        for category, data in sorted(stats.items()):
            total_files += data['count']
            total_size += data['size']
            print(f"  - {category:<15}: {data['count']:>6} files | Total Size: {format_size(data['size'])}")
        print("  ---------------------------------------------")
        print(f"  - {'Total':<15}: {total_files:>6} files | Total Size: {format_size(total_size)}")
        
        print("\n--- Vector Embedding Estimation ---")
        print(f"  - Total Extracted Text Content: {format_size(text_bytes)}")
        print(f"  - Estimated Number of Vectors:  {int(vectors):,}")
        print(f"  - Estimated Vector Storage:     {format_size(vec_storage)}")
        print(f"  - Estimated Metadata Storage:   {format_size(meta_storage)}")
        print("  ---------------------------------------------")
        print(f"  - TOTAL ESTIMATED PINECONE STORAGE: {format_size(total_storage)}")

        print("\n" + "="*50)
        print("[*] Assumptions Made:")
        print("  - Model: text-embedding-3-large (3072 dimensions)")
        print("  - Avg. 4 characters per token.")
        print("  - 256 tokens per vector/chunk.")
        print("="*50)