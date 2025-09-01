import sys
sys.path.append(r"d:\VSCode\re-assistant")
# --- CHANGED: Import chroma_collection and df instead of index and df ---
from lib.load_data import chroma_collection, df
from langchain.tools import tool

@tool("semantic_search_tool", parse_docstring=True)
def semantic_search_tool(query: str) -> str:
    """
    This tool performs a semantic search over the indexed documents to retrieve the most relevant chunks based on a given query.
    
    Args:
        query (str): The natural language query.

    Returns:
        str: Top 5 most similar email chunks with metadata (threadId, sender, subject, date).
    """
    print(f'semantic_search_tool is being called with {query}')

    # --- CHANGED: Query ChromaDB instead of FAISS ---
    # The manual embedding step is no longer needed, as Chroma handles it.
    # We now query Chroma to get the IDs of the most relevant documents.
    if chroma_collection is None:
        return "Error: ChromaDB connection is not available."
        
    search_results = chroma_collection.query(
        query_texts=[query],
        n_results=5
    )
    
    # Extract the string IDs and convert them to integer indices for DataFrame lookup
    string_indices = search_results.get('ids', [[]])[0]
    if not string_indices:
        return "No relevant information found."
    
    integer_indices = [int(i) for i in string_indices]

    # --- UNCHANGED: Metadata lookup logic ---
    # This part remains the same, using the indices to get data from the global 'df'.
    # NOTE: Your original code used Pandas syntax (df.iloc), but your data is loaded
    # with Polars. This version uses the correct Polars syntax (df.row).
    results = []
    for i in integer_indices:
        # Use df.row(i, named=True) for Polars to get a dictionary
        result_metadata = df.row(i, named=True)
        results.append(result_metadata)
        
    if not results:
        return "No relevant information found."
        
    # --- UNCHANGED: Formatting logic ---
    # This formatting block is identical to your original.
    formatted_results = "\n\n---\n\n".join([
        f"threadId: {res.get('threadId', 'N/A')}\n"
        f"From: {res.get('from', 'N/A')}\n"
        f"To: {res.get('to', 'N/A')}\n"
        f"Subject: {res.get('subject', 'N/A')}\n"
        f"Date: {res.get('date').strftime('%Y-%m-%d') if res.get('date') else 'N/A'}\n"
        f"Content Chunk: {res.get('original_text', 'N/A')}"
        for res in results
    ])

    return formatted_results