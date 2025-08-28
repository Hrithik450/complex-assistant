import numpy as np
from lib.faiss import index
from lib.dataframe import df
from langchain.tools import tool
from lib.embeddings import embeddings

@tool("semantic_search_tool", parse_docstring=True)
def semantic_search_tool(query: str) -> str:
    """
    This tool performs a semantic search over the indexed documents to retrieve the most relevant chunks based on a given query.
    
    Args:
        query (str): The natural language query.

    Returns:
        str: Top 5 most similar email chunks with metadata (threadId, sender, subject, date).
    """
    print('semantic_search_tool is being called')
    query_embedding = np.array(embeddings.embed_query(query)).astype('float32').reshape(1, -1)
    
    # Search the FAISS index for the top 5 most similar chunks
    _, indices = index.search(query_embedding, k=5)
    
    results = []
    for i in indices[0]:
        result_metadata = df.iloc[i].to_dict()
        results.append(result_metadata)
        
    if not results:
        return "No relevant information found."
        
    # Format the results for the agent
    formatted_results = "\n\n---\n\n".join([
        f"threadId: {res.get('threadId', 'N/A')}\n"
        f"From: {res.get('from', 'N/A')}\n"
        f"To: {res.get('to', 'N/A')}\n"
        f"Subject: {res.get('subject', 'N/A')}\n"
        f"Date: {res.get('date', 'N/A').strftime('%Y-%m-%d') if res.get('date') else 'N/A'}\n"
        f"Content Chunk: {res.get('original_text', 'N/A')}"
        for res in results
    ])

    return formatted_results