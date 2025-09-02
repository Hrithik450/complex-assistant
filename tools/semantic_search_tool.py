# --- THIS IS THE FIX ---
# It checks if the app is running on Streamlit Cloud by looking for a specific environment variable.
# The sqlite3 patch will ONLY run when deployed to the cloud.
import os
import sys

# Streamlit Cloud sets the 'STREAMLIT_SERVER_PORT' environment variable.
# We can use its presence to detect the cloud environment.
if 'STREAMLIT_SERVER_PORT' in os.environ:
    print("Streamlit Cloud environment detected. Applying sqlite3 patch.")
    try:
        __import__('pysqlite3')
        sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
        print("Successfully patched sqlite3.")
    except ImportError:
        print("pysqlite3-binary not found, skipping patch. This may cause issues on Streamlit Cloud.")
# --- END OF FIX ---

#--- CHANGED: Import chroma_collection and df instead of index and df ---
from lib.load_data import chroma_collection, df
from langchain.tools import tool
from langchain_openai import OpenAIEmbeddings # <-- 1. IMPORT THE CORRECT EMBEDDING CLIENT
from lib.utils import EMBEDDING_MODEL_NAME, AGENT_MODEL
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

embedding_function = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)

# Multi Query: Different Perspectives
template = """You are an AI language model assistant. Your task is to generate 3 
different versions of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions separated by newlines. Original question: {question}"""
prompt_perspectives = ChatPromptTemplate.from_template(template)

generate_queries = (
    prompt_perspectives 
    | ChatOpenAI(model=AGENT_MODEL, temperature=0) 
    | StrOutputParser() 
    | (lambda x: x.split("\n"))
)

def get_unique_union(documents: list[str]):
    """ Unique union of retrieved docs """
    # Flatten list of lists, and convert each Document to string
    return list(set(documents))

@tool("semantic_search_tool", parse_docstring=True)
def semantic_search_tool(query: str) -> str:
    """
    This tool performs a semantic search over the indexed documents to retrieve the most relevant chunks based on a given query.
    Use this for conceptual, topic-based, or fuzzy questions.
    
    Args:
        query (str): The natural language query.

    Returns:
        str: Top 10 most similar email chunks with metadata (sender, subject, date).
    """
    print(f'semantic_search_tool is being called with {query}')
    
    # --- CHANGED: Query ChromaDB instead of FAISS ---
    # The manual embedding step is no longer needed, as Chroma handles it.
    # We now query Chroma to get the IDs of the most relevant documents.
    if chroma_collection is None:
        return "Error: ChromaDB connection is not available."
    
    # 1. Expand into multiple queries
    expanded_queries = generate_queries.invoke({"question": query})
    all_results = []

    # 2. For each expanded query, embed and fetch document
    for q in expanded_queries:
        query_embedding = embedding_function.embed_query(q)
        search_results = chroma_collection.query(query_embeddings=[query_embedding])

        # Chroma returns lists inside lists (one per query)
        docs = search_results["documents"][0]
        dists = search_results["distances"][0]

        for doc, dist in zip(docs, dists):
            if dist >= 0.70:
                all_results.append(doc)

    # 3. Deduplicate (get_unique_union effect)
    unique_results = get_unique_union(all_results)

    print(unique_results, 'unqiue results')

    if not unique_results:
        return "No relevant documents found."
    
    return "\n\n".join(unique_results[:10])

    # # # Extract the string IDs and convert them to integer indices for DataFrame lookup
    # # string_indices = search_results.get('ids', [[]])[0]
    # # if not string_indices:
    # #     return "No relevant information found."
    
    # # integer_indices = [int(i) for i in string_indices]

    # # --- UNCHANGED: Metadata lookup logic ---
    # # This part remains the same, using the indices to get data from the global 'df'.
    # # NOTE: Your original code used Pandas syntax (df.iloc), but your data is loaded
    # # with Polars. This version uses the correct Polars syntax (df.row).
    # results = []
    # for i in integer_indices:
    #     # Use df.row(i, named=True) for Polars to get a dictionary
    #     result_metadata = df.row(i, named=True)
    #     results.append(result_metadata)
        
    # if not results:
    #     return "No relevant information found."
        
    # # --- UNCHANGED: Formatting logic ---
    # # This formatting block is identical to your original.
    # formatted_results = "\n\n---\n\n".join([
    #     f"threadId: {res.get('threadId', 'N/A')}\n"
    #     f"From: {res.get('from', 'N/A')}\n"
    #     f"To: {res.get('to', 'N/A')}\n"
    #     f"Subject: {res.get('subject', 'N/A')}\n"
    #     f"Date: {res.get('date').strftime('%Y-%m-%d') if res.get('date') else 'N/A'}\n"
    #     f"Content Chunk: {res.get('original_text', 'N/A')}"
    #     for res in results
    # ])

    # print(f"Formatted results: {formatted_results}")

    # return formatted_results
