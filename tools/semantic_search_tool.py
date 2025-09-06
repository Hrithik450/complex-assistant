#--- CHANGED: Import chroma_collection and df instead of index and df ---
from lib.load_data import chroma_collection
from langchain.tools import tool
from langchain_openai import OpenAIEmbeddings # <-- 1. IMPORT THE CORRECT EMBEDDING CLIENT
from lib.utils import EMBEDDING_MODEL_NAME, AGENT_MODEL
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
import numpy as np

# --- Heavy initializations ---
# 1. Embedding function
embedding_function = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)

# 2. BM25 (An superfast algo which focus more on imp key words to retrieve relavant docs) - will need documents loaded once
if chroma_collection is not None:
    # Load chunks from chroma
    documents = [doc for doc in chroma_collection.get()['documents']]
    # Pre-tokenize for BM25
    tokenized_docs = [doc.lower().split() for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)
    # Mapping: doc text -> index
    doc_to_index = {doc: i for i, doc in enumerate(documents)}
else:
    documents = []
    tokenized_docs = []
    bm25 = None

# 3. Cross-encoder (to compare the lists & re-rank based on the semantic meaning)
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# 4. Query expansion pipeline
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

@tool("semantic_search_tool", parse_docstring=True)
def semantic_search_tool(query: str) -> str:
    """
    This tool performs a semantic search over the documents to retrieve 
    the most relevant chunks based on user asked query.
    
    Args:
        query (str): The natural language query.

    Returns:
        str: Top 10 most relavent documents with data.
    """
    print(f'semantic_search_tool is being called with {query}')
    
    # --- CHANGED: Query ChromaDB---
    # We now query Chroma to get the most relevant documents.
    if chroma_collection is None:
        return "Error: ChromaDB connection is not available."

    # 2. Expand into multiple queries
    expanded_queries = generate_queries.invoke({"question": query})
    all_results = []

    # 2. For each expanded query, embed and fetch document
    for q in expanded_queries:
        bm25_scores = bm25.get_scores(q.lower().split())
        bm25_scores = np.array(bm25_scores) / (np.max(bm25_scores)+1e-6)
        
        # Create embeddings for query and filter candidate docs
        query_embedding = embedding_function.embed_query(q)
        search_results = chroma_collection.query(query_embeddings=[query_embedding])

        # Chroma returns lists inside lists (one per query)
        docs = search_results["documents"][0]
        dists = search_results["distances"][0]

        for i, doc in enumerate(docs):
            bm25_index = doc_to_index.get(doc, None)
            bm25_score = bm25_scores[bm25_index] if bm25_index is not None else 0
            dense_score = dists[i]
            combined_score = 0.5 * bm25_score + 0.5 * dense_score
            all_results.append((doc, combined_score))

    # Deduplicate (get_unique_union effect)
    unique_results = {}
    for doc, score in all_results:
        if doc not in all_results or score > unique_results[doc]:
            unique_results[doc] = score

    top_chunks = sorted(unique_results.items(), key=lambda x:x[1], reverse=True)[:20] # top 20

    # Re-ranking with Cross-Encoder
    pairs = [[query, doc] for doc, _ in top_chunks]
    rerank_scores = cross_encoder.predict(pairs)
    final_ranked = [doc for _, doc in sorted(zip(rerank_scores, [doc for doc, _ in top_chunks]), reverse=True)]

    # Return results
    return "\n\n".join(final_ranked[:10]) if final_ranked else "No relevant documents found."
