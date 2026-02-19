#--- CHANGED: Import chroma_collection and df instead of index and df ---
from lib.load_data import chroma_collection
from langchain.tools import tool
from langchain_openai import OpenAIEmbeddings # <-- 1. IMPORT THE CORRECT EMBEDDING CLIENT
from lib.utils import AGENT_MODEL, EMBEDDING_MODEL_NAME
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
import numpy as np
import os

# --- Heavy initializations ---
# 1. Embedding function with batching ---
embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME, api_key=os.getenv("OPENAI_API_KEY"))

# 2. BM25 (An superfast algo which focus more on imp key words to retrieve relavant docs) - will need documents loaded once
if chroma_collection is not None:
    # Load chunks from chroma
    all_chroma = chroma_collection.get(include=["documents", "metadatas"])
    documents  = all_chroma["documents"]
    metadatas  = all_chroma["metadatas"]
    # Pre-tokenize for BM25
    tokenized_docs = [doc.lower().split() for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)
    # Mapping: doc text -> index
    doc_to_index = {doc: i for i, doc in enumerate(documents)}
    index_to_doc = {i: doc for doc, i in doc_to_index.items()}
    doc_to_meta = {doc: (meta if meta is not None else {}) for doc, meta in zip(documents, metadatas or [{}]*len(documents))}
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
    metadata_results = []

    # 2. For each expanded query, embed and fetch document
    for q in expanded_queries:
        bm25_scores = bm25.get_scores(q.lower().split())
        bm25_scores = np.array(bm25_scores) / (np.max(bm25_scores)+1e-6)
        top_bm25_indices = np.argsort(bm25_scores)[::-1]
        top_bm25_docs = []
        for i in top_bm25_indices:
            idx = int(i)
            if bm25_scores[idx] > 0 and idx in index_to_doc:
                top_bm25_docs.append((index_to_doc[idx], bm25_scores[idx]))

        # Create embeddings for query and filter candidate docs
        query_embedding = embedding_model.embed_query(q)
        search_results = chroma_collection.query(query_embeddings=[query_embedding])

        # Chroma returns lists inside lists (one per query)
        sem_docs = search_results["documents"][0]
        sem_scores = search_results["distances"][0]
        sem_metadata = search_results["metadatas"][0]

        for i, doc in enumerate(sem_docs):
            bm25_index = doc_to_index.get(doc, None)
            bm25_score = bm25_scores[bm25_index] if bm25_index is not None else 0
            dense_score = sem_scores[i]
            combined_score = 0.5 * bm25_score + 0.5 * dense_score

            # check if email_id is present inside metadata
            meta_item = sem_metadata[i] if i < len(sem_metadata) else {}
            email_id = (meta_item.get("email_id") if isinstance(meta_item, dict) else None)

            if doc.startswith("Metadata:"):
                metadata_results.append((doc, email_id, combined_score))
            else:
                all_results.append((doc, email_id, combined_score))

        for doc, bm25_score in top_bm25_docs:
            if doc.startswith("Metadata:"):
                metadata_results.append((doc, doc_to_meta.get(doc, {}).get("email_id") if doc in doc_to_meta else None, bm25_score))
                continue

            if doc not in sem_docs:
                email_id = doc_to_meta[doc].get("email_id") if doc in doc_to_meta else None
                all_results.append((doc, email_id, bm25_score))

    # Deduplicate (get_unique_union effect)
    unique_results = {}
    for doc, email_id, score in all_results:
        if doc not in unique_results or score > unique_results[doc]["score"]:
            unique_results[doc] = {"email_id": email_id, "score": score}

    top_chunks = sorted(unique_results.items(), key=lambda x:x[1]["score"], reverse=True) # top 25

    # Re-ranking with Cross-Encoder
    pairs = [[query, doc] for doc, _ in top_chunks]
    rerank_scores = cross_encoder.predict(pairs)
    ranked = sorted(zip(rerank_scores, top_chunks), key=lambda x: x[0], reverse=True)

    # Deduplicate metadata docs separately
    unique_metadata = {}
    for doc, email_id, score in metadata_results:
        if doc not in unique_metadata or score > unique_metadata[doc]["score"]:
            unique_metadata[doc] = {"email_id": email_id, "score": score}

    top_metadata = sorted(unique_metadata.items(), key=lambda x: x[1]["score"], reverse=True)

    # Combine final results: top 10 main docs, then all metadata as low-priority
    results_for_llm = []
    for _, (doc, meta) in ranked:
        email_id = meta.get("email_id") if isinstance(meta, dict) else None
        if email_id:
            results_for_llm.append(f"[id: {email_id}]\n{doc}")
        else:
            results_for_llm.append(doc)

    threshold = 50
    for doc, meta in top_metadata[:25]:
        if meta["score"] < threshold:
            continue
        email_id = meta.get("email_id") if isinstance(meta, dict) else None
        if email_id:
            results_for_llm.append(f"[id: {email_id}]\n{doc}")
        else:
            results_for_llm.append(doc)

    # Return results
    return "\n\n---\n\n".join(results_for_llm) if results_for_llm else "No relevant documents found."