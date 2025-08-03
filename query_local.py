import os
import pickle
import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import logging
import re
import tiktoken
import argparse

# --- CONFIGURATION ---
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FINETUNED_MODEL_PATH = os.path.join(SCRIPT_DIR, "finetuned_bge_real_estate_model")
FAISS_INDEX_PATH = os.path.join(SCRIPT_DIR, "real_estate_finetuned_local_faiss.bin")
METADATA_PATH = os.path.join(SCRIPT_DIR, "real_estate_finetuned_local_metadata.pkl")
CONTEXT_TOKEN_LIMIT = 100000 # Leave a large buffer for gpt-4o's 128k window

# --- INITIALIZATION ---
def initialize_services():
    """Load all necessary clients, models, and local data files."""
    print("[*] Loading services and data...")
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if not client.api_key: raise ValueError("FATAL: OPENAI_API_KEY not found.")
    
    if not os.path.isdir(FINETUNED_MODEL_PATH):
        raise FileNotFoundError(f"FATAL: Fine-tuned model not found at '{FINETUNED_MODEL_PATH}'.")
    
    print(f"[*] Loading fine-tuned model for queries...")
    embedding_model = SentenceTransformer(FINETUNED_MODEL_PATH, device='cpu')
    
    try:
        faiss_index = faiss.read_index(FAISS_INDEX_PATH)
        with open(METADATA_PATH, "rb") as f:
            metadata_store = pickle.load(f)
    except FileNotFoundError:
        print(f"\n[!] FATAL: Database files not found. Please run 'process_data.py' with the fine-tuned model first.")
        return None, None, None

    print("[+] Services and data loaded.")
    return client, embedding_model, faiss_index, metadata_store

# --- ANSWER GENERATION (THE DEFINITIVE FIX) ---
def get_answer_from_llm(client, query, context):
    """
    Uses a powerful LLM (gpt-4o) to perform analysis on a broad context and provide a specific, evidence-backed answer.
    """
    # --- NEW: The "Data Analyst" System Prompt ---
    system_prompt = """
    You are a world-class AI data analyst. You will be given a user's specific, often analytical, question and a large, unfiltered set of retrieved text chunks from a knowledge base. Your task is to act as an expert detective to provide a precise, evidence-backed answer.

    **MULTI-STEP ANALYSIS PROCESS:**
    1.  **Understand the User's Goal:** Carefully read the user's question to identify the specific analytical task they want. This could be a COUNT, a LIST, a SUMMARY, or a comparison. Identify all constraints (e.g., "qualified leads", "Social Media", "this weekend").
    2.  **Scour the Context:** Read through ALL the provided context chunks. Your primary job is to find any and all snippets of text that are relevant to the user's goal.
    3.  **Extract and Analyze:** Internally, extract the key pieces of information from the relevant snippets. If the user asks to COUNT items, you must find each individual item in the text and count them.
    4.  **Synthesize the Final Answer:** Formulate a direct answer to the user's question. If you performed a calculation (like a count), state the final number clearly.
    5.  **Provide Evidence:** After your answer, you MUST provide a "Supporting Evidence" section. In this section, list the specific, direct quotes from the context that you used to arrive at your conclusion, and cite the source file for each quote.

    **BE HONEST:** If the provided context does not contain enough information to perform the analysis (e.g., no mention of lead sources), you MUST state: "Based on the retrieved documents, I could not find enough information to answer your question." Do not make up data.
    """
    
    user_prompt = f"Retrieved Context:\n---\n{context}\n---\n\nUser's Original Question:\n{query}\n\nFinal Answer:"
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.0
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"(Could not generate answer from gpt-4o. Error: {e})"

# --- MAIN EXECUTION ---
def main():
    parser = argparse.ArgumentParser(description="Ask a natural language question to your local real estate data.")
    parser.add_argument("query", type=str, help="Your full question, enclosed in quotes.")
    parser.add_argument("--top_k", type=int, default=200, help="Number of initial candidate documents to retrieve for the LLM to analyze.")
    args = parser.parse_args()
    
    client, embedding_model, faiss_index, metadata_store = initialize_services()
    if not client: return

    try:
        print(f"\n[1] Performing a broad semantic search for: \"{args.query}\"")
        
        query_embedding = embedding_model.encode([args.query])
        query_embedding_np = np.array(query_embedding).astype('float32')
        
        distances, indices = faiss_index.search(query_embedding_np, args.top_k)
        retrieved_indices = indices[0]
        
        if len(retrieved_indices) == 0 or retrieved_indices[0] == -1:
            print("\n[!] No relevant documents found in the initial search.")
            return

        retrieved_metadatas = [metadata_store[i] for i in retrieved_indices]
        
        # Debugging Section
        print("\n" + "="*80)
        print("                  INITIAL RETRIEVED CONTEXT (Top 5 for Verification)")
        print("="*80)
        for i, doc in enumerate(retrieved_metadatas[:5]):
            print(f"\n--- Result {i+1} ---")
            print(f"  - Source:  {doc.get('source', 'N/A')}")
            text_snippet = doc.get('original_text', '').strip().replace('\n', ' ')[:600]
            print(f"  - Text:    \"{text_snippet}...\"")
        print("="*80)

        # Build and truncate context for the final LLM
        tokenizer = tiktoken.get_encoding("cl100k_base")
        context_for_llm, total_tokens, included_chunks = "", 0, 0
        
        base_prompt_tokens = len(tokenizer.encode(f"User's Question: {args.query}")) + 500 # Larger buffer for the new prompt
        token_budget = CONTEXT_TOKEN_LIMIT - base_prompt_tokens

        for doc in retrieved_metadatas:
            chunk_text = f"Source: {doc.get('source', 'N/A')}\nContent: {doc.get('original_text', '')}\n---\n"
            chunk_tokens = len(tokenizer.encode(chunk_text))
            if total_tokens + chunk_tokens > token_budget: break
            context_for_llm += chunk_text
            total_tokens += chunk_tokens
            included_chunks += 1
        
        print(f"\n[2] Passing {included_chunks} of {len(retrieved_metadatas)} chunks ({total_tokens} tokens) to gpt-4o for final analysis...")
        final_answer = get_answer_from_llm(client, args.query, context_for_llm)
        
        print("\n" + "="*80 + "\n                        FINAL ANSWER\n" + "="*80 + "\n")
        print(f"Answer:\n{final_answer}\n")

    except Exception as e:
        print(f"\n[!] An unexpected critical error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()