import os
import pickle
import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import logging
import tiktoken
import streamlit as st

# --- CONFIGURATION ---
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Use Streamlit's secrets management or .env for local development
try:
    # For Streamlit Community Cloud
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except FileNotFoundError:
    # For local development
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- CRITICAL FILE PATHS ---
FINETUNED_MODEL_PATH = os.path.join(SCRIPT_DIR, "finetuned_bge_real_estate_model")
FAISS_INDEX_PATH = os.path.join(SCRIPT_DIR, "real_estate_finetuned_local_faiss.bin")
METADATA_PATH = os.path.join(SCRIPT_DIR, "real_estate_finetuned_local_metadata.pkl")
CONTEXT_TOKEN_LIMIT = 100000

# --- CORE APPLICATION LOGIC (Moved from query_local.py) ---

# Use Streamlit's caching to load models and data only once
@st.cache_resource
def initialize_services():
    """Load all necessary clients, models, and local data files."""
    print("[*] Loading services and data for the first time...")
    
    client = OpenAI(api_key=OPENAI_API_KEY)
    if not client.api_key: raise ValueError("FATAL: OPENAI_API_KEY not found.")
    
    if not os.path.isdir(FINETUNED_MODEL_PATH):
        st.error(f"FATAL: Fine-tuned model not found at '{FINETUNED_MODEL_PATH}'. Please run the training pipeline.")
        return None, None, None
    
    embedding_model = SentenceTransformer(FINETUNED_MODEL_PATH, device='cpu')
    
    try:
        faiss_index = faiss.read_index(FAISS_INDEX_PATH)
        with open(METADATA_PATH, "rb") as f:
            metadata_store = pickle.load(f)
    except FileNotFoundError:
        st.error(f"FATAL: Database files not found. Please run 'process_data.py' first.")
        return None, None, None

    print("[+] Services and data loaded successfully.")
    return client, embedding_model, faiss_index, metadata_store

def get_answer_from_llm(client, query, context):
    """Uses gpt-4o to analyze context and provide an answer."""
    system_prompt = """
    You are a world-class AI data analyst... (Full prompt from the last correct version)
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

# --- STREAMLIT USER INTERFACE ---

st.set_page_config(page_title="AI Business Analyst", layout="wide")

st.title("🤖 AI Business Data Assistant")
st.markdown("Ask any question about your business documents, including emails, WhatsApp chats, PDFs, and spreadsheets.")

# Load all the necessary components
client, embedding_model, faiss_index, metadata_store = initialize_services()

if client:
    # Initialize session state to hold the conversation
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Get user input from the chat box
    if prompt := st.chat_input("e.g., How many qualified leads came from Social Media last week?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # --- Start the RAG process ---
        with st.chat_message("assistant"):
            with st.spinner("Performing a broad semantic search..."):
                # 1. Embed the user's query
                query_embedding = embedding_model.encode([prompt])
                query_embedding_np = np.array(query_embedding).astype('float32')
                
                # 2. Perform the FAISS search
                top_k = 150 # Retrieve a large number of candidates
                distances, indices = faiss_index.search(query_embedding_np, top_k)
                retrieved_indices = indices[0]
                
                if len(retrieved_indices) == 0 or retrieved_indices[0] == -1:
                    st.warning("Could not find any relevant documents in the initial search.")
                    st.stop()

                retrieved_metadatas = [metadata_store[i] for i in retrieved_indices]

            # Display the retrieved context for transparency (in an expander)
            with st.expander("🔍 View Retrieved Context (Top 5 Results)"):
                for i, doc in enumerate(retrieved_metadatas[:5]):
                    st.markdown(f"**Result {i+1}**")
                    st.info(f"**Source:** `{doc.get('source', 'N/A')}`")
                    st.code(f"{doc.get('original_text', '').strip()[:600]}...")

            with st.spinner("Building context and passing to gpt-4o for final analysis..."):
                # 3. Build and truncate context for the final LLM
                tokenizer = tiktoken.get_encoding("cl100k_base")
                context_for_llm = ""
                total_tokens = 0
                included_chunks = 0
                
                base_prompt_tokens = len(tokenizer.encode(f"User's Question: {prompt}")) + 500
                token_budget = CONTEXT_TOKEN_LIMIT - base_prompt_tokens

                for doc in retrieved_metadatas:
                    chunk_text = f"Source: {doc.get('source', 'N/A')}\nContent: {doc.get('original_text', '')}\n---\n"
                    chunk_tokens = len(tokenizer.encode(chunk_text))
                    if total_tokens + chunk_tokens > token_budget:
                        break
                    context_for_llm += chunk_text
                    total_tokens += chunk_tokens
                    included_chunks += 1
                
                st.info(f"Passing {included_chunks} of {len(retrieved_metadatas)} chunks ({total_tokens} tokens) to the reasoning engine...")

                # 4. Generate the final answer
                final_answer = get_answer_from_llm(client, prompt, context_for_llm)
            
            # Display the final answer
            st.markdown(final_answer)
            # Add the assistant's response to the session state
            st.session_state.messages.append({"role": "assistant", "content": final_answer})
else:
    st.error("Application failed to initialize. Please check the console for errors.")