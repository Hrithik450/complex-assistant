import streamlit as st
import sys
import os
import re
import gdown  # For downloading from Google Drive
import zipfile # For unzipping the model folder
from contextlib import contextmanager, redirect_stdout
from io import StringIO

# --- Add the current directory to the Python path ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)

# --- Import the agent and its dependencies ---
# This assumes your agent.py file is in the same directory
from agent import ReActAgent, KnowledgeBaseTool
from openai import OpenAI
from dotenv import load_dotenv


# --- NEW: Function to download necessary files from Google Drive ---
def download_files_from_gdrive():
    """
    Downloads the model, FAISS index, and metadata from Google Drive if they don't already exist.
    """
    # Define local paths within the Streamlit container
    faiss_path = os.path.join(SCRIPT_DIR, "real_estate_finetuned_local_faiss.bin")
    metadata_path = os.path.join(SCRIPT_DIR, "real_estate_finetuned_local_metadata.pkl")
    model_path = os.path.join(SCRIPT_DIR, "finetuned_bge_real_estate_model")
    model_zip_path = os.path.join(SCRIPT_DIR, "finetuned_model.zip")

    # Check if files already exist. If so, do nothing.
    if os.path.exists(faiss_path) and os.path.exists(metadata_path) and os.path.exists(model_path):
        st.info("Model and data files already exist. Skipping download.")
        return

    st.info("Downloading required model and data files from Google Drive. This may take a moment...")

    try:
        # Get URLs from Streamlit secrets
        faiss_url = st.secrets["GDRIVE_FAISS_URL"]
        metadata_url = st.secrets["GDRIVE_METADATA_URL"]
        model_zip_url = st.secrets["GDRIVE_MODEL_ZIP_URL"]

        # Download files using gdown
        with st.spinner("Downloading FAISS index..."):
            gdown.download(url=faiss_url, output=faiss_path, quiet=False)
        st.success("FAISS index downloaded.")

        with st.spinner("Downloading metadata..."):
            gdown.download(url=metadata_url, output=metadata_path, quiet=False)
        st.success("Metadata downloaded.")

        with st.spinner("Downloading fine-tuned model..."):
            gdown.download(url=model_zip_url, output=model_zip_path, quiet=False)
        st.success("Model zip file downloaded.")

        # Unzip the model folder
        with st.spinner("Unzipping model..."):
            with zipfile.ZipFile(model_zip_path, 'r') as zip_ref:
                zip_ref.extractall(SCRIPT_DIR)
        st.success("Model unzipped successfully.")

        # Clean up the downloaded zip file
        os.remove(model_zip_path)

        st.success("All required files are ready!")

    except Exception as e:
        st.error(f"Failed to download files from Google Drive. Error: {e}")
        st.stop()


# --- Page Configuration ---
st.set_page_config(
    page_title="AI Business Analyst",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Business Analyst for Real Estate")
st.caption("Ask complex questions about your business documents. The agent will reason and use a knowledge base to find the answer.")


# --- Main App Logic ---

# Run the download function at the start of the app
download_files_from_gdrive()


# --- Initialization and Caching (Now runs AFTER files are downloaded) ---
@st.cache_resource
def initialize_agent():
    """Loads the environment variables and initializes the agent and its tools."""
    load_dotenv()
    # Use st.secrets for deployment, fallback to os.getenv for local dev
    openai_api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        st.error("FATAL: OPENAI_API_KEY not found. Please add it to your Streamlit secrets.")
        st.stop()
    
    try:
        # The agent.py file automatically uses the correct file paths as they are in the same directory
        openai_client = OpenAI(api_key=openai_api_key)
        kb_tool = KnowledgeBaseTool(client=openai_client)
        agent = ReActAgent(
            client=openai_client, 
            tool=kb_tool, 
            metadata_store=kb_tool.metadata_store
        )
        return agent
    except FileNotFoundError as e:
        st.error(f"A required file was not found after download: {e}. Please check the file paths and Google Drive links.")
        st.stop()
    except Exception as e:
        st.error(f"A critical error occurred during initialization: {e}")
        import traceback
        st.text(traceback.format_exc())
        st.stop()

# Initialize the agent
agent = initialize_agent()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your business..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_container = st.container()
        thought_box = response_container.empty()
        
        thinking_log = ""
        
        def append_to_thought_box(text):
            global thinking_log
            thinking_log += text
            thought_box.markdown(f"**Thinking Process...**\n```log\n{thinking_log}\n```")

        history = [("system", agent.system_prompt)]
        history.append(("user", prompt))
        final_answer = "The agent did not produce a final answer after its steps."

        for i in range(5):
            log_header = f"\n======================== STEP {i+1} ========================\n"
            append_to_thought_box(log_header)

            prompt_messages = [{"role": role, "content": content} for role, content in history]
            
            try:
                response = agent.client.chat.completions.create(
                    model="gpt-4o", messages=prompt_messages, temperature=0.0
                )
                action_text = response.choices[0].message.content
                
                append_to_thought_box(action_text)
                history.append(("assistant", action_text))

                if "Final Answer(" in action_text:
                    match = re.search(r"Final Answer\(answer=(['\"])(.*)\1\)", action_text, re.DOTALL)
                    if match:
                        final_answer = match.group(2)
                    else:
                        final_answer = "Could not parse the final answer from the agent's response."
                    break

                elif "knowledge_base_search(" in action_text:
                    match = re.search(r"knowledge_base_search\(query=(['\"])(.*)\1\)", action_text, re.DOTALL)
                    if match:
                        query = match.group(2)
                        append_to_thought_box(f"\n> Searching knowledge base for: '{query}'\n")
                        observation = agent.tool.search(query=query)
                        history.append(("user", f"Observation: {observation}"))
                        append_to_thought_box(f"\n> Observation received from tool.\n")
                    else:
                        history.append(("user", "Observation: Could not parse the tool query."))
                else:
                    final_answer = "Agent stopped because it generated a response without a valid Action."
                    break

            except Exception as e:
                st.error(f"An error occurred during agent execution: {e}")
                final_answer = f"An error occurred: {e}"
                break
        
        thought_box.empty()
        response_container.markdown(final_answer)
        st.session_state.messages.append({"role": "assistant", "content": final_answer})