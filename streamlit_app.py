import streamlit as st
import sys
import os
import uuid
import gdown
import zipfile
from dotenv import load_dotenv

# --- Add the current directory to the Python path ---
# This ensures that the app can find the agent_pro module
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)

# --- Import the final, robust agent ---
from agent_pro import ManagerAgent

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Business Analyst (Advanced ReAct)",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Business Analyst")
st.caption("An intelligent agent that learns from your feedback.")

# --- File Downloader & Provisioning ---
def provision_files():
    """
    Checks for necessary model and data files and downloads them from Google Drive if missing.
    This is the single source of truth for setting up the app's environment on Streamlit Cloud.
    """
    faiss_path = os.path.join(SCRIPT_DIR, "real_estate_finetuned_local_faiss.bin")
    metadata_path = os.path.join(SCRIPT_DIR, "real_estate_finetuned_local_metadata.pkl")
    model_path = os.path.join(SCRIPT_DIR, "finetuned_bge_real_estate_model")
    model_zip_path = os.path.join(SCRIPT_DIR, "finetuned_model.zip")

    # If everything already exists, we don't need to do anything.
    if os.path.exists(faiss_path) and os.path.exists(metadata_path) and os.path.isdir(model_path):
        return

    st.info("Downloading required model and data files. This may take a moment on first startup...")
    try:
        # Get File IDs from Streamlit secrets
        faiss_id = st.secrets["GDRIVE_FAISS_FILE_ID"]
        metadata_id = st.secrets["GDRIVE_METADATA_FILE_ID"]
        model_zip_id = st.secrets["GDRIVE_MODEL_ZIP_FILE_ID"]

        # Download files using gdown, only if they don't already exist
        if not os.path.exists(faiss_path):
            with st.spinner("Downloading FAISS index (vector store)..."):
                gdown.download(id=faiss_id, output=faiss_path, quiet=True)
            st.success("FAISS index downloaded.")

        if not os.path.exists(metadata_path):
            with st.spinner("Downloading metadata store..."):
                gdown.download(id=metadata_id, output=metadata_path, quiet=True)
            st.success("Metadata downloaded.")

        if not os.path.isdir(model_path):
            with st.spinner("Downloading fine-tuned model..."):
                gdown.download(id=model_zip_id, output=model_zip_path, quiet=True)
            st.success("Model zip file downloaded.")

            with st.spinner("Unzipping model..."):
                with zipfile.ZipFile(model_zip_path, 'r') as zip_ref:
                    zip_ref.extractall(SCRIPT_DIR)
            st.success("Model unzipped successfully.")
            
            # Clean up the downloaded zip file
            os.remove(model_zip_path)
        
        st.success("All required files are ready!")
    except Exception as e:
        st.error(f"Failed to download files from Google Drive. Please ensure GDrive File IDs are correct in secrets. Error: {e}")
        st.stop()

# --- Initialization and Caching ---
@st.cache_resource
def initialize_system():
    """
    Initializes the ManagerAgent. This runs only once per session.
    """
    load_dotenv()
    
    # This function will block until all files are present before proceeding.
    provision_files()
    
    # Check for necessary API keys
    required_secrets = ["OPENAI_API_KEY", "TAVILY_API_KEY"]
    missing_secrets = [secret for secret in required_secrets if secret not in st.secrets and not os.getenv(secret)]
    if missing_secrets:
        st.error(f"FATAL: Missing the following secrets: {', '.join(missing_secrets)}. Please add them to your Streamlit secrets.")
        st.stop()
        
    try:
        with st.spinner("Initializing AI Analyst and connecting to knowledge base..."):
            agent = ManagerAgent()
        return agent
    except Exception as e:
        st.error(f"A critical error occurred during agent initialization: {e}")
        import traceback
        st.code(traceback.format_exc())
        st.stop()

# Initialize the agent system
manager_agent = initialize_system()
st.success("AI Analyst is ready.", icon="✅")

# --- Session State Management and UI ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Feedback and Correction Logic ---
def toggle_correction_box(record_id):
    """Callback to show the correction text area."""
    st.session_state[f"show_correction_{record_id}"] = True

def handle_correction(record_id):
    """Callback to handle the submission of a correction."""
    corrected_text = st.session_state[f"correction_text_{record_id}"]
    if corrected_text and not corrected_text.isspace():
        with st.spinner("Learning from your feedback... This may take a moment."):
            try:
                # The agent's data_manager has the upsert logic
                manager_agent.data_manager.upsert_correction(corrected_text)
                st.toast("Thank you! I've updated my knowledge base.", icon="🧠")
                # Update UI state
                st.session_state[f"feedback_given_{record_id}"] = True
                st.session_state[f"show_correction_{record_id}"] = False
                st.rerun()
            except Exception as e:
                st.error(f"Failed to save correction: {e}")
    else:
        st.warning("Please enter a corrected response.")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "record_id" in message:
            record_id = message["record_id"]
            if not st.session_state.get(f"feedback_given_{record_id}", False):
                cols = st.columns(12)
                cols[0].button("👍", key=f"up_{record_id}", on_click=manager_agent.data_manager.update_feedback, args=(record_id, 1))
                cols[1].button("👎", key=f"down_{record_id}", on_click=toggle_correction_box, args=(record_id,))
            
            if st.session_state.get(f"show_correction_{record_id}", False):
                st.text_area("Provide the correct answer here:", key=f"correction_text_{record_id}")
                st.button("Submit Correction", key=f"submit_{record_id}", on_click=handle_correction, args=(record_id,))

# Main chat input logic
if prompt := st.chat_input("Ask a complex question or provide a correction..."):
    if prompt and not prompt.isspace():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("The AI team is thinking..."):
                response, record_id = manager_agent.run(
                    user_query=prompt, 
                    session_id=st.session_state.session_id
                )
                st.markdown(response)
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response, 
                    "record_id": record_id
                })
                st.rerun()
