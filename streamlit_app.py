import streamlit as st
import sys
import os
import uuid
from dotenv import load_dotenv

# --- Add the current directory to the Python path ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)

# --- Import the agent and its dependencies ---
# This assumes your agent_pro.py file is in the same directory
from agent_pro import ManagerAgent, DataManager

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Business Analyst Pro",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 AI Business Analyst Pro")
st.caption("An intelligent multi-agent system with persistent memory and feedback.")

# --- Initialization and Caching ---
@st.cache_resource
def initialize_system():
    """Initializes the ManagerAgent. This runs only once."""
    load_dotenv()
    # Check for the API key
    if not os.getenv("OPENAI_API_KEY"):
        st.error("FATAL: OPENAI_API_KEY not found. Please create a .env file or set it as a secret.")
        st.stop()

    try:
        agent = ManagerAgent()
        return agent
    except Exception as e:
        st.error(f"A critical error occurred during initialization: {e}")
        import traceback
        st.code(traceback.format_exc())
        st.stop()

# Initialize the agent system
manager_agent = initialize_system()

# --- Session State Management ---
# Ensure a unique session ID for each user's browser tab
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Feedback Handling ---
def handle_feedback(record_id, feedback_value):
    """Callback function to update feedback in the database."""
    # The DataManager is part of the agent, we can access it
    manager_agent.data_manager.update_feedback(record_id, feedback_value)
    st.toast(f"Thank you for your feedback!", icon="✅")
    # Disable buttons after feedback is given
    st.session_state[f"feedback_given_{record_id}"] = True

# --- Display Chat History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Add feedback buttons to assistant messages that haven't received feedback yet
        if message["role"] == "assistant" and "record_id" in message:
            record_id = message["record_id"]
            if not st.session_state.get(f"feedback_given_{record_id}", False):
                cols = st.columns(10)
                with cols[0]:
                    st.button("👍", key=f"up_{record_id}", on_click=handle_feedback, args=(record_id, 1))
                with cols[1]:
                    st.button("👎", key=f"down_{record_id}", on_click=handle_feedback, args=(record_id, -1))

# --- Main App Logic ---
if prompt := st.chat_input("Ask a complex question..."):
    # Add user message to UI
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get agent's response
    with st.chat_message("assistant"):
        with st.spinner("The AI team is thinking..."):
            # Run the agent with the query and the unique session ID
            response, record_id = manager_agent.run(
                user_query=prompt, 
                session_id=st.session_state.session_id
            )
            st.markdown(response)
            
            # Add the response and its record_id to the UI history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response, 
                "record_id": record_id
            })
            
            # Add feedback buttons for the new message
            cols = st.columns(10)
            with cols[0]:
                st.button("👍", key=f"up_{record_id}", on_click=handle_feedback, args=(record_id, 1))
            with cols[1]:
                st.button("👎", key=f"down_{record_id}", on_click=handle_feedback, args=(record_id, -1))