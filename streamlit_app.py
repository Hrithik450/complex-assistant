# --- THIS IS THE FIX ---
# It checks if the app is running on Streamlit Cloud by looking for a specific environment variable.
# The sqlite3 patch will ONLY run when deployed to the cloud.
import sys

# --- THIS IS THE NEW, ROBUST FIX ---
# This ensures the patch runs before chromadb is ever touched.
IS_STREAMLIT_ENVIRONMENT = "streamlit" in sys.modules
if IS_STREAMLIT_ENVIRONMENT:
    print("Streamlit environment detected. Applying sqlite3 patch in load_data.py.")
    try:
        __import__('pysqlite3')
        sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
        print("Successfully patched sqlite3.")
    except ImportError:
        print("pysqlite3-binary not found, skipping patch.")

import pytz
import json
import redis
from typing import List
import streamlit as st
from datetime import datetime

st.set_page_config(page_title="AI Email Assistant", page_icon="📧")

# Import the tools and agent components from your existing files
from langgraph.prebuilt import ToolNode
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, MessagesState, START, END
from lib.utils import HELPER_MODEL, BASE_MODEL, SYSTEM_PROMPT,MEMORY_LAYER_PROMPT
from lib.helpers.general import parse_json

from tools.semantic_search_tool import semantic_search_tool
from tools.metadata_filtering_tool import email_filtering_tool

# This will trigger the data loading and Chroma connection via st.cache_resource
from lib.load_data import df, chroma_collection

# Import your database logic
from lib.db.db_service import ThreadService
from lib.db.db_conn import conn

# -------------------- CONFIG --------------------
IST = pytz.timezone("Asia/Kolkata")
today_date = datetime.now(IST).strftime("%B %d, %Y")
USER_ID = "63f05e7a-35ac-4deb-9f38-e2864cdf3a1d" # Hardcoded for this example

tools = [semantic_search_tool, email_filtering_tool]
tool_node = ToolNode(tools)

@st.cache_resource
def get_base_model():
    base_model = init_chat_model(model=BASE_MODEL, temperature=0.4)
    return base_model.bind_tools(tools)

@st.cache_resource
def get_helper_model():
    helper_model = init_chat_model(model=HELPER_MODEL, temperature=0)
    return helper_model.bind_tools(tools)

@st.cache_resource
def get_memory():
    """Initializes and returns the Redis client and ThreadService."""
    redis_client = redis.from_url(st.secrets["REDIS_URL"], decode_responses=True)
    memory = ThreadService(connection=conn, redis_client=redis_client)
    return memory

def call_helper_model(system_prompt: str, user_prompt: str) -> str:
    response = get_helper_model().invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ])
    return response.content

def reframeUserQuery(user_input: str, last_messages: List[dict]) -> dict:
    """
    Analyze deeply & decide if user input is a follow-up or is it related to previous questions.
    If yes -> reframe into an optimized query.
    If no -> return original query.
    """
    context = "\n".join(
        [f"{msg['role'].capitalize()}: {msg['content']}" for msg in last_messages]
    )

    user_prompt = f"""
    Conversation context: 
    {context}

    New user question:
    {user_input}
    """

    raw_response = call_helper_model(MEMORY_LAYER_PROMPT, user_prompt)
    try:
        result = parse_json(raw_response)

    except (json.JSONDecodeError, TypeError) as e:
        result = {
            "is_followup": False,
            "optimized_query": user_input,
            "selected_tools": []
        }

    return result

@st.cache_resource
def initialize_agent():
    """
    Initializes and compiles the LangGraph agent.
    This is cached to avoid rebuilding the graph on every interaction.
    """
    print("Initializing LangGraph agent...")

    def call_base_model(state: MessagesState) -> MessagesState:
        """
        Sends messages to the model and returns the response wrapped in MessagesState format.
        """
        response = get_base_model().invoke(input=state["messages"])
        return {"messages": [response]}

    def should_continue(state: MessagesState) -> bool:
        last_message = state["messages"][-1]
        return 'tools' if last_message.tool_calls else END

    builder = StateGraph(MessagesState)
    builder.add_node("call_base_model", call_base_model)
    builder.add_node("tools", tool_node)
    builder.add_edge(START, "call_base_model")
    builder.add_conditional_edges("call_base_model", should_continue, ["tools", END])
    builder.add_edge("tools", "call_base_model")

    agent_graph = builder.compile()
    return agent_graph

# --- Load cached resources ---
memory = get_memory()
email_agent_graph = initialize_agent()

# -------------------- SESSION STATE INITIALIZATION --------------------
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------- NEW: ENHANCED SIDEBAR --------------------
st.sidebar.title("Chat Sessions")

if st.sidebar.button("➕ New Chat", use_container_width=True):
    # Use the first user prompt as the title for the new chat
    st.session_state.thread_id = memory.createNewThread(user_id=USER_ID, title="New Conversation")
    st.session_state.messages = []
    st.rerun()

# --- NEW: Section for managing the CURRENTLY active chat ---
if st.session_state.thread_id:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Manage Chat")
    
    # Find the current thread's title to pre-fill the text input
    threads = memory.getThreads(USER_ID)
    current_thread = next((t for t in threads if t["id"] == st.session_state.thread_id), None)
    current_title = current_thread["title"] if current_thread else ""

    # RENAME functionality
    new_title = st.sidebar.text_input("Rename chat", value=current_title, key=f"rename_{st.session_state.thread_id}")
    if st.sidebar.button("Save Name", use_container_width=True):
        if new_title and new_title != current_title:
            memory.renameThread(st.session_state.thread_id, new_title)
            st.sidebar.success("Renamed!")
            st.rerun()

    # DELETE functionality with confirmation
    with st.sidebar.expander("Delete Chat"):
        st.warning("This action cannot be undone.")
        if st.button("Confirm Delete", use_container_width=True, type="primary"):
            memory.deleteThread(st.session_state.thread_id)
            st.session_state.thread_id = None
            st.session_state.messages = []
            st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("**Previous Chats**")

# List all existing threads with their creation dates
all_threads = memory.getThreads(USER_ID)
for thread in all_threads:
    # Use columns for a cleaner layout
    col1, col2 = st.sidebar.columns([3, 1])
    with col1:
        if st.button(thread["title"], key=thread["id"], use_container_width=True):
            st.session_state.thread_id = thread["id"]
            st.session_state.messages = [] 
            st.rerun()
    with col2:
        # Display the formatted date next to the button
        st.caption(thread["created_at"].strftime("%b %d"))

# -------------------- STREAMLIT UI --------------------
# st.set_page_config(page_title="AI Email Assistant", page_icon="📧")
st.title("📧 AI Email Assistant")
st.write("Ask me anything about your emails. I can search for content, filter by sender/date, and more.")

# If no thread is selected, show a welcome message
if not st.session_state.thread_id:
    st.info("Select a chat from the sidebar or start a new one.")
    st.stop()

# Load messages for the current thread if they haven't been loaded yet
if not st.session_state.messages:
    thread_history = memory.getThreadMessages(st.session_state.thread_id)
    messages_from_db = thread_history.get("messages", [])

    st.session_state.messages = messages_from_db

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if input := st.chat_input("Ask a question about your emails..."):
    # Add user's original message to chat history and display it
    st.session_state.messages.append({"role": "user", "content": input})
    with st.chat_message("user"):
        st.markdown(input)

    # Display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # --- START: UPDATED LOGIC FROM CHATBOT.PY ---

            # 1. Get the recent message history for context
            conversation_history = memory.getRecentThreadMessages(st.session_state.thread_id)

            # 2. Run the reframing function
            reframed = reframeUserQuery(input, conversation_history['messages'])

            # (Optional) Display the reframed query for debugging
            if reframed.get("is_followup"):
                st.info(f"Continuing conversation with query: `{reframed.get('optimized_query')}`")

            # 3. Prepare the internal message for the agent
            internal_message = {
                "query": reframed["optimized_query"],
                "selected_tools": reframed.get("selected_tools", []),
            }
            # st.write(internal_message)

            # 4. Construct the input for the agent, matching chatbot.py's structure
            initialState = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT.format(today_date=today_date)},
                    {"role": "user", "content": "optimized_query: " + json.dumps(internal_message)}
                ]
            }

            # 5. Invoke the agent to get the final response
            final_state = email_agent_graph.invoke(initialState)
            agent_answer = final_state["messages"][-1].content

            # --- END: UPDATED LOGIC FROM CHATBOT.PY ---
            st.markdown(agent_answer)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": agent_answer})

    # Save the user's original prompt and the agent's answer to the database
    memory.updateThreadMessages(st.session_state.thread_id, {"role": "user", "content": input})
    memory.updateThreadMessages(st.session_state.thread_id, {"role": "assistant", "content": agent_answer})