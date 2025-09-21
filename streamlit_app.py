# --- THIS IS THE FIX ---
# It checks if the app is running on Streamlit Cloud by looking for a specific environment variable.
# The sqlite3 patch will ONLY run when deployed to the cloud.
import os
import sys

import nest_asyncio
nest_asyncio.apply()

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
# --- END OF FIX ---

import re
import json
import asyncio
from typing import List, Dict, Any, Optional, Coroutine
import streamlit as st
st.set_page_config(page_title="AI Email Assistant", page_icon="📧")
import pytz
import redis
from datetime import datetime
from rich.console import Console
from rich.markdown import Markdown

# Import the tools and agent components from your existing files
from lib.utils import AGENT_MODEL, SYSTEM_PROMPT
from langchain.chat_models import init_chat_model
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, MessagesState, START, END
from tools.semantic_search_tool import semantic_search_tool
from tools.metadata_filtering_tool import email_filtering_tool
from tools.conversation_retriever_tool import conversation_retriever_tool
from tools.sentiment_analysis_tool import sentiment_analysis_tool
from tools.web_search_tool import web_search_tool
from tools.summarization_tool import summarization_tool


# This will trigger the data loading and Chroma connection via st.cache_resource
from lib.load_data import df, chroma_collection

# Import your database logic
from lib.db.db_service import ThreadService
from lib.db.db_conn import conn

from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from lib.utils import MEMORY_LAYER_PROMPT

# -------------------- CONFIG --------------------
IST = pytz.timezone("Asia/Kolkata")
today_date = datetime.now(IST).strftime("%B %d, %Y")
USER_ID = "63f05e7a-35ac-4deb-9f38-e2864cdf3a1d" # Hardcoded for this example

# -------------------- CACHED RESOURCES --------------------
@st.cache_resource
def get_helper_llm():
    """Initializes a separate, cached LLM for helper tasks like query reframing."""
    # return ChatOpenAI(model="gpt-4o", temperature=0, api_key=st.secrets["OPENAI_API_KEY"])
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-pro", 
        temperature=0, 
        google_api_key=st.secrets["GOOGLE_API_KEY"]
    )

def call_llm(system_prompt: str, user_prompt: str) -> str:
    """Async helper to call the LLM."""
    llm = get_helper_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", user_prompt),
    ])
    chain = prompt | llm | StrOutputParser()
    # Note: In Streamlit, it's often simpler to use .invoke() for synchronous calls
    # unless you are building a fully async app.
    response = chain.invoke({"system_prompt": system_prompt, "user_prompt": user_prompt})
    return response

def parse_json(raw_response):
    """Safely parses a JSON object from a string."""
    if not raw_response:
        return None
    # Use a more robust regex to find the JSON object
    match = re.search(r'```json\s*(\{.*?\})\s*```', raw_response, re.S)
    if not match:
        match = re.search(r'(\{.*?\})', raw_response, re.S)

    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            return None # Or handle the error as needed
    return None

def reframe_user_query(user_input: str, last_messages: list) -> dict:
    """
    Analyzes user input in the context of the conversation to create an optimized query.
    """
    context = "\n".join(
        [f"{msg['role'].capitalize()}: {msg['content']}" for msg in last_messages]
    )
    user_prompt = f"Conversation context:\n{context}\n\nNew user question:\n{user_input}"

    raw_response = call_llm(MEMORY_LAYER_PROMPT, user_prompt)
    try:
        result = parse_json(raw_response)
        if result is None: # Handle cases where JSON isn't found
            raise json.JSONDecodeError("No JSON found", raw_response, 0)
    except (json.JSONDecodeError, TypeError):
        result = {"is_followup": False, "optimized_query": user_input, "selected_tools": []}

    return result

def get_memory():
    """Initializes and returns the Redis client and ThreadService."""
    redis_client = redis.from_url(st.secrets["REDIS_URL"], decode_responses=True)
    memory = ThreadService(connection=conn, redis_client=redis_client)
    return memory

@st.cache_resource
def initialize_agent():
    """
    Initializes and compiles the LangGraph agent.
    This is cached to avoid rebuilding the graph on every interaction.
    """
    print("Initializing LangGraph agent...")
    tools = [semantic_search_tool, email_filtering_tool, conversation_retriever_tool, sentiment_analysis_tool, summarization_tool, web_search_tool]
    tool_node = ToolNode(tools)

    # Use Streamlit secrets for the OpenAI API key
    # model = init_chat_model(model=AGENT_MODEL, temperature=0, api_key=st.secrets["OPENAI_API_KEY"])
    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        temperature=0,
        google_api_key=st.secrets["GOOGLE_API_KEY"]
    )
    model_with_tools = model.bind_tools(tools)

    def call_model(state: MessagesState) -> MessagesState:
        messages = state["messages"]
        response = model_with_tools.invoke(input=messages)
        return {"messages": [response]}

    def should_continue(state: MessagesState) -> bool:
        last_message = state["messages"][-1]
        return 'tools' if last_message.tool_calls else END

    builder = StateGraph(MessagesState)
    builder.add_node("call_model", call_model)
    builder.add_node("tools", tool_node)
    builder.add_edge(START, "call_model")
    builder.add_conditional_edges("call_model", should_continue, ["tools", END])
    builder.add_edge("tools", "call_model")

    agent_graph = builder.compile()
    print("LangGraph agent initialized successfully.")
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
    st.session_state.thread_id = memory.create_new_thread(user_id=USER_ID, title="New Conversation")
    st.session_state.messages = []
    st.rerun()

# --- NEW: Section for managing the CURRENTLY active chat ---
if st.session_state.thread_id:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Manage Chat")
    
    # Find the current thread's title to pre-fill the text input
    threads = memory.get_all_threads_for_user(USER_ID)
    current_thread = next((t for t in threads if t["id"] == st.session_state.thread_id), None)
    current_title = current_thread["title"] if current_thread else ""

    # RENAME functionality
    new_title = st.sidebar.text_input("Rename chat", value=current_title, key=f"rename_{st.session_state.thread_id}")
    if st.sidebar.button("Save Name", use_container_width=True):
        if new_title and new_title != current_title:
            memory.rename_thread(st.session_state.thread_id, new_title)
            st.sidebar.success("Renamed!")
            st.rerun()

    # DELETE functionality with confirmation
    with st.sidebar.expander("Delete Chat"):
        st.warning("This action cannot be undone.")
        if st.button("Confirm Delete", use_container_width=True, type="primary"):
            memory.delete_thread(st.session_state.thread_id)
            st.session_state.thread_id = None
            st.session_state.messages = []
            st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("**Previous Chats**")

# List all existing threads with their creation dates
all_threads = memory.get_all_threads_for_user(USER_ID)
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
    thread_history = memory.get_thread_messages(st.session_state.thread_id)
    messages_from_db = thread_history.get("messages", [])
    # 1. Group the flat list into pairs of (assistant, user)
    it = iter(messages_from_db)
    # This creates pairs like [(assistant_2, user_2), (assistant_1, user_1)]
    grouped_pairs = list(zip(it, it))

    # 2. Reverse the order of the pairs to get chronological order
    # Now we have [(assistant_1, user_1), (assistant_2, user_2)]
    grouped_pairs.reverse()

    # 3. Flatten the list, swapping the order within each pair to [user, assistant]
    corrected_messages = []
    for assistant_msg, user_msg in grouped_pairs:
        corrected_messages.append(user_msg)
        corrected_messages.append(assistant_msg)

    st.session_state.messages = corrected_messages

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask a question about your emails..."):
    # Add user's original message to chat history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # --- START: UPDATED LOGIC FROM CHATBOT.PY ---

            # 1. Get the recent message history for context
            history_for_reframing = st.session_state.messages[:-1]

            # 2. Run the reframing function
            reframed = reframe_user_query(prompt, history_for_reframing)

            # (Optional) Display the reframed query for debugging
            if reframed.get("is_followup"):
                st.info(f"Continuing conversation with query: `{reframed.get('optimized_query')}`")

            # 3. Prepare the internal message for the agent
            internal_message = {
                "query": reframed["optimized_query"],
                "selected_tools": reframed.get("selected_tools", []),
            }

            # 4. Construct the input for the agent, matching chatbot.py's structure
            initialState = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT.format(today_date=today_date)},
                    # Use only the last 5 messages for context
                    *history_for_reframing[-5:],
                    # The final user message is prefixed and contains the JSON-dumped internal message
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
    memory.put_thread_message(st.session_state.thread_id, [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": agent_answer}
    ])

# # Accept user input
# if prompt := st.chat_input("Ask a question about your emails..."):
#     # Add user's original message to chat history and display it
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     with st.chat_message("user"):
#         st.markdown(prompt)

#     # Display assistant response
#     with st.chat_message("assistant"):
#         with st.spinner("Thinking..."):
#             # --- NEW LOGIC: Reframe the query before calling the agent ---
            
#             # 1. Get the recent message history for context
#             history_for_reframing = st.session_state.messages[:-1]

#             # 2. Run the async reframing function
#             # reframed = asyncio.run(reframe_user_query(prompt, history_for_reframing))
#             reframed = reframe_user_query(prompt, history_for_reframing)

#             # (Optional) Display the reframed query for debugging
#             if reframed["is_followup"]:
#                 st.info(f"Continuing conversation with query: `{reframed['optimized_query']}`")

#             # 3. Prepare the message for the agent, matching chatbot.py's logic
#             if reframed["is_followup"]:
#                 # Include selected_tools and the optimized query
#                 internal_message = {
#                     "query": reframed["optimized_query"],
#                     "selected_tools": reframed.get("selected_tools", []),
#                 }
#             else:
#                 # Use the raw user input string for non-follow-ups
#                 internal_message = prompt

#             # 4. Construct the input for the agent, matching chatbot.py's structure
#             initialState = {
#                 "messages": [
#                     {"role": "system", "content": SYSTEM_PROMPT.format(today_date=today_date)},
#                     # Use only the last 5 messages for context
#                     *history_for_reframing[-5:],
#                     # The final user message is prefixed and contains the JSON-dumped internal message
#                     {"role": "user", "content": "optimized_query: " + json.dumps(internal_message)}
#                 ]
#             }
            
#             # 5. Invoke the agent to get the final response
#             final_state = email_agent_graph.invoke(initialState)
#             agent_answer = final_state["messages"][-1].content
            
#             st.markdown(agent_answer)
    
#     # Add assistant response to chat history
#     st.session_state.messages.append({"role": "assistant", "content": agent_answer})

#     # Save the user's original prompt and the agent's answer to the database
#     memory.put_thread_message(st.session_state.thread_id, [
#         {"role": "user", "content": prompt},
#         {"role": "assistant", "content": agent_answer}
#     ])
