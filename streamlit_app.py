import streamlit as st
st.set_page_config(page_title="AI Email Assistant", page_icon="📧")
import pytz
from datetime import datetime
from rich.console import Console
from rich.markdown import Markdown

# Import the tools and agent components from your existing files
from lib.utils import AGENT_MODEL, SYSTEM_PROMPT
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, MessagesState, START, END
from tools.semantic_search_tool import semantic_search_tool
from tools.metadata_filtering_tool import metadata_filtering_tool
from tools.conversation_retriever_tool import conversation_retriever_tool

# This will trigger the data loading and Chroma connection via st.cache_resource
from lib.load_data import df, chroma_collection

# -------------------- CONFIG --------------------
IST = pytz.timezone("Asia/Kolkata")
today_date = datetime.now(IST).strftime("%B %d, %Y")

# -------------------- AGENT SETUP (CACHED) --------------------
@st.cache_resource
def initialize_agent():
    """
    Initializes and compiles the LangGraph agent.
    This is cached to avoid rebuilding the graph on every interaction.
    """
    print("Initializing LangGraph agent...")
    tools = [semantic_search_tool, metadata_filtering_tool, conversation_retriever_tool]
    tool_node = ToolNode(tools)

    # Use Streamlit secrets for the OpenAI API key
    model = init_chat_model(model=AGENT_MODEL, temperature=0, api_key=st.secrets["OPENAI_API_KEY"])
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

# -------------------- STREAMLIT UI --------------------
# st.set_page_config(page_title="AI Email Assistant", page_icon="📧")
st.title("📧 AI Email Assistant")
st.write("Ask me anything about your emails. I can search for content, filter by sender/date, and more.")

# Initialize the agent
email_agent_graph = initialize_agent()

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask a question about your emails..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Construct the input for the agent
            # Note: We are not using the database/redis memory here for simplicity in Streamlit,
            # but using the session state history instead.
            chat_history_for_agent = [msg for msg in st.session_state.messages if msg["role"] != "system"]

            initialState = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT.format(today_date=today_date)},
                    *chat_history_for_agent,
                ]
            }
            
            # Invoke the agent to get the final response
            # .invoke is synchronous and simpler for a direct request-response in Streamlit
            final_state = email_agent_graph.invoke(initialState)
            agent_answer = final_state["messages"][-1].content
            
            st.markdown(agent_answer)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": agent_answer})