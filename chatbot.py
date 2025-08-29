import pytz
from datetime import datetime
from lib.utils import AGENT_MODEL, SYSTEM_PROMPT
from rich.console import Console
from rich.markdown import Markdown
from langgraph.checkpoint.memory import InMemorySaver
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, MessagesState, START, END
from tools.semantic_search_tool import semantic_search_tool
from tools.metadata_filtering_tool import metadata_filtering_tool
from tools.conversation_retriever_tool import conversation_retriever_tool
from lib.db.db_service import ThreadService
from lib.db.db_conn import conn

# Datetime setup
IST = pytz.timezone("Asia/Kolkata")
today_date = datetime.now(IST).strftime("%B %d, %Y")

# -------------------- CREATE THE AGENT --------------------
# Initialize Postgres Checkpointer
checkpoint = InMemorySaver()
postgres_checkpointer = ThreadService(connection=conn)
print(postgres_checkpointer, "db success")

# Define the list of tools the agent can use
tools = [semantic_search_tool, metadata_filtering_tool, conversation_retriever_tool]
tool_node = ToolNode(tools)

# -------------------- INITIALIZE THE MODEL --------------------
# Use LangGraph's init_chat_model
model = init_chat_model(model=AGENT_MODEL, temperature=0)

# Bind the tools to the model
model_with_tools = model.bind_tools(tools)

def call_model(state: MessagesState) -> MessagesState:
    """
    Sends messages to the model and returns the response wrapped in MessagesState format.
    """
    messages = state["messages"]
    response = model_with_tools.invoke(input=messages)
    return {"messages": [response]}

def should_continue(state: MessagesState) -> bool:
    """
    Decides whether to call tools next based on the last model output.
    """
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return 'tools'
    return END

# -------------------- BUILD THE STATE GRAPH --------------------
builder = StateGraph(MessagesState)
builder.add_node("call_model", call_model)
builder.add_node("tools", tool_node)

# Define edges
builder.add_edge(START, "call_model")
builder.add_conditional_edges("call_model", should_continue, ["tools", END])
builder.add_edge("tools", "call_model")

# Compile the graph
email_agent_graph = builder.compile(checkpointer=checkpoint)

# -------------------- RUN THE AGENT --------------------
if __name__ == "__main__":
    print("\nEmail Chatbot is ready. Type 'exit' to end the session.")

    # Create a new thread for this chat
    USER_ID = "68b1bac3-89bc-8324-be3a-97231ecc5e8a"

    # Check if user already has a thread
    thread_id = postgres_checkpointer.get_last_thread(USER_ID)

    if thread_id:
        print(f"Resuming last thread: {thread_id}")
    else:
        thread_id = postgres_checkpointer.create_new_thread(
            user_id=USER_ID, 
            system_prompt=SYSTEM_PROMPT, 
            thread_name="Email's related qns"
        )
        print(f"Created new thread: {thread_id}")
        print(f"New chat started. Thread ID: {thread_id}")

    config = {"configurable": {"thread_id": thread_id}}

    while True:
        user_input = input("\nAsk a question about your emails: ")
        if user_input.lower() == 'exit':
            break

        # # Fetch previous state from Postgres (context)
        # previous_state = postgres_checkpointer.get_tuple(config)

        # # Add new user message to state
        # previous_state['checkpoint']["messages"].append({"role": "user", "content": user_input})

        # Stream events
        initial_state = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_input}
            ]
        }
        events = email_agent_graph.stream(initial_state, config, stream_mode="values")

        final_message = None
        for event in events:
            final_message = event["messages"][-1]

        # # Save conversation state to Postgres
        # postgres_checkpointer.put_tuple(config, state=previous_state)
        
        console = Console()
        message = Markdown(final_message.content)
        print("\n--- Final Answer ---")
        console.print(message)
        print("--------------------\n")

# from tiktoken import encoding_for_model

# MAX_TOKENS = 8000
# encoding = encoding_for_model("gpt-4")

# def token_count(messages):
#     text = " ".join(m["content"] for m in messages)
#     return len(encoding.encode(text))

# def call_model(state: MessagesState) -> MessagesState:
#     messages = state["messages"]

#     if token_count(messages) > MAX_TOKENS - 500:  # Keep buffer for response
#         print("Context limit reached. Ending session.")
#         return {"messages": [{"role": "system", "content": "Context limit reached. Please start a new thread."}]}

#     response = model_with_tools.invoke(input=messages)
#     return {"messages": [response]}