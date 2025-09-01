import os
import sys
import pytz
import redis
import asyncio
from datetime import datetime
from rich.console import Console
from rich.markdown import Markdown
from lib.utils import AGENT_MODEL, SYSTEM_PROMPT
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, MessagesState, START, END
from tools.semantic_search_tool import semantic_search_tool
from tools.metadata_filtering_tool import email_filtering_tool
from tools.conversation_retriever_tool import conversation_retriever_tool
from tools.web_search_tool import web_search_tool
from lib.db.db_service import ThreadService
from lib.db.db_conn import conn

# -------------------- CONFIG --------------------
console = Console()
IST = pytz.timezone("Asia/Kolkata")
today_date = datetime.now(IST).strftime("%B %d, %Y")

# -------------------- INITIALIZE --------------------
DATABASE_URL = os.getenv("DATABASE_URL")
REDIS_URL = os.getenv("REDIS_URL")

# Initialize Redis client
if REDIS_URL:
    redis_client = redis.from_url(REDIS_URL, decode_responses=True)
print(redis_client, 'redis')

# Define the list of tools the agent can use
tools = [semantic_search_tool, email_filtering_tool, conversation_retriever_tool, web_search_tool]
tool_node = ToolNode(tools)

# -------------------- INITIALIZE THE MODEL --------------------
# Use LangGraph's init_chat_model
model = init_chat_model(model=AGENT_MODEL, temperature=0)
model_with_tools = model.bind_tools(tools)

# -------------------- HELPER FUNCTIONS --------------------
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

# Initialize the ThreadService for managing chat threads
memory = ThreadService(connection=conn, redis_client=redis_client)

# Define an async function to chat with the agent
async def main():
    # Create a new thread for this chat
    USER_ID = "63f05e7a-35ac-4deb-9f38-e2864cdf3a1d"

    # Check if user already has a thread
    thread_id = memory.get_last_thread(USER_ID)

    if thread_id:
        print(f"Resuming last thread: {thread_id}")
    else:
        thread_id = memory.create_new_thread(
            user_id=USER_ID, 
            title="Email's related qns"
        )
        print(f"Created new thread: {thread_id}")
        print(f"New chat started. Thread ID: {thread_id}")

        # Create a LangGraph agent
    builder = StateGraph(MessagesState)
    builder.add_node("call_model", call_model)
    builder.add_node("tools", tool_node)

    # Define edges
    builder.add_edge(START, "call_model")
    builder.add_conditional_edges("call_model", should_continue, ["tools", END])
    builder.add_edge("tools", "call_model")

    # Compile the graph
    email_agent_graph = builder.compile()

    # Loop until the user chooses to quit the chat
    while True:
        last_20_messages = memory.get_thread_messages(thread_id)

        user_input = input("\nAsk a question about your emails: ")
        if user_input.lower() == 'exit':
            break
 
        initialState = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT.format(today_date=today_date)},
                *last_20_messages['messages'],
                {"role": "user", "content": user_input}
            ]
        }

        # Use the async stream method of the LangGraph agent to get the agent's answer
        events = email_agent_graph.astream(initialState)
        async for event in events:
            for _, value in event.items():
                if "messages" in value:
                    agent_answer = value["messages"][-1].content

        memory.put_thread_message(thread_id, [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": agent_answer}
        ])

        # Display the agent's answer
        print("\n--- Final Answer ---")
        console.print(Markdown(agent_answer))
        print("--------------------\n")

# -------------------- RUN THE AGENT --------------------
if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # Run the main async function
    asyncio.run(main())