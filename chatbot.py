import os
import sys
import pytz
import asyncio
from datetime import datetime
from rich.console import Console
from rich.markdown import Markdown
from lib.utils import AGENT_MODEL, SYSTEM_PROMPT
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.postgres import Checkpoint
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, MessagesState, START, END
from tools.semantic_search_tool import semantic_search_tool
from tools.metadata_filtering_tool import metadata_filtering_tool
from tools.conversation_retriever_tool import conversation_retriever_tool
from tiktoken import encoding_for_model
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

# -------------------- CONFIG --------------------
console = Console()
IST = pytz.timezone("Asia/Kolkata")
today_date = datetime.now(IST).strftime("%B %d, %Y")

# Token limit settings
MAX_TOKENS = 8000
encoding = encoding_for_model("gpt-4o")
MAX_MESSAGES = 50  # Keep last 50 exchanges

# -------------------- INITIALIZE --------------------
# Initialize Postgres Checkpointer
DATABASE_URL = os.getenv("DATABASE_URL")
# checkpointer = PostgresSaver.from_conn_string(DATABASE_URL)
# print("Postgres checkpointer initialized.")

# Initialize ThreadService for thread management
# thread_service = ThreadService(connection=conn)
# print(thread_service, "DB connected")

# Define the list of tools the agent can use
tools = [semantic_search_tool, metadata_filtering_tool, conversation_retriever_tool]
tool_node = ToolNode(tools)

# -------------------- INITIALIZE THE MODEL --------------------
# Use LangGraph's init_chat_model
model = init_chat_model(model=AGENT_MODEL, temperature=0)
model_with_tools = model.bind_tools(tools)

# -------------------- HELPER FUNCTIONS --------------------
def token_count(messages):
    text = " ".join(m.content for m in messages)
    return len(encoding.encode(text))

def call_model(state: MessagesState) -> MessagesState:
    """
    Sends messages to the model and returns the response wrapped in MessagesState format.
    """
    messages = state["messages"]

    # Truncate context if too long
    if len(messages) > MAX_MESSAGES:
        messages = messages[-MAX_MESSAGES:]

    # Check token count
    if token_count(messages) > MAX_TOKENS - 500:
        return {"messages": [{"role": "system", "content": "Context limit reached. Please start a new thread."}]}

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

# Define an async function to process checkpoints from the memory
async def process_checkpoints(checkpoints):
    console.print("\n==========================================================\n")

    # Initialize an empty list to store the checkpoints
    checkpoints_list = []

    # Iterate over the checkpoints and add them to the list in an async way
    async for checkpoint_tuple in checkpoints:
        checkpoints_list.append(checkpoint_tuple)

    # Iterate over the list of checkpoints
    for _, checkpoint_tuple in enumerate(checkpoints_list):
        checkpoint = checkpoint_tuple.checkpoint
        messages = checkpoint["channel_values"].get("messages", [])

        # Display checkpoint information
        console.print(f"[white]Checkpoint:[/white]")
        console.print(f"[black]Timestamp: {checkpoint['ts']}[/black]")
        console.print(f"[black]Checkpoint ID: {checkpoint['id']}[/black]")

        # Display checkpoint messages
        for message in messages:
            if isinstance(message, HumanMessage):
                console.print(
                    f"[bright_magenta]User: {message.content}[/bright_magenta] [bright_cyan](Message ID: {message.id})[/bright_cyan]"
                )
            elif isinstance(message, AIMessage):
                console.print(
                    f"[bright_magenta]Agent: {message.content}[/bright_magenta] [bright_cyan](Message ID: {message.id})[/bright_cyan]"
                )

        console.print("")
    console.print("==========================================================")

# Define an async function to chat with the agent
async def main():
    async with AsyncConnectionPool(
        conninfo=DATABASE_URL,
        max_size=20,
        kwargs={
            "autocommit": True,
            "prepare_threshold": 0,
            "row_factory": dict_row,
        },
    ) as pool:
        async with pool.connection() as conn:
            # Register the vector data type with the database connection
            memory = AsyncPostgresSaver(conn)
            # await memory.setup()

            # Create a LangGraph agent
            builder = StateGraph(MessagesState)
            builder.add_node("call_model", call_model)
            builder.add_node("tools", tool_node)

            # Define edges
            builder.add_edge(START, "call_model")
            builder.add_conditional_edges("call_model", should_continue, ["tools", END])
            builder.add_edge("tools", "call_model")

            # Compile the graph
            email_agent_graph = builder.compile(checkpointer=memory)
            system_message = SystemMessage(content=SYSTEM_PROMPT)

            # Loop until the user chooses to quit the chat
            while True:
                user_input = input("\nAsk a question about your emails: ")
                if user_input.lower() == 'exit':
                    break

                # Prepare messages (i.e., human and system messages) to be passed to the LangGraph agent
                # Add the user's question to the HumanMessage object
                messages = [system_message, HumanMessage(content=user_input)]

                # Use the async stream method of the LangGraph agent to get the agent's answer
                events = email_agent_graph.astream({"messages": messages}, {"configurable": {"thread_id": "68b30d4f-275c-8329-b60f-967dec0d079f"}})
                async for event in events:
                    for _, value in event.items():
                        if "messages" in value:
                            agent_answer = value["messages"][-1].content

                # Use the async list method of the memory to list all checkpoints that match a given configuration
                checkpoints = memory.alist({"configurable": {"thread_id": "68b30d4f-275c-8329-b60f-967dec0d079f"}})
                # Process the checkpoints from the memory in an async way
                await process_checkpoints(checkpoints)

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