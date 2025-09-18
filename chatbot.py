import os
import re
import sys
import json
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional

import pytz
import redis
from rich.console import Console
from rich.markdown import Markdown
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, MessagesState, START, END

# ---- Project Imports ----
load_dotenv()
from lib.utils import AGENT_MODEL, SYSTEM_PROMPT, MEMORY_LAYER_PROMPT
from lib.db.db_service import ThreadService
from lib.db.db_conn import conn
from tools.semantic_search_tool import semantic_search_tool
from tools.metadata_filtering_tool import email_filtering_tool
from tools.sentiment_analysis_tool import sentiment_analysis_tool

# ============================================================
# CONFIG & GLOBALS
# ============================================================
console = Console()
IST = pytz.timezone("Asia/Kolkata")
today_date = datetime.now(IST).strftime("%B %d, %Y")

DATABASE_URL = os.getenv("DATABASE_URL")
REDIS_URL = os.getenv("REDIS_URL")

# Safe Redis client
redis_client: Optional[redis.Redis] = None
if REDIS_URL:
    try:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        console.log("[green]Connected to Redis[/green]")
    except Exception as e:
        console.log(f"[red]Redis connection failed: {e}[/red]")

# Thread memory service
memory = ThreadService(connection=conn, redis_client=redis_client)

# LangGraph model + tools
tools = [semantic_search_tool, email_filtering_tool, sentiment_analysis_tool]
tool_node = ToolNode(tools)

base_model = init_chat_model(model=AGENT_MODEL, temperature=0)
model_with_tools = base_model.bind_tools(tools)

llm = init_chat_model(model=AGENT_MODEL, temperature=0)
llm_with_tools = llm.bind_tools(tools)  # for small helper tasks

# ============================================================
# HELPER FUNCTIONS
# ============================================================
async def call_llm(system_prompt: str, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # invoke expects a list of message dicts (or LangChain Message objects)
    response = await llm_with_tools.ainvoke(messages)

    # response is an AIMessage object
    return response.content

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

def parse_json(raw_response):
    if not raw_response:
        return None
    match = re.search(r'\{.*\}', raw_response, re.S)
    if match:
        return json.loads(match.group(0))
    return None

async def reframe_user_query(user_input: str, last_messages: List[dict]) -> dict:
    """
    Analyze deeply & decide if user input is a follow-up or is it related to previous questions.
    If yes -> reframe into an optimized query.
    If no -> return original query.
    """
    context = "\n".join(
        [f"{msg['role'].capitalize()}: {msg['content']}" for msg in last_messages]
    )

    user_prompt = f"""
    Conversation context (last 10 messages): 
    {context}

    New user question:
    {user_input}
    """

    raw_response = await call_llm(MEMORY_LAYER_PROMPT, user_prompt)
    try:
        result = parse_json(raw_response)
    except (json.JSONDecodeError, TypeError) as e:
        result = {
            "is_followup": False,
            "optimized_query": user_input,
            "selected_tools": []
        }

    return result

def build_agent_graph() -> StateGraph:
    """Compile the LangGraph agent once."""
    builder = StateGraph(MessagesState)
    builder.add_node("call_model", call_model)
    builder.add_node("tools", tool_node)
    builder.add_edge(START, "call_model")
    builder.add_conditional_edges("call_model", should_continue, ["tools", END])
    builder.add_edge("tools", "call_model")
    return builder.compile()

# ============================================================
# CHAT LOOP
# ============================================================
async def chat_loop(user_id: str) -> None:
    """Main interactive chat loop."""
    # get or create thread
    thread_id = memory.get_last_thread(user_id) or memory.create_new_thread(
        user_id=user_id,
        title="Email's related questions"
    )
    console.log(f"[cyan]Using thread {thread_id}[/cyan]")
    email_agent_graph = build_agent_graph()

    while True:
        user_input = input("\nAsk a question about your emails (type 'exit' to quit): ")
        if user_input.strip().lower() == "exit":
            break

        last_msgs = memory.get_thread_messages(thread_id).get("messages", [])
        reframed = await reframe_user_query(user_input, last_msgs)
        print(reframed, 'reframed')

        if reframed["is_followup"]:
            internal_message = {
                "query": reframed["optimized_query"],
                "selected_tools": reframed.get("selected_tools", []),
            }
        else:
            internal_message = user_input

        # Prepare initial state
        initial_state: Dict[str, Any] = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT.format(today_date=today_date)},
                *last_msgs[-5:],
                {"role": "user", "content": "optimized_query " + json.dumps(internal_message)}
            ]
        }

        # Stream events
        events = email_agent_graph.astream(initial_state)
        async for event in events:
            for _, value in event.items():
                if "messages" in value:
                    agent_answer = value["messages"][-1].content

        memory.put_thread_message(thread_id, {"role": "user", "content": user_input})
        memory.put_thread_message(thread_id, {"role": "assistant", "content": agent_answer})

        # Display the agent's answer
        print("\n--- Final Answer ---")
        console.print(Markdown(agent_answer))
        print("--------------------\n")

# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    USER_ID = "63f05e7a-35ac-4deb-9f38-e2864cdf3a1d"
    asyncio.run(chat_loop(USER_ID))