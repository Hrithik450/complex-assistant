import pytz
from datetime import datetime
from lib.utils import AGENT_MODEL
from rich.console import Console
from rich.markdown import Markdown
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, MessagesState, START, END
from tools.semantic_search_tool import semantic_search_tool
from tools.metadata_filtering_tool import metadata_filtering_tool
from tools.conversation_retriever_tool import conversation_retriever_tool

# Datetime setup
IST = pytz.timezone("Asia/Kolkata")
today_date = datetime.now(IST).strftime("%B %d, %Y")

# -------------------- SYSTEM PROMPT --------------------
system_prompt = """You are an intelligent email assistant. 
Answer user questions based only on the information available in the emails.

Use the available tools to find relevant information and summarize it clearly.
Include metadata such as thread ID, sender, recipient, subject, and date when relevant.
If no information is found, state that clearly.
Today's date is {today_date} IST.
"""

# -------------------- CREATE THE AGENT --------------------
# Define the list of tools the agent can use
tools = [semantic_search_tool, metadata_filtering_tool, conversation_retriever_tool]

# Wrap tools in a ToolNode
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
email_agent_graph = builder.compile()

# -------------------- RUN THE AGENT --------------------
if __name__ == "__main__":
    print("\nEmail Chatbot is ready. Type 'exit' to end the session.")
    while True:
        user_input = input("\nAsk a question about your emails: ")
        if user_input.lower() == 'exit':
            break

        initial_state = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ]
        }
        response_state = email_agent_graph.invoke(initial_state)
        final_message = response_state["messages"][-1]
        
        console = Console()
        md = Markdown(final_message.content)
        print("\n--- Final Answer ---")
        console.print(md)
        print("--------------------\n")