import os
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv

# Load environment variables for local execution
load_dotenv()

# Tavily automatically uses the TAVILY_API_KEY from the environment.
# For Streamlit, we need to ensure the key is passed if it's not in the environment.
# However, LangChain's integration is smart enough to pick it up if the env var is set.
# We can initialize it with a max_results parameter.
web_search_tool = TavilySearchResults(max_results=3)

# We can directly use the initialized object as the tool.
# To give it a more specific docstring for our agent, we can wrap it.
web_search_tool.name = "web_search_tool"
web_search_tool.description = """
A powerful search engine for finding information on the internet.
Use this tool as a last resort if you cannot find the answer to a user's question in the email database.
It is best for general knowledge questions, current events, or topics not related to the user's personal emails.
"""

# The 'web_search_tool' object itself is the tool we will import.