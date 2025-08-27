import os
import logging
from dotenv import load_dotenv
import streamlit as st
import json
import pandas as pd
import faiss
import pickle
from datetime import datetime
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np

# --- LangChain Imports ---
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults

# --- Import our existing, well-defined components from agent_pro ---
# This treats agent_pro.py as a library of our core tools and data manager
from agent_pro import (
    SentenceTransformerEmbeddings,
    DataManager, # We'll use this for the history DB logic
    AliasResolver
)

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- NEW: Point to the dedicated email knowledge base files ---
EMAIL_FAISS_INDEX_PATH = os.path.join(SCRIPT_DIR, "emails_faiss.bin")
EMAIL_METADATA_PATH = os.path.join(SCRIPT_DIR, "emails_metadata.pkl")
FINETUNED_MODEL_PATH = os.path.join(SCRIPT_DIR, "finetuned_bge_real_estate_model")

class EmailRAGTool:
    """A specialized RAG tool that only searches the email knowledge base."""
    def __init__(self):
        logging.info("Initializing Email RAG Tool...")
        # Load the embedding model
        self.embedding_model = SentenceTransformer(FINETUNED_MODEL_PATH, device='cpu')
        
        # Load the email-specific FAISS index and metadata
        self.faiss_index = faiss.read_index(EMAIL_FAISS_INDEX_PATH)
        with open(EMAIL_METADATA_PATH, "rb") as f:
            self.metadata_store = pickle.load(f)
        
        logging.info(f"Email RAG Tool loaded with {self.faiss_index.ntotal} email vectors.")

    def run(self, query: str) -> str:
        """Performs a semantic search on the email knowledge base."""
        query_embedding = self.embedding_model.encode([query])
        query_embedding_np = np.array(query_embedding).astype('float32')
        
        # Search for the top 15 most relevant email chunks
        distances, indices = self.faiss_index.search(query_embedding_np, k=15)
        
        retrieved_docs = [self.metadata_store[i] for i in indices[0] if i != -1]
        
        if not retrieved_docs:
            return "No relevant emails found."
            
        # Format the results for the LLM
        context = "\n---\n".join([
            f"Source: {doc.get('source', 'N/A')}\n"
            f"From: {doc.get('from', 'N/A')}\n"
            f"To: {doc.get('to', 'N/A')}\n"
            f"Subject: {doc.get('subject', 'N/A')}\n"
            f"Date: {doc.get('parsed_date', 'N/A')}\n"
            f"Summary: {doc.get('summary', 'N/A')}\n"
            f"Content Snippet: {doc.get('original_text', '')}"
            for doc in retrieved_docs
        ])
        return context

class EmailPandasTool:
    """A specialized Pandas tool that only uses the email metadata."""
    def __init__(self):
        logging.info("Initializing Email Pandas Tool...")
        with open(EMAIL_METADATA_PATH, "rb") as f:
            metadata = pickle.load(f)
        
        self.df = pd.DataFrame(metadata)
        if 'parsed_date' in self.df.columns:
            self.df['parsed_date'] = pd.to_datetime(self.df['parsed_date'], errors='coerce')
        
        # Create the Pandas Agent
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        df_prefix = """
        You are a world-class data analyst working with a pandas DataFrame named `df` that contains ONLY email metadata.

        **CRITICAL RULES FOR QUERYING:**
        1.  **USE `parsed_date` FOR ALL DATE OPERATIONS**: This column is a standardized datetime object.
        2.  **COUNT UNIQUE ITEMS**: To count unique emails or threads, use `nunique()` on the `id` or `threadId` columns.
        3.  **FILTER BY NAME**: When filtering by a person's name (in `from`, `to`, or `cc`), always use a case-insensitive, partial string match.

        **DataFrame Schema:**
        - `source`, `file_type`, `from`, `to`, `cc`, `subject`, `id`, `threadId`, `parsed_date`, `summary`
        """
        self.agent_executor = create_pandas_dataframe_agent(
            llm=llm, df=self.df, agent_type="openai-tools",
            verbose=True, allow_dangerous_code=True, prefix=df_prefix
        )

    def run(self, query: str) -> str:
        """Runs a query against the email metadata DataFrame."""
        logging.info(f"\n[EmailPandasTool] Executing query: '{query}'")
        result = self.agent_executor.invoke({"input": query})
        return result.get('output', "Query executed, but no output was returned.")

class EmailSpecialistAgent:
    """
    The definitive ReAct agent, equipped with specialized email tools and a robust reasoning prompt.
    """
    def __init__(self):
        logging.info("\n[*] Initializing Email Specialist Agent...")
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        
        # The DataManager is now only used for conversation history
        self.data_manager = DataManager()

        # --- Assemble the specialized email tools ---
        email_rag_tool = EmailRAGTool()
        email_pandas_tool = EmailPandasTool()
        web_search_tool = TavilySearchResults(max_results=3)

        self.tools = [
            Tool(name="EmailDataAnalyzer", func=email_pandas_tool.run, description="Use for quantitative questions about email METADATA (how many, list, count, sort, latest, earliest)."),
            Tool(name="EmailContentSearch", func=email_rag_tool.run, description="Use for qualitative questions that require reading and summarizing email CONTENT (what is, summarize, sentiment)."),
            Tool(name="WebSearch", func=web_search_tool.invoke, description="Use this as a LAST RESORT if the information is not found in the emails. Good for public information about companies or people.")
        ]

        # --- THIS IS THE FIX: Use a template with the required placeholders ---
        prompt_template = """
        You are a master AI assistant specializing in analyzing a private database of company emails. Your job is to use the tools at your disposal to answer the user's question.
        Today's date is {current_date}. Use this for any relative date calculations (e.g., "last year").

        **CONVERSATION HISTORY:**
        ---
        {chat_history}
        ---

        **AVAILABLE TOOLS:**
        ---
        {tools}
        ---

        **--- AGENT CONSTITUTION (CRITICAL RULES) ---**
        1.  **ANALYZE THE QUERY:** Understand the user's true intent from the "New input" and "CONVERSATION HISTORY".
        2.  **CHOOSE THE RIGHT TOOL:**
            - For counting, listing, or finding emails by date (e.g., "latest," "last year"), you MUST use `EmailDataAnalyzer`.
            - For understanding or summarizing the content of emails (e.g., "what is," "summarize"), you MUST use `EmailContentSearch`.
        3.  **MULTI-STEP REASONING:** For complex queries like "Summarize the last email from Raja," you must chain your thoughts. First, find the email with `EmailDataAnalyzer`, then use `EmailContentSearch` to get the content for summarization.
        4.  **OUTPUT FORMATTING (THE GOLDEN RULE):** You MUST use the following format. Do not add any extra text or formatting.
            ```
            Thought: Do I need to use a tool? Yes
            Action: The action to take, should be one of [{tool_names}]
            Action Input: The input to the action
            ```
            OR, if you have the final answer:
            ```
            Thought: I now have the final answer.
            Final Answer: The final answer to the original input question.
            ```

        **--- START OF CURRENT TASK ---**

        **New input:** {input}
        **Agent scratchpad:** {agent_scratchpad}
        """
        
        # Create the prompt template object
        prompt = ChatPromptTemplate.from_template(prompt_template)
        
        # Create the ReAct agent, allowing it to format the tools itself
        agent = create_react_agent(self.llm, self.tools, prompt)
        
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=10
        )
        logging.info("[+] Email Specialist Agent is ready.")

    def run(self, user_query: str, session_id: str):
        """Runs the agent and logs the interaction."""
        recent_history = self.data_manager.get_recent_history(session_id)
        chat_history_str = "\n".join([f"User: {h['query']}\nAssistant: {h['response']}" for h in recent_history])
        
        current_date_str = datetime.now().strftime('%Y-%m-%d')
        
        input_payload = {
            "input": user_query,
            "chat_history": chat_history_str,
            "current_date": current_date_str
        }
        
        result = self.agent_executor.invoke(input_payload)
        
        response = result.get('output', 'Agent stopped due to an unexpected error.')
        
        record_id = self.data_manager.log_interaction(session_id, user_query, response)
        
        return response, record_id

# --- Main block for local testing ---
if __name__ == "__main__":
    load_dotenv()
    if not all(os.getenv(var) for var in ["OPENAI_API_KEY", "TAVILY_API_KEY"]):
        print("[!] FATAL: Missing one or more required environment variables.")
    else:
        try:
            agent = EmailSpecialistAgent()
            session_id = "cli_session_email_agent"
            print("\n--- Email Specialist AI Analyst is Ready ---")
            print("Ask a question about your emails (or type 'exit' to quit).")
            while True:
                user_input = input("> ")
                if not user_input or user_input.isspace():
                    print("Please enter a question.")
                    continue
                if user_input.lower() == 'exit': break
                response, record_id = agent.run(user_query=user_input, session_id=session_id)
                print("\n" + "="*50 + " FINAL ANSWER " + "="*50)
                print(response)
                print(f"(Reference ID for feedback: {record_id})")
        except Exception as e:
            print(f"\n[!] A critical error occurred: {e}")
            import traceback
            traceback.print_exc()