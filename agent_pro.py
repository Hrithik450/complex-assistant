import os
import pickle
import faiss
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer, CrossEncoder
import logging
import re
import json
from datetime import datetime, timedelta
import warnings
import sqlite3
from typing import List

# --- LangChain Imports ---
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain.chains import LLMChain

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
COLLECTION_NAME = "real_estate_finetuned_local"
FAISS_INDEX_PATH = os.path.join(SCRIPT_DIR, f"{COLLECTION_NAME}_faiss.bin")
METADATA_PATH = os.path.join(SCRIPT_DIR, f"{COLLECTION_NAME}_metadata.pkl")
FINETUNED_MODEL_PATH = os.path.join(SCRIPT_DIR, "finetuned_bge_real_estate_model")
BRIEFING_DOC_NAME = "1.Briefing_to_2g_AI_Ally 25Jul2025.docx"
DB_PATH = os.path.join(SCRIPT_DIR, "history.db") # Path for the SQLite database


# --- CORE UTILITIES ---
class SentenceTransformerEmbeddings(Embeddings):
    """A LangChain-compatible adapter for SentenceTransformer models."""
    def __init__(self, model_object):
        self.model = model_object

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """LangChain expects this method for batch embedding."""
        return self.model.encode(texts).tolist()

    def embed_query(self, text: str) -> List[float]:
        """LangChain expects this method for single text embedding."""
        return self.model.encode(text).tolist()
    
class UnifiedDataParser:
    """A centralized class for parsing data to ensure consistency across all agents."""
    @staticmethod
    def parse_flexible_date(date_string: str):
        if not isinstance(date_string, str): return None
        formats_to_try = ['%m/%d/%y, %H:%M', '%A, %d %B, %Y %I.%M %p', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d']
        for fmt in formats_to_try:
            try: return datetime.strptime(date_string.strip(), fmt)
            except (ValueError, TypeError): continue
        return None

class AliasResolver:
    """NEW: Resolves short names and aliases into a list of full names from the database."""
    def __init__(self, data_manager):
        self.dm = data_manager
        # Create a unique, clean list of all known senders and 'from' names
        from_names = self.dm.df['from'].dropna().unique().tolist()
        sender_names = self.dm.df['sender'].dropna().unique().tolist()
        self.all_known_names = list(set(from_names + sender_names))
        logging.info(f"[AliasResolver] Initialized with {len(self.all_known_names)} unique names.")

    def resolve(self, alias: str) -> list[str]:
        alias_lower = alias.lower()
        
        # Simple direct match first for performance
        direct_matches = [name for name in self.all_known_names if alias_lower in name.lower()]
        if direct_matches:
            logging.info(f"[AliasResolver] Found direct matches for '{alias}': {direct_matches}")
            return [m.lower() for m in direct_matches]

        # If no direct match, use an LLM for smarter resolution
        logging.info(f"[AliasResolver] No direct match for '{alias}'. Using LLM to resolve.")
        prompt = ChatPromptTemplate.from_template(
            "You are an entity resolution expert. Given a user's mention of a name or alias, find all possible full names from the provided list that are a plausible match.\n"
            "Respond with a JSON list of the matching full names. If no matches are found, respond with an empty list.\n\n"
            "User's Mention: '{alias}'\n\n"
            "List of Known Full Names:\n{name_list}\n\n"
            "JSON List of Matches:"
        )
        resolver_chain = prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0) | json.loads
        
        try:
            resolved_names = resolver_chain.invoke({
                "alias": alias,
                "name_list": "\n".join(self.all_known_names)
            })
            logging.info(f"[AliasResolver] LLM resolved '{alias}' to: {resolved_names}")
            # Return the original alias as a fallback if LLM finds nothing
            return [n.lower() for n in resolved_names] if resolved_names else [alias_lower]
        except Exception as e:
            logging.error(f"[AliasResolver] LLM resolution failed: {e}")
            return [alias_lower] # Fallback to the original alias

class DataManager:
    """Loads and holds all data sources in one place for easy access by all agents."""
    def __init__(self):
        logging.info("[DataManager] Initializing and loading all data sources...")
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # --- THIS IS THE CRITICAL FIX ---
        # 1. Load the raw model for our custom code
        self.raw_embedding_model = SentenceTransformer(FINETUNED_MODEL_PATH, device='cpu')
        # 2. Create the LangChain adapter for LangChain components
        self.embedding_model = SentenceTransformerEmbeddings(self.raw_embedding_model)
        # --- NEW: Load a Cross-Encoder for Reranking ---
        logging.info("[DataManager] Loading Cross-Encoder for reranking...")
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        # Load Vector DB
        self.faiss_index = faiss.read_index(FAISS_INDEX_PATH)
        with open(METADATA_PATH, "rb") as f:
            self.metadata_store = pickle.load(f)
        
        # Load and prepare Metadata DataFrame
        self.df = pd.DataFrame(self.metadata_store)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for col in ['date', 'timestamp']:
                if col in self.df.columns:
                    self.df[col] = pd.to_datetime(self.df[col], errors='coerce')

        self.metadata_keys = list(self.metadata_store[0].keys()) if self.metadata_store else []
        # NEW: Initialize the AliasResolver
        self.alias_resolver = AliasResolver(self)
        logging.info(f"[DataManager] Loaded {len(self.metadata_store)} vectors and a DataFrame with {len(self.df)} rows.")
        # --- NEW: Database Connection ---
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self.create_history_table()
        logging.info(f"[DataManager] Connected to SQLite database at {DB_PATH}")
    
    def create_history_table(self):
        """Creates the conversation history table if it doesn't exist."""
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                query TEXT NOT NULL,
                response TEXT,
                feedback INTEGER DEFAULT 0,
                corrected_response TEXT
            )
        """)
        self.conn.commit()

    def log_interaction(self, session_id: str, query: str, response: str) -> int:
        """Logs a new query and its response, returning the record ID."""
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO conversation_history (session_id, query, response) VALUES (?, ?, ?)",
            (session_id, query, response)
        )
        self.conn.commit()
        return cursor.lastrowid

    def update_feedback(self, record_id: int, feedback: int):
        """Updates the feedback for a specific record."""
        cursor = self.conn.cursor()
        cursor.execute(
            "UPDATE conversation_history SET feedback = ? WHERE id = ?",
            (feedback, record_id)
        )
        self.conn.commit()
        logging.info(f"[DataManager] Updated feedback for record {record_id} to {feedback}")

    def get_recent_history(self, session_id: str, limit: int = 5) -> List[dict]:
        """Gets the last few turns of a conversation for short-term memory."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT query, response FROM conversation_history WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?",
            (session_id, limit)
        )
        rows = cursor.fetchall()
        return [{"query": q, "response": r} for q, r in reversed(rows)] # Return in chronological order

# --- SPECIALIZED TOOLS ---

class RAGSearchTool:
    """The advanced RAG tool for qualitative searches on document content."""
    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager

    def _decompose_and_resolve_query(self, user_query: str):
        dm = self.data_manager
        today_str = datetime.now().strftime('%Y-%m-%d')
        system_prompt = f"""
        You are an expert query analyzer. Decompose a user's query into a structured JSON object.
        Today's date is {today_str}.
        
        ## AVAILABLE METADATA FIELDS FOR FILTERING:
        These are the only fields you can use: {', '.join(f"'{key}'" for key in dm.metadata_keys)}.
        The `timestamp` and `date` fields are strings; you must create filters for them.
        **RULES:**
        1. Decompose into `semantic_query` (core topic) and `metadata_filter` (dictionary).
        2. Map names (e.g., "from Raja", "sender KSRT EC") to the `from` or `sender` fields.
        3. Map document types ("in emails", "whatsapp chat") to a `source` filter.
        4. If the query contains a date or a relative time period (e.g., "last month", "this week", "in July 2024"), create a date filter using the `timestamp` or `date` field.
        5. Your response MUST be ONLY the single JSON object.

        **EXAMPLES:**
        User Query: "what is the sentiment of 2g tula customers from jan 2025 to july 2025 in emails?"
        Your JSON: {{"semantic_query": "sentiment of 2g tula customers", "metadata_filter": {{"source": "mail", "date": {{"$gte": "2025-01-01", "$lte": "2025-07-31"}}}}}}

        User Query: "Find the most recent email from sender 'KSRT EC'"
        Your JSON: {{"semantic_query": "most recent email from KSRT EC", "metadata_filter": {{"from": "KSRT EC", "source": "mail"}}}}

        User Query: "Find emails from customer communications this week"
        Your JSON:
        {{
        "semantic_query": "emails from customer communications",
        "metadata_filter": {{
            "from": "customer communications",
            "source": "mail",
            "date": {{ "$gte": "{ (datetime.now() - timedelta(days=datetime.now().weekday())).strftime('%Y-%m-%d') }" }}
        }}
        }}
        User Query: "What is the sentiment of venkata satya in the Houston carpool whatsapp chat last month?"
        Your JSON:
        {{
        "semantic_query": "sentiment analysis of venkata satya's messages",
        "metadata_filter": {{
            "sender": "venkata satya",
            "source": "Houston carpool whatsapp",
            "timestamp": {{
            "$gte": "{ (datetime.now().replace(day=1) - timedelta(days=1)).replace(day=1).strftime('%Y-%m-%d') }",
            "$lte": "{ (datetime.now().replace(day=1) - timedelta(days=1)).strftime('%Y-%m-%d') }"
            }}
        }}
        }}
        """
        try:
            response = dm.client.chat.completions.create(
                model="gpt-4o-mini", messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_query}],
                temperature=0.0, response_format={"type": "json_object"}
            )
            decomposed_plan = json.loads(response.choices[0].message.content)
        except Exception as e:
            decomposed_plan = {"semantic_query": user_query, "metadata_filter": {}}

        # --- NEW: Integrated Alias Resolution Step ---
        metadata_filter = decomposed_plan.get("metadata_filter", {})
        for key in ["from", "sender"]:
            if key in metadata_filter and isinstance(metadata_filter[key], str):
                alias = metadata_filter[key]
                resolved_names = dm.alias_resolver.resolve(alias)
                metadata_filter[key] = resolved_names # Replace string with list of resolved names
        
        decomposed_plan["metadata_filter"] = metadata_filter
        return decomposed_plan

    def _apply_strict_filter(self, filter_dict: dict):
        candidate_indices = []
        for i, metadata in enumerate(self.data_manager.metadata_store):
            if os.path.basename(metadata.get('source', '')) == BRIEFING_DOC_NAME:
                continue

            matches_all_filters = True
            for key, filter_value in filter_dict.items():
                metadata_value = metadata.get(key)
                if metadata_value is None:
                    matches_all_filters = False; break

                metadata_value_lower = str(metadata_value).lower()

                # --- UPGRADED LOGIC: Handle both strings and lists of aliases ---
                if isinstance(filter_value, list):
                    # For aliases, check if metadata matches ANY of the resolved names
                    if not any(alias in metadata_value_lower for alias in filter_value):
                        matches_all_filters = False; break
                elif isinstance(filter_value, dict): # Date filter
                    doc_date = UnifiedDataParser.parse_flexible_date(metadata_value)
                    if not doc_date:
                        matches_all_filters = False; break
                    try:
                        if "$gte" in filter_value and doc_date < datetime.strptime(filter_value["$gte"], '%Y-%m-%d'):
                            matches_all_filters = False; break
                        if "$lte" in filter_value and doc_date.date() > datetime.strptime(filter_value["$lte"], '%Y-%m-%d').date():
                            matches_all_filters = False; break
                    except (ValueError, TypeError):
                        matches_all_filters = False; break
                # NEW, ROBUST TEXT LOGIC
                else:
                    metadata_value_lower = str(metadata_value).lower()
                    # Split the filter value into parts and check if ALL parts are in the metadata
                    query_parts = filter_value.split()
                    if not all(part in metadata_value_lower for part in query_parts):
                        matches_all_filters = False
                        break
            
            if matches_all_filters:
                candidate_indices.append(i)
        return candidate_indices

    def run(self, query: str, search_type: str = "narrow") -> str:
        dm = self.data_manager
        logging.info(f"[RAGSearchTool] Running a '{search_type}' search for: '{query}'")
        
        # This one function now handles both decomposition and resolution
        search_plan = self._decompose_and_resolve_query(query)
        
        semantic_query = search_plan.get("semantic_query", query)
        metadata_filter = search_plan.get("metadata_filter", {})
        
        if search_type == "broad":
            metadata_filter = {}

        logging.info(f"    - Decomposed & Resolved Plan -> Semantic: \"{semantic_query}\" | Filter: {json.dumps(metadata_filter)}")
        
        # --- THIS IS THE CRITICAL FIX ---
        # Use the raw model's .encode() method
        query_embedding = dm.raw_embedding_model.encode([semantic_query])
        query_embedding_np = np.array(query_embedding).astype('float32')
        
        final_indices = []
        top_k = 50

        if not metadata_filter:
            logging.info("    - Strategy: Performing direct semantic search.")
            distances, indices = dm.faiss_index.search(query_embedding_np, top_k)
            final_indices = [i for i in indices[0] if i != -1]
        else:
            logging.info("    - Strategy: Applying filter-then-rank.")
            candidate_indices = self._apply_strict_filter(metadata_filter)
            logging.info(f"    - Found {len(candidate_indices)} candidates after filtering.")
            if not candidate_indices:
                return "No documents were found matching your specific filter criteria (e.g., name, date, or document type)."
            
            candidate_vectors = np.array([dm.faiss_index.reconstruct(i) for i in candidate_indices]).astype('float32')
            if candidate_vectors.size == 0:
                return "Found documents matching filters, but could not retrieve their vector data."
            
            temp_index = faiss.IndexFlatL2(candidate_vectors.shape[1])
            temp_index.add(candidate_vectors)
            distances, temp_indices = temp_index.search(query_embedding_np, k=min(top_k, len(candidate_indices)))
            final_indices = [candidate_indices[i] for i in temp_indices[0] if i != -1]

        if not final_indices:
            return "No information was found that was semantically relevant to your query."
        
        retrieved_metadatas = [dm.metadata_store[i] for i in final_indices]
        
        MIN_WORDS_THRESHOLD = 15
        initial_count = len(retrieved_metadatas)
        filtered_metadatas = [doc for doc in retrieved_metadatas if len(doc.get('original_text', '').split()) > MIN_WORDS_THRESHOLD]
        
        if initial_count > len(filtered_metadatas):
            logging.info(f"    - Post-retrieval: Discarded {initial_count - len(filtered_metadatas)} short/uninformative chunks.")
        
        if not filtered_metadatas:
            return "No detailed information was found. The retrieved documents were too brief to provide a meaningful answer."
        
        # --- NEW: Reranking Step ---
        logging.info(f"    - Reranking {len(filtered_metadatas)} documents for relevance...")
        
        # Create pairs of [query, document_text] for the cross-encoder
        cross_inp = [[semantic_query, doc.get('original_text', '')] for doc in filtered_metadatas]
        cross_scores = dm.cross_encoder.predict(cross_inp)
        
        # Add scores to the documents
        for i in range(len(cross_scores)):
            filtered_metadatas[i]['rerank_score'] = cross_scores[i]

        # Sort documents by their new rerank score in descending order
        reranked_metadatas = sorted(filtered_metadatas, key=lambda x: x['rerank_score'], reverse=True)
        
        # --- CRITICAL FIX: Drastically reduce the final context size ---
        # Only take the TOP 5 most relevant documents after reranking
        final_context_docs = reranked_metadatas[:5]
        
        context = "\n---\n".join([f"Source: {doc.get('source', 'N/A')}\nContent: {doc.get('original_text', '')}" for doc in final_context_docs])
        logging.info(f"    - TOOL: Returning final context from TOP {len(final_context_docs)} reranked documents.")
        return context

class PandasQueryTool:
    """The quantitative analysis tool using a Pandas DataFrame Agent."""
    def __init__(self, data_manager: DataManager, llm):
        self.df = data_manager.df
        df_prefix = f"""
You are a world-class data analyst working with a pandas DataFrame named `df`.
**DataFrame Schema:**
- `source`: The file path. Use to filter by type (e.g., '.pdf', 'whatsapp', '.jsonl').
- `from`: The sender of an **email**.
- `sender`: The sender of a **WhatsApp message**.
- `subject`: The subject line of an email.
- `date`, `timestamp`: The date of the document. Already converted to datetime objects.

**CRITICAL RULES:**
1. A person's name could be in the 'from' column (for emails) or the 'sender' column (for WhatsApp).
2. When filtering for a name (e.g., in 'from' or 'sender'), you MUST use a case-insensitive, partial string match. Correct code: `df[df['column_name'].str.contains('name', case=False, na=False)]`.
3. When asked for the "last" or "latest" item, you must sort by date descending and take the first one. E.g., `.sort_values(by='date', ascending=False).head(1)`.
"""
        self.agent_executor = create_pandas_dataframe_agent(
            llm=llm, df=self.df, agent_type="openai-tools",
            verbose=True, allow_dangerous_code=True, prefix=df_prefix
        )

    def run(self, query: str) -> str:
        logging.info(f"\n[PandasQueryTool] Executing query: '{query}'")
        result = self.agent_executor.invoke({"input": query})
        return result.get('output', "Query executed, but no output was returned.")

# --- WORKER AGENTS ---

# --- Corrected Prompt for Worker Agents ---
WORKER_AGENT_PROMPT_TEMPLATE = """
You are a specialized AI assistant. Your goal is to use your powerful tool to answer the user's question.

TOOLS:
------
You have access to the following tool:
{tools}

To use a tool, please use the following format:```
Thought: The user is asking a question that my tool can answer. I need to use the tool to find the information.
Action: The action to take. Should be one of [{tool_names}]
Action Input: The input to the action
Observation: The result of the action

When you have a response to say to the Human, you MUST use the format:

Thought: I have the answer from my tool.
Final Answer: [your response here]

Begin!
New input: {input}
{agent_scratchpad}
"""

class HistorySearchAgent:
    """NEW: A specialist agent that searches the conversation history database."""
    def __init__(self, data_manager: DataManager, llm):
        self.dm = data_manager
        self.llm = llm
        
    def _build_history_vectorstore(self):
        """Builds a temporary FAISS index from the conversation history."""
        cursor = self.dm.conn.cursor()
        cursor.execute("SELECT id, query, response FROM conversation_history")
        rows = cursor.fetchall()
        if not rows:
            return None
        
        # Create LangChain documents from the history
        history_docs = [
            Document(page_content=f"User asked: {q}\nAgent responded: {r}", metadata={"id": id})
            for id, q, r in rows
        ]
        
        # --- MODIFIED: This now works because embedding_model has the right methods ---
        history_vectorstore = FAISS.from_documents(history_docs, self.dm.embedding_model)
        return history_vectorstore

    def run(self, query: str) -> str:
        logging.info(f"[HistorySearchAgent] Searching conversation history for: '{query}'")
        vectorstore = self._build_history_vectorstore()
        if not vectorstore:
            return "No conversation history has been recorded yet."
        
        # Perform a similarity search on the history
        results = vectorstore.similarity_search(query, k=5)
        
        if not results:
            return "I couldn't find any relevant past conversations."

        # Format the results for the Manager
        context = "\n---\n".join([f"Past Conversation (ID {doc.metadata['id']}):\n{doc.page_content}" for doc in results])
        return f"Found relevant past conversations:\n{context}"

class QualitativeAgent:
    """Worker agent for answering questions from document content."""
    def __init__(self, llm, data_manager):
        self.rag_tool = RAGSearchTool(data_manager)
        tool = Tool(
            name="ContentSearch",
            func=self.rag_tool.run,
            description="Searches document content for qualitative information. Use for 'what is', 'summarize', 'sentiment', etc. Input should be a concise search query."
        )
        prompt = ChatPromptTemplate.from_template(WORKER_AGENT_PROMPT_TEMPLATE)
        agent = create_react_agent(llm, [tool], prompt)
        self.agent_executor = AgentExecutor(agent=agent, tools=[tool], verbose=True, handle_parsing_errors=True)

    def run(self, query: str):
        return self.agent_executor.invoke({"input": query})

class QuantitativeAgent:
    """Worker agent for answering questions from document metadata."""
    def __init__(self, llm, data_manager):
        self.pandas_tool = PandasQueryTool(data_manager, llm)
        tool = Tool(
            name="MetadataAnalysis",
            func=self.pandas_tool.run,
            description="Analyzes document metadata to answer quantitative questions. Use for 'how many', 'list', 'count', 'sort', etc. Input is a natural language question."
        )
        prompt = ChatPromptTemplate.from_template(WORKER_AGENT_PROMPT_TEMPLATE)
        agent = create_react_agent(llm, [tool], prompt)
        self.agent_executor = AgentExecutor(agent=agent, tools=[tool], verbose=True, handle_parsing_errors=True)

    def run(self, query: str):
        return self.agent_executor.invoke({"input": query})

# --- THE MANAGER AGENT ---

class ManagerAgent:
    """The orchestrator that manages the worker agents, now including history search."""
    def __init__(self):
        logging.info("\n[*] Initializing Manager Agent and its team...")
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self.data_manager = DataManager()
        
        # Create the worker agents
        qualitative_agent = QualitativeAgent(self.llm, self.data_manager)
        quantitative_agent = QuantitativeAgent(self.llm, self.data_manager)
        history_agent = HistorySearchAgent(self.data_manager, self.llm)

        # --- THIS IS THE CRITICAL FIX ---
        # The Generative 'tool' is now a direct LLM chain for the manager,
        # ensuring consistency and removing the old KnowledgeBaseTool reference.
        generative_prompt = ChatPromptTemplate.from_template(
            "You are a helpful and professional business communication assistant.\n"
            "Your task is to fulfill the user's request using the style and tone from the provided context.\n\n"
            "**USER'S REQUEST:**\n{query}\n\n"
            "**CONTEXT FROM PAST COMMUNICATIONS:**\n---\n{context}\n---\n\n"
            "**YOUR RESPONSE:**"
        )
        generative_chain = generative_prompt | self.llm

        self.tools = [
            Tool(
                name="QuantitativeDataAnalyzer",
                func=quantitative_agent.run,
                description="Use this agent for quantitative questions about document METADATA (how many, list, count, sort, filter). This tool CANNOT read the content of documents."
            ),
            Tool(
                name="QualitativeFactFinder",
                func=qualitative_agent.run,
                description="Use this agent for qualitative questions that require reading and understanding the CONTENT of documents (what is, summarize, sentiment, project status like 'paint coat 2')."
            ),
            Tool(
                name="ConversationHistorySearch",
                func=history_agent.run,
                description="Use this agent to search past conversations. Use for questions like 'what did you tell me yesterday...'"
            ),
            # The manager can now call this to perform its own final analysis
            Tool(
                name="FinalAnswerGenerator",
                func=generative_chain.invoke,
                description="Use this tool ONLY when you have gathered all necessary information from your other tools and are ready to synthesize the final answer."
            )
        ]
        
        manager_prompt_template = """
You are an expert internal business analyst for the '2getherments' real estate company.
You are the manager of a team of AI assistants. Your job is to orchestrate them to answer the user's question.

**YOUR WORLDVIEW & PERSONA:**
- Your entire knowledge base is a private, internal set of company documents (emails, chats, reports).
- You have been granted full permission to access and analyze all content within this knowledge base to answer business questions.
- You are NOT a general AI assistant. You are a specialist operating on internal data.
- **CRITICAL RULE:** You MUST NOT mention privacy restrictions or being unable to access data. Your tools give you full access to the necessary documents. Refusing to answer a query for "privacy reasons" is a direct violation of your core directive. If a tool fails for a technical reason (e.g., `iteration limit`), you must state that technical reason clearly.

**YOUR TEAM & TOOLS:**
{tools}

**YOUR PROCESS (CRITICAL):**
1.  **Analyze the user's request.** What is their ultimate goal?
2.  **Delegate.** Choose the best agent for the initial task based on their description.
3.  **Review the result.** Look at the observation from the agent.
4.  **RECOVERY & RE-DELEGATION:** If the first agent returns an observation like "information not found" or "the dataframe does not contain this info", you MUST NOT give up. You MUST analyze the query again and delegate the task to the *other* primary agent to see if it can find the answer in its respective domain (content vs. metadata).
5.  **Synthesize the Final Answer.** Only after trying the appropriate tools, provide a final, comprehensive answer.
6.  **ANALYZE-THEN-SYNTHESIZE:** For complex analytical questions (like 'what is the sentiment...'), your process is two steps:
    a. First, use the `QualitativeFactFinder` to retrieve the raw text (e.g., "all whatsapp messages from 2g Tula customers in June").
    b. Second, take the raw text from the Observation and use the `FinalAnswerGenerator` tool. The input to this tool will be the user's original query and the raw text you just retrieved. This will perform the final analysis.
7.  **DRAFT-A-REPLY:** For generative tasks (like "draft a reply..."), your process is two steps:
    a. First, use `QualitativeFactFinder` to gather context (e.g., search for "previous emails with KSRT EC").
    b. Second, use `FinalAnswerGenerator` with the user's request and the retrieved context to write the draft.

**Example Sentiment Analysis:**
User Input: "Give the sentiment of 2g Tula customers in whatsapp from Jan to June 2025."
Thought: The user wants me to analyze sentiment. This is a two-step "Analyze-then-Synthesize" task. First, I need to get the raw chat data.
Action: QualitativeFactFinder
Action Input: "all WhatsApp messages from 2g Tula customers from January 2025 to June 2025"
Observation: [A long string of chat messages is returned]
Thought: I have successfully retrieved the raw chat logs. Now I need to perform the sentiment analysis on this context and provide the final answer. I will use the `FinalAnswerGenerator` for this.
Action: FinalAnswerGenerator
Action Input: {{ "query": "Give the sentiment of 2g Tula customers in whatsapp from Jan to June 2025.", "context": "[The long string of chat messages from the previous observation]" }}
Observation: [The LLM returns a well-written summary of the sentiment]
Thought: I now have the final answer.
Final Answer: [The well-written summary from the previous observation]

**Example Recovery Scenario:**
User Input: "What are all flats left pending for Paint coat 2?"
Thought: The user is asking for a list of flats, which seems like a quantitative task. I will first try the `QuantitativeDataAnalyzer`.
Action: QuantitativeDataAnalyzer
Action Input: "List all flats pending for Paint coat 2"
Observation: The provided DataFrame does not contain any information about flats or their painting status.
Thought: The `QuantitativeDataAnalyzer` failed because the information is not in the metadata. This kind of specific project status is likely inside the *content* of a project report or spreadsheet. I must now use the `QualitativeFactFinder` to search the document content for the same information.
Action: QualitativeFactFinder
Action Input: "flats left pending for Paint coat 2"
Observation: [Context from a progress report PDF is returned]
Thought: I have successfully found the information by using the second tool after the first one failed. I can now provide the final answer.
Final Answer: The flats pending for the second coat of paint are: 815, 205, 604, and 615.

Use the following format:

Thought: Do I need to use a tool? [Yes or No]
Action: The action to take, should be one of [{tool_names}]
Action Input: The input to the action
Observation: The result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now have the final answer
Final Answer: ...

Begin!

**New input:** {input}
{agent_scratchpad}
"""
        prompt = ChatPromptTemplate.from_template(manager_prompt_template)
        agent = create_react_agent(self.llm, self.tools, prompt)
        self.agent_executor = AgentExecutor(
            agent=agent, tools=self.tools, verbose=True, handle_parsing_errors=True, max_iterations=8
        )

    def run(self, user_query: str, session_id: str):
        recent_history = self.data_manager.get_recent_history(session_id)
        chat_history_str = "\n".join([f"User: {h['query']}\nAssistant: {h['response']}" for h in recent_history])
        
        result = self.agent_executor.invoke({
            "input": user_query,
            "chat_history": chat_history_str
        })
        
        # Log the interaction
        response = result.get('output', 'Agent did not return a final answer.')
        record_id = self.data_manager.log_interaction(session_id, user_query, response)
        
        return response, record_id

# --- MAIN SCRIPT EXECUTION ---
if __name__ == "__main__":
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        print("[!] FATAL: OPENAI_API_KEY not found in .env file.")
    else:
        try:
            manager_agent = ManagerAgent()
            session_id = "cli_session" # A fixed session ID for command-line use
            print("\n--- Professional Multi-Agent AI Analyst is Ready ---")
            print("Ask a complex question (or type 'exit' to quit).")
            
            while True:
                user_input = input("> ")
                if user_input.lower() == 'exit': break
                response, record_id = manager_agent.run(user_query=user_input, session_id=session_id)
                # result = manager_agent.run(user_query=user_input)
                
                print("\n" + "="*50 + " FINAL ANSWER " + "="*50)
                # print(result.get('output', 'Manager agent did not return a final answer.'))
                print(response)
                print(f"(Reference ID for feedback: {record_id})")

        except Exception as e:
            print(f"\n[!] A critical error occurred: {e}")
            import traceback
            traceback.print_exc()