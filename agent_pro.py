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
from typing import List, Dict, Any

# --- LangChain Imports ---
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain.chains import LLMChain
# --- NEW IMPORT FOR WEB SEARCH ---
from langchain_community.tools.tavily_search import TavilySearchResults

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
COLLECTION_NAME = "real_estate_finetuned_local"
FAISS_INDEX_PATH = os.path.join(SCRIPT_DIR, f"{COLLECTION_NAME}_faiss.bin")
METADATA_PATH = os.path.join(SCRIPT_DIR, f"{COLLECTION_NAME}_metadata.pkl")
FINETUNED_MODEL_PATH = os.path.join(SCRIPT_DIR, "finetuned_bge_real_estate_model")
BRIEFING_DOC_NAME = "1.Briefing_to_2g_AI_Ally 25Jul2025.docx"
DB_PATH = os.path.join(SCRIPT_DIR, "history.db")


# --- CORE UTILITIES (Unchanged) ---
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_object): self.model = model_object
    def embed_documents(self, texts: List[str]) -> List[List[float]]: return self.model.encode(texts).tolist()
    def embed_query(self, text: str) -> List[float]: return self.model.encode(text).tolist()

class AliasResolver:
    def __init__(self, data_manager):
        self.dm = data_manager
        from_names = self.dm.df['from'].dropna().unique().tolist()
        sender_names = self.dm.df['sender'].dropna().unique().tolist()
        self.all_known_names = list(set(from_names + sender_names))
        logging.info(f"[AliasResolver] Initialized with {len(self.all_known_names)} unique names.")
    def resolve(self, alias: str) -> list[str]:
        alias_lower = alias.lower()
        direct_matches = [name for name in self.all_known_names if alias_lower in name.lower()]
        if direct_matches: return [m.lower() for m in direct_matches]
        logging.info(f"[AliasResolver] No direct match for '{alias}'. Using LLM to resolve.")
        prompt = ChatPromptTemplate.from_template("You are an entity resolution expert...") # Unchanged
        resolver_chain = prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0) | json.loads
        try:
            resolved_names = resolver_chain.invoke({"alias": alias, "name_list": "\n".join(self.all_known_names)})
            return [n.lower() for n in resolved_names] if resolved_names else [alias_lower]
        except Exception as e:
            logging.error(f"[AliasResolver] LLM resolution failed: {e}")
            return [alias_lower]

class DataManager:
    def __init__(self):
        logging.info("[DataManager] Initializing and loading all data sources...")
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.raw_embedding_model = SentenceTransformer(FINETUNED_MODEL_PATH, device='cpu')
        self.embedding_model = SentenceTransformerEmbeddings(self.raw_embedding_model)
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.faiss_index = faiss.read_index(FAISS_INDEX_PATH)
        with open(METADATA_PATH, "rb") as f: self.metadata_store = pickle.load(f)

        self.df = pd.DataFrame(self.metadata_store)

        if 'parsed_date' in self.df.columns:
            self.df['parsed_date'] = pd.to_datetime(self.df['parsed_date'], errors='coerce')

        self.metadata_keys = list(self.metadata_store[0].keys()) if self.metadata_store else []
        self.alias_resolver = AliasResolver(self)
        logging.info(f"[DataManager] Loaded {len(self.metadata_store)} vectors and a DataFrame with {len(self.df)} rows.")
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self.create_history_table()
        logging.info(f"[DataManager] Connected to SQLite database at {DB_PATH}")
    def create_history_table(self):
        cursor = self.conn.cursor()
        cursor.execute("""CREATE TABLE IF NOT EXISTS conversation_history (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, query TEXT NOT NULL, response TEXT, feedback INTEGER DEFAULT 0, corrected_response TEXT)""")
        self.conn.commit()
    def log_interaction(self, session_id: str, query: str, response: str) -> int:
        cursor = self.conn.cursor()
        cursor.execute("INSERT INTO conversation_history (session_id, query, response) VALUES (?, ?, ?)", (session_id, query, response))
        self.conn.commit()
        return cursor.lastrowid
    # --- THIS IS THE FIX ---
    def update_feedback(self, record_id: int, feedback: int):
        """Updates the feedback for a specific record."""
        cursor = self.conn.cursor()
        cursor.execute("UPDATE conversation_history SET feedback = ? WHERE id = ?", (feedback, record_id))
        self.conn.commit()
        logging.info(f"[DataManager] Updated feedback for record {record_id} to {feedback}")
    def get_recent_history(self, session_id: str, limit: int = 5) -> List[Dict]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT query, response FROM conversation_history WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?", (session_id, limit))
        rows = cursor.fetchall()
        return [{"query": q, "response": r} for q, r in reversed(rows)]

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

        ## AVAILABLE METADATA FIELDS FOR FILTERING: {', '.join(f"'{key}'" for key in dm.metadata_keys)}.

        **RULES:**
        1. Decompose into `semantic_query` (core topic) and `metadata_filter` (dictionary).
        2. Map names (e.g., "from Raja") to `from` or `sender` fields.
        3. Map document types ("in emails", "whatsapp chat") to a `file_type` filter.
        4. For any date-related filtering (e.g., "last month", "in July 2024"), you MUST create a filter for the `parsed_date` field.
        5. Your response MUST be ONLY the single JSON object.

        **EXAMPLES:**
        -   User Query: "what is the sentiment of 2g tula customers from jan 2025 to july 2025 in emails?"
            Your JSON: {{"semantic_query": "sentiment of 2g tula customers", "metadata_filter": {{"file_type": "mail", "parsed_date": {{"$gte": "2025-01-01", "$lte": "2025-07-31"}}}}}}
        -   User Query: "what is the sentiment of 2gtula whatsapp chat from jan 2025 to june 2025?"
            Your JSON: {{"semantic_query": "sentiment analysis of 2gtula WhatsApp chat", "metadata_filter": {{"file_type": "2gtula whatsapp", "parsed_date": {{"$gte": "2025-01-01", "$lte": "2025-06-30"}}}}}}
        -   User Query: "Find emails from customer communications this week"
            Your JSON:
                        {{
                        "semantic_query": "emails from customer communications",
                        "metadata_filter": {{
                            "from": "customer communications",
                            "file_type": "email",
                            "parsed_date": {{ "$gte": "{ (datetime.now() - timedelta(days=datetime.now().weekday())).strftime('%Y-%m-%d') }" }}
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
        metadata_filter = decomposed_plan.get("metadata_filter", {})
        for key in ["from", "sender"]:
            if key in metadata_filter and isinstance(metadata_filter[key], str):
                alias = metadata_filter[key]
                resolved_names = dm.alias_resolver.resolve(alias)
                metadata_filter[key] = resolved_names
        decomposed_plan["metadata_filter"] = metadata_filter
        return decomposed_plan

    def _apply_flexible_filter(self, filter_dict: dict, max_candidates: int = 500):
        candidate_scores = []
        filter_values = {key: (str(value).lower().split() if key != 'parsed_date' else value) for key, value in filter_dict.items()}
        # Get a version of the dataframe with just the columns we need for filtering
        df_for_filtering = self.data_manager.df[list(filter_values.keys())].copy()

        for i, row in df_for_filtering.iterrows():
            metadata = row.to_dict()
            if os.path.basename(metadata.get('source', '')) == BRIEFING_DOC_NAME: continue
            
            score = 0
            matches_any_filter = False
            for key, query_value in filter_values.items():
                metadata_value = metadata.get(key)
                if pd.isna(metadata_value): continue

                if key == 'parsed_date' and isinstance(query_value, dict):
                    doc_date = metadata_value # This is already a datetime object
                    if doc_date:
                        is_match = True
                        try:
                            # --- THIS IS THE FIX: Compare date parts directly ---
                            if "$gte" in query_value and doc_date.date() < datetime.strptime(query_value["$gte"], '%Y-%m-%d').date(): is_match = False
                            if "$lte" in query_value and doc_date.date() > datetime.strptime(query_value["$lte"], '%Y-%m-%d').date(): is_match = False
                            if is_match:
                                score += 5
                                matches_any_filter = True
                        except (ValueError, TypeError, AttributeError): continue
                else:
                    metadata_value_lower = str(metadata_value).lower()
                    query_parts = query_value if isinstance(query_value, list) else str(query_value).lower().split()
                    num_matches = sum(1 for part in query_parts if part in metadata_value_lower)
                    if num_matches > 0:
                        score += num_matches
                        matches_any_filter = True
            if matches_any_filter:
                candidate_scores.append({'index': i, 'score': score})
        
        candidate_scores.sort(key=lambda x: x['score'], reverse=True)
        logging.info(f"    - Found {len(candidate_scores)} candidates after flexible filtering.")
        return [item['index'] for item in candidate_scores[:max_candidates]]

    def run(self, query: str, search_type: str = "narrow") -> str:
        dm = self.data_manager
        logging.info(f"[RAGSearchTool] Running a '{search_type}' search for: '{query}'")
        search_plan = self._decompose_and_resolve_query(query)
        semantic_query = search_plan.get("semantic_query", query)
        metadata_filter = search_plan.get("metadata_filter", {})
        if search_type == "broad": metadata_filter = {}
        logging.info(f"    - Decomposed & Resolved Plan -> Semantic: \"{semantic_query}\" | Filter: {json.dumps(metadata_filter)}")
        query_embedding = dm.raw_embedding_model.encode([semantic_query])
        query_embedding_np = np.array(query_embedding).astype('float32')
        final_indices = []
        top_k, rerank_candidates_count = 50, 250
        if not metadata_filter:
            logging.info("    - Strategy: Performing direct semantic search.")
            distances, indices = dm.faiss_index.search(query_embedding_np, top_k)
            final_indices = [i for i in indices[0] if i != -1]
        else:
            logging.info("    - Strategy: Applying flexible filter-then-rank.")
            candidate_indices = self._apply_flexible_filter(metadata_filter, max_candidates=rerank_candidates_count)
            if not candidate_indices: return "No documents were found matching your specific filter criteria (e.g., name, date, or document type)."
            candidate_vectors = np.array([dm.faiss_index.reconstruct(i) for i in candidate_indices]).astype('float32')
            if candidate_vectors.size == 0: return "Found documents matching filters, but could not retrieve their vector data."
            temp_index = faiss.IndexFlatL2(candidate_vectors.shape[1])
            temp_index.add(candidate_vectors)
            distances, temp_indices = temp_index.search(query_embedding_np, k=min(top_k, len(candidate_indices)))
            final_indices = [candidate_indices[i] for i in temp_indices[0] if i != -1]
        if not final_indices: return "No information was found that was semantically relevant to your query."
        retrieved_metadatas = [dm.metadata_store[i] for i in final_indices]
        MIN_WORDS_THRESHOLD = 15
        filtered_metadatas = [doc for doc in retrieved_metadatas if len(doc.get('original_text', '').split()) > MIN_WORDS_THRESHOLD]
        if not filtered_metadatas: return "No detailed information was found. The retrieved documents were too brief to provide a meaningful answer."
        logging.info(f"    - Reranking {len(filtered_metadatas)} documents for relevance...")
        cross_inp = [[semantic_query, doc.get('original_text', '')] for doc in filtered_metadatas]
        cross_scores = dm.cross_encoder.predict(cross_inp)
        for i in range(len(cross_scores)): filtered_metadatas[i]['rerank_score'] = cross_scores[i]
        reranked_metadatas = sorted(filtered_metadatas, key=lambda x: x['rerank_score'], reverse=True)
        final_context_docs = reranked_metadatas[:15]
        context = "\n---\n".join([f"Source: {doc.get('source', 'N/A')}\nContent: {doc.get('original_text', '')}" for doc in final_context_docs])
        logging.info(f"    - TOOL: Returning final context from TOP {len(final_context_docs)} reranked documents.")
        return context

class PandasQueryTool:
    """The quantitative analysis tool using a Pandas DataFrame Agent."""
    def __init__(self, data_manager: DataManager, llm):
        self.df = data_manager.df
        df_prefix = f"""
You are a world-class data analyst working with a pandas DataFrame named `df`.

**CRITICAL RULES FOR QUERYING:**

1.  **USE `parsed_date` FOR ALL DATE OPERATIONS**: The DataFrame has a standardized column named `parsed_date` which is already a datetime object. For any questions involving dates (e.g., "in September 2024", "last month", "latest"), you MUST use this column.
    -   Correct Example: `df[df['parsed_date'].dt.month == 9]`
    -   **DO NOT USE** any other date-like columns.

2.  **IDENTIFY FILE TYPES**: Use the `file_type` column to filter for specific document types.
    -   `df[df['file_type'] == 'email']`
    -   `df[df['file_type'] == 'whatsapp']`

3.  **COUNT UNIQUE ITEMS**: When asked to count unique things like email threads, use `nunique()` on the appropriate ID column.
    -   Correct Example for unique threads: `df[df['file_type'] == 'email']['threadId'].nunique()`

4.  **FILTER BY NAME**: When filtering by a person's name (in `from` or `sender`), always use a case-insensitive, partial string match.
    -   Correct Example: `df[df['from'].str.contains('Sankar', case=False, na=False)]`

**DataFrame Schema:**
- `source`:  The original file path.
- `file_type`: The type of document ('email', 'whatsapp', 'pdf', 'docx', etc.).
- `from`, `to`, `cc`, `subject`: Email fields.
- `sender`: WhatsApp message sender.
- `id`, `threadId`: Unique IDs for emails.
- `parsed_date`: The standardized datetime object for all records. **USE THIS FOR DATES.**

"""
        self.agent_executor = create_pandas_dataframe_agent(
            llm=llm, df=self.df, agent_type="openai-tools",
            verbose=True, allow_dangerous_code=True, prefix=df_prefix
        )
    def run(self, query: str) -> str:
        logging.info(f"\n[PandasQueryTool] Executing query: '{query}'")
        result = self.agent_executor.invoke({"input": query})
        return result.get('output', "Query executed, but no output was returned.")

class HistorySearchAgent:
    """Specialist agent that searches the conversation history database."""
    def __init__(self, data_manager: DataManager, llm):
        self.dm = data_manager
        self.llm = llm
    def _build_history_vectorstore(self):
        cursor = self.dm.conn.cursor()
        cursor.execute("SELECT id, query, response FROM conversation_history")
        rows = cursor.fetchall()
        if not rows: return None
        history_docs = [Document(page_content=f"User asked: {q}\nAgent responded: {r}", metadata={"id": id}) for id, q, r in rows]
        return FAISS.from_documents(history_docs, self.dm.embedding_model)
    def run(self, query: str) -> str:
        logging.info(f"[HistorySearchAgent] Searching conversation history for: '{query}'")
        vectorstore = self._build_history_vectorstore()
        if not vectorstore: return "No conversation history has been recorded yet."
        results = vectorstore.similarity_search(query, k=5)
        if not results: return "I couldn't find any relevant past conversations."
        context = "\n---\n".join([f"Past Conversation (ID {doc.metadata['id']}):\n{doc.page_content}" for doc in results])
        return f"Found relevant past conversations:\n{context}"

# --- THE MANAGER AGENT ---
class ManagerAgent:
    """The orchestrator that uses tools directly to answer questions."""
    def __init__(self):
        logging.info("\n[*] Initializing Manager Agent and its tools...")
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self.data_manager = DataManager()
        rag_tool = RAGSearchTool(self.data_manager)
        pandas_tool = PandasQueryTool(self.data_manager, self.llm)
        history_agent = HistorySearchAgent(self.data_manager, self.llm)

        def _load_briefing_context() -> str:
            logging.info(f"[ManagerAgent] Searching for briefing document: '{BRIEFING_DOC_NAME}'...")
            for record in self.data_manager.metadata_store:
                if os.path.basename(record.get('source', '')) == BRIEFING_DOC_NAME:
                    logging.info("[ManagerAgent] Briefing document found and loaded into Core Knowledge.")
                    return record.get('original_text', 'No content found.')
            logging.warning(f"[ManagerAgent] Briefing document '{BRIEFING_DOC_NAME}' not found.")
            return "No briefing was provided."
        briefing_context = _load_briefing_context()

        generative_prompt = ChatPromptTemplate.from_template(
            "You are a helpful and professional business communication assistant.\n"
            "Your task is to fulfill the user's request using the style and tone from the provided context.\n\n"
            "**USER'S REQUEST:**\n{query}\n\n"
            "**CONTEXT FROM PAST COMMUNICATIONS:**\n---\n{context}\n---\n\n"
            "**YOUR RESPONSE:**"
        )
        generative_chain = generative_prompt | self.llm

        # --- THIS IS THE FIX: A wrapper function for the generative tool ---
        def run_generative_chain_with_parsing(tool_input: Any) -> str:
            """
            A wrapper to parse the tool input, which might be a stringified JSON,
            before passing it to the actual chain.
            """
            logging.info(f"[GeneratorWrapper] Received input of type {type(tool_input)}")
            parsed_input = {}
            if isinstance(tool_input, str):
                try:
                    # The agent often wraps its dictionary output in a string.
                    parsed_input = json.loads(tool_input)
                except json.JSONDecodeError:
                    return "Error: The input for FinalAnswerGenerator was not valid JSON. Please provide a valid JSON object with 'query' and 'context' keys."
            elif isinstance(tool_input, dict):
                parsed_input = tool_input
            else:
                return f"Error: Invalid input type for FinalAnswerGenerator. Expected a dictionary or a JSON string, but got {type(tool_input)}."

            if "query" not in parsed_input or "context" not in parsed_input:
                return "Error: The input for FinalAnswerGenerator must be a dictionary with 'query' and 'context' keys."

            # Now call the actual chain with the correctly formatted dictionary
            return generative_chain.invoke(parsed_input)
        
        # --- SOLUTION 2: Initialize the web search tool ---
        web_search_tool = TavilySearchResults(max_results=3)

        self.tools = [
            Tool(name="QuantitativeDataAnalyzer", func=pandas_tool.run, description="Use for quantitative questions about METADATA (how many, list, count, sort). CANNOT read document content."),
            Tool(name="QualitativeFactFinder", func=rag_tool.run, description="Use for qualitative questions that require reading document CONTENT (what is, summarize, sentiment)."),
            Tool(name="ConversationHistorySearch", func=history_agent.run, description="Use to search past conversations (e.g., 'what did you tell me yesterday...')."),
            # --- SOLUTION 2: Add the new tool to the list ---
            Tool(name="WebSearch", func=web_search_tool.invoke, description="Use this as a LAST RESORT if the information is not found in internal documents. Good for public information about companies or people."),
            # The tool now uses our robust wrapper function
            Tool(name="FinalAnswerGenerator", func=run_generative_chain_with_parsing, description="Use ONLY when you have gathered all necessary information and are ready to synthesize the final answer.")
        ]
        
        manager_prompt_template = """
You are a master AI assistant for the '2getherments' real estate company. Your job is to use the tools at your disposal to answer the user's question.

**CORE KNOWLEDGE & INTERNAL BRIEFING (Your Brain):**
---
{briefing_context}
---

**CONVERSATION HISTORY:**
---
{chat_history}
---


**YOUR WORLDVIEW & PERSONA:**
- Your knowledge base is a private, internal set of company documents. Always search these first.
- You have full permission to access and analyze all content to answer business questions.
- You are a specialist operating on internal data, NOT a general AI.
- **CRITICAL RULE:** You MUST NOT mention privacy restrictions or being unable to access data. Your tools give you full access. Refusing to answer for "privacy reasons" is a direct violation of your core directive.

**YOUR TOOLS:**
{tools}

**YOUR PROCESS (CRITICAL):**
1.  **Analyze the user's request.** Pay close attention to the **CONVERSATION HISTORY**. If the new input is a follow-up question (e.g., "what about...", "and for that project?"), you MUST use the context from the history to understand the full query.
2.  **Internal Search First:** Always start by using `QualitativeFactFinder` or `QuantitativeDataAnalyzer` to search the internal documents.
3.  **Review:** Look at the observation from the tool.
4.  **RECOVERY & RETRY:** If one internal tool fails (e.g., `QualitativeFactFinder` finds nothing), you MUST try the *other* internal tool (`QuantitativeDataAnalyzer`) if it's relevant.
5.  **EXTERNAL SEARCH (LAST RESORT):** If, and only if, both internal search tools fail to find a relevant answer, use the `WebSearch` tool to look for public information.
6.  **DELIVER THE FINAL ANSWER:** After you have gathered all the information you need (from internal tools or the web), you MUST conclude your work. To do this, you MUST use the `Final Answer:` format. Do not simply state the answer in plain text. Your final turn must be structured like this:
    Thought: I have all the information required to answer the user's question. I will now provide the final answer.
    Final Answer: [The complete, synthesized answer for the user.]
7.  **CONCLUDING WHEN INFORMATION IS NOT FOUND:** If you have used all relevant tools and still cannot find the answer, you MUST conclude by using the `Final Answer:` format to inform the user that the information is not available.
    Thought: I have exhausted all my tools and cannot find the requested information. I will now inform the user.
    Final Answer: I could not find any information regarding [the user's query] in the available documents or through a web search.

**Example of using CONVERSATION HISTORY:**
User: "how many emails did we receive in august 2024?"
Assistant: "458 emails were received in August 2024."
New User Input: "what about december 2023?"
Thought: The user is asking a follow-up question. Based on the history, their question means "how many emails did we receive in december 2023?". This is a quantitative question. I will use the QuantitativeDataAnalyzer.
Action: QuantitativeDataAnalyzer
Action Input: "number of emails received in december 2023"

**Example of Web Search Fallback:**
User Input: "Who are our board of directors?"
Thought: The user is asking for company leadership information. I will first search my internal documents.
Action: QualitativeFactFinder
Action Input: "board of directors of 2getherments"
Observation: [The tool returns documents about a Flat Owners Association, not the company's board.]
Thought: The internal documents do not contain information about the company's board of directors. This is public information, so I should now use the WebSearch tool as a last resort.
Action: WebSearch
Action Input: "2getherments real estate board of directors"
Observation: [Web search results are returned with the names of the directors.]
Thought: I have found the answer using the web search. I can now provide the final answer.
Final Answer: The board of directors for 2getherments are [names from web search].

**EXAMPLE OF CORRECT FORMATTING FOR FinalAnswerGenerator:**
Thought: I have the raw text. Now I will call the FinalAnswerGenerator to create the final answer.
Action: FinalAnswerGenerator
Action Input: {{"query": "Give the sentiment of 2g Tula customers in whatsapp from Jan to June 2025.", "context": "Source: ... Content: ... --- Source: ... Content: ... "}}
Observation: [The LLM returns a well-written summary of the sentiment]
Thought: I now have the final answer.
Final Answer: [The well-written summary from the previous observation]

Use the following format:
Thought: ...
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
        prompt = ChatPromptTemplate.from_template(manager_prompt_template).partial(briefing_context=briefing_context)
        agent = create_react_agent(self.llm, self.tools, prompt)
        # --- SOLUTION 1: Add robust error handling to the executor ---
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            # This provides a specific instruction to the agent if it messes up the format.
            handle_parsing_errors="Check your output and make sure it conforms to the required format: `Thought: ...\nAction: ...\nAction Input: ...` or `Thought: ...\nFinal Answer: ...`",
            max_iterations=7
        )

    def run(self, user_query: str, session_id: str):
        recent_history = self.data_manager.get_recent_history(session_id)
        chat_history_str = "\n".join([f"User: {h['query']}\nAssistant: {h['response']}" for h in recent_history])
        result = self.agent_executor.invoke({"input": user_query, "chat_history": chat_history_str})
        response = result.get('output', 'Agent did not return a final answer.')
        record_id = self.data_manager.log_interaction(session_id, user_query, response)
        return response, record_id

# --- MAIN SCRIPT EXECUTION ---
if __name__ == "__main__":
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("TAVILY_API_KEY"):
        print("[!] FATAL: OPENAI_API_KEY or TAVILY_API_KEY not found in .env file.")
    else:
        try:
            manager_agent = ManagerAgent()
            session_id = "cli_session"
            print("\n--- Professional AI Analyst is Ready ---")
            print("Ask a complex question (or type 'exit' to quit).")
            while True:
                user_input = input("> ")
                if user_input.lower() == 'exit': break
                response, record_id = manager_agent.run(user_query=user_input, session_id=session_id)
                print("\n" + "="*50 + " FINAL ANSWER " + "="*50)
                print(response)
                print(f"(Reference ID for feedback: {record_id})")
        except Exception as e:
            print(f"\n[!] A critical error occurred: {e}")
            import traceback
            traceback.print_exc()