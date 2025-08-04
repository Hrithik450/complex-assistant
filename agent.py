import os
import pickle
import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import logging
import re
import json
from datetime import datetime, timedelta

# --- CONFIGURATION ---
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
COLLECTION_NAME = "real_estate_finetuned_local"
FAISS_INDEX_PATH = os.path.join(SCRIPT_DIR, f"{COLLECTION_NAME}_faiss.bin")
METADATA_PATH = os.path.join(SCRIPT_DIR, f"{COLLECTION_NAME}_metadata.pkl")
FINETUNED_MODEL_PATH = os.path.join(SCRIPT_DIR, "finetuned_bge_real_estate_model")
BRIEFING_DOC_NAME = "1.Briefing_to_2g_AI_Ally 25Jul2025.docx" # The document to use as a system prompt

# --- KNOWLEDGE BASE TOOL (Unchanged from previous version) ---
class KnowledgeBaseTool:
    """
    A tool that encapsulates an advanced RAG pipeline using a fine-tuned local model.
    It decomposes queries, applies fuzzy metadata filters, and re-ranks results.
    """
    def __init__(self, client: OpenAI):
        print("[*] Initializing Knowledge Base Tool...")
        self.client = client
        if not os.path.isdir(FINETUNED_MODEL_PATH):
            raise FileNotFoundError(f"FATAL: Fine-tuned model not found at '{FINETUNED_MODEL_PATH}'.")
        if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(METADATA_PATH):
            raise FileNotFoundError(f"FATAL: Database files not found. Please run 'process_data.py' first.")

        print(f"[*] Loading fine-tuned model for queries...")
        self.embedding_model = SentenceTransformer(FINETUNED_MODEL_PATH, device='cpu')

        print(f"[*] Loading database from disk...")
        self.faiss_index = faiss.read_index(FAISS_INDEX_PATH)
        with open(METADATA_PATH, "rb") as f:
            self.metadata_store = pickle.load(f)
        
        self.metadata_keys = list(self.metadata_store[0].keys()) if self.metadata_store else []
        print("[+] Knowledge Base Tool is ready.")

    def _parse_flexible_date(self, date_string: str):
        if not isinstance(date_string, str): return None
        formats_to_try = [
            '%m/%d/%y, %H:%M', '%A, %d %B, %Y %I.%M %p', 
            '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d'
        ]
        for fmt in formats_to_try:
            try:
                return datetime.strptime(date_string.strip(), fmt)
            except (ValueError, TypeError):
                continue
        return None

    def _decompose_query_with_llm(self, user_query: str):
        today_str = datetime.now().strftime('%Y-%m-%d')
        system_prompt = f"""
        You are an expert query analyzer. Decompose a user's query into a structured JSON object.
        Today's date is {today_str}.

        ## AVAILABLE METADATA FIELDS FOR FILTERING:
        These are the only fields you can use: {', '.join(f"'{key}'" for key in self.metadata_keys)}.
        The `timestamp` and `date` fields are strings; you must create filters for them.
        
        Decompose into:
        1.  `semantic_query`: The core topic to search for.
        2.  `metadata_filter`: A dictionary of key-value pairs to filter on.

        RULES:
        - Your response MUST be ONLY the single JSON object.
        - If no filters are mentioned, `metadata_filter` must be an empty dictionary {{}}.
        - For emails, the sender is `from`. For WhatsApp, the sender is `sender`.
        - For document names, use the `source` field.
        - If the query contains a date or a relative time period (e.g., "last month", "this week", "in July 2024"), create a date filter using the `timestamp` or `date` field.
        - A date filter should be a dictionary with `"$gte"` (start date) and/or `"$lte"` (end date) in "YYYY-MM-DD" format.

        EXAMPLE:
        User Query: "What is the sentiment of users in the Houston carpool whatsapp chat?"
        Your JSON:
        {{
        "semantic_query": "sentiment analysis",
        "metadata_filter": {{
            "source": "Houston carpool whatsapp"
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

        User Query: "What is the latest email from venkat satya?"
        Your JSON:
        {{
        "semantic_query": "latest email from venkat satya",
        "metadata_filter": {{
            "source": "mail",
            "from": "venkat satya
        }}
        }}
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_query}],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"\n[!] Could not decompose query. Falling back to simple search. Error: {e}")
            return {"semantic_query": user_query, "metadata_filter": {}}

    def _apply_fuzzy_filter(self, filter_dict: dict, max_candidates: int):    
        candidate_scores = []
        # Normalize filter values once
        filter_values = {key: (str(value).lower().split() if key not in ['timestamp', 'date'] else value) for key, value in filter_dict.items()}
        for i, metadata in enumerate(self.metadata_store):
            score = 0
            match_count = 0
            if os.path.basename(metadata.get('source', '')) == BRIEFING_DOC_NAME:
                continue # Explicitly skip the briefing document from search results
            # --- MODIFIED LOGIC FOR MORE FLEXIBLE MATCHING ---
            matches_any_filter = False
            for key, query_parts in filter_values.items():
                metadata_value = metadata.get(key)
                if metadata_value is not None:
                    # Date logic remains the same
                    if key in ['timestamp', 'date'] and isinstance(query_parts, dict):
                        # ... (existing date logic is fine)
                        doc_date = self._parse_flexible_date(metadata_value)
                        if doc_date:
                            is_match = True
                            try:
                                if "$gte" in query_parts and doc_date < datetime.strptime(query_parts["$gte"], '%Y-%m-%d'):
                                    is_match = False
                                if "$lte" in query_parts and doc_date.date() > datetime.strptime(query_parts["$lte"], '%Y-%m-%d').date():
                                    is_match = False
                                if is_match:
                                    score += 1 # Give a point for matching the date filter
                                    matches_any_filter = True
                            except (ValueError, TypeError):
                                continue
                    else:
                        # Text logic is now more flexible
                        metadata_value_lower = str(metadata_value).lower()
                        # If ANY part of the query matches, it's a candidate.
                        # We score based on HOW MANY parts match.
                        num_matches = sum(1 for part in query_parts if part in metadata_value_lower)
                        if num_matches > 0:
                            score += num_matches
                            matches_any_filter = True
            
            if matches_any_filter:
                candidate_scores.append({'index': i, 'score': score})
            # --- END OF MODIFIED LOGIC ---

            # for key, query_parts in filter_values.items():
            #     metadata_value = metadata.get(key)
            #     if metadata_value is not None:
            #         # --- DATE FILTERING LOGIC (Using the robust parser) ---
            #         if key in ['timestamp', 'date'] and isinstance(query_parts, dict):
            #             doc_date = self._parse_flexible_date(metadata_value)
            #             if doc_date:    
            #                 try:
            #                     if "$gte" in query_parts:
            #                         start_date = datetime.strptime(query_parts["$gte"], '%Y-%m-%d')
            #                         if doc_date < start_date:
            #                             match_count = 0
            #                             continue
            #                     if "$lte" in query_parts:
            #                         end_date = datetime.strptime(query_parts["$lte"], '%Y-%m-%d')
            #                         # We only check the date part, ignore time for lte
            #                         if doc_date.date() > end_date.date(): 
            #                             match_count = 0
            #                             continue
            #                 except (ValueError, TypeError):
            #                     continue # Skip if filter date is malformed
            #         else:    
            #             metadata_value_lower = str(metadata_value).lower()
            #             # Check if all parts of the query value are in the metadata value
            #             if any(part in metadata_value_lower for part in query_parts):
            #                 score += sum([1 if part in metadata_value_lower else 0 for part in query_parts]) # Add 1 point for each matching key
            #                 match_count += 1

            # # Only consider documents that matched at least one filter condition
            # if match_count > 0:
            #     # We can add more sophisticated scoring here later if needed
            #     candidate_scores.append({'index': i, 'score': score})
            # # print(metadata['source'])

        # Sort candidates by their score (higher is better)
        candidate_scores.sort(key=lambda x: x['score'], reverse=True)
        
        # Return the indices of the sorted candidates
        print(f"    - inside fuzzy filter and score, Found {len(candidate_scores)} candidates after fuzzy filtering.")
        return [item['index'] for item in candidate_scores[:max_candidates]]

    def search(self, query: str, top_k: int = 100, candidates: int = 250) -> str:
        print(f"\n    - TOOL: Starting hybrid search for: '{query}'")
        search_plan = self._decompose_query_with_llm(query)
        semantic_query = search_plan.get("semantic_query", query)
        metadata_filter = search_plan.get("metadata_filter", {})
        print(f"    - Decomposed Plan -> Semantic: \"{semantic_query}\" | Filter: {json.dumps(metadata_filter)}")
        query_embedding = self.embedding_model.encode([semantic_query])
        query_embedding_np = np.array(query_embedding).astype('float32')
        final_indices = []

        if not metadata_filter:
            print("    - Strategy: No filters found. Performing direct semantic search.")
            distances, indices = self.faiss_index.search(query_embedding_np, top_k)
            # Filter out the briefing document if it appears in pure semantic search
            final_indices = [i for i in indices[0] if i != -1 and os.path.basename(self.metadata_store[i].get('source', '')) != BRIEFING_DOC_NAME]
        else:
            print("    - Strategy: Filters found. Applying filter-then-rank.")
            candidate_indices = self._apply_fuzzy_filter(metadata_filter, max_candidates=candidates)
            print(f"    - Found {len(candidate_indices)} candidates after metadata filtering.")
            if not candidate_indices:
                return "No documents were found matching your filter criteria. Try removing or changing filters like dates or names."
            
            print(f"    - Performing semantic re-ranking on candidates...")
            candidate_vectors = np.array([self.faiss_index.reconstruct(i) for i in candidate_indices]).astype('float32')
            if candidate_vectors.size == 0:
                return "Found documents matching filters, but could not retrieve their vector data for ranking."
            
            temp_index = faiss.IndexFlatL2(candidate_vectors.shape[1])
            temp_index.add(candidate_vectors)
            distances, temp_indices = temp_index.search(query_embedding_np, k=min(top_k, len(candidate_indices)))
            final_indices = [candidate_indices[i] for i in temp_indices[0] if i != -1]

        if not final_indices:
            return "No relevant information was found in the knowledge base for that query."
        
        retrieved_metadatas = [self.metadata_store[i] for i in final_indices]
        context = "\n---\n".join([f"Source: {doc.get('source', 'N/A')}\nContent: {doc.get('original_text', '')}" for doc in retrieved_metadatas])
        print(f"    - TOOL: Returning context from {len(retrieved_metadatas)} documents.")
        return context

# --- AGENT IMPLEMENTATION (Modified) ---
class ReActAgent:
    """A simple ReAct Agent that can reason and use the advanced knowledge base tool."""
    def __init__(self, client: OpenAI, tool: KnowledgeBaseTool, metadata_store: list):
        self.client = client
        self.tool = tool
        
        # --- NEW: Load briefing context and build the dynamic system prompt ---
        briefing_context = self._load_briefing_context(metadata_store)

        # --- MODIFIED PROMPT FOR STRONGER GROUNDING ---
        base_prompt = """
You are an autonomous AI business analyst for a real estate company. Your goal is to provide clear, accurate answers to the user's question by using the tool provided.
You have access to ONE powerful tool: `knowledge_base_search(query: str)`. This tool is intelligent and can understand natural language queries with dates, names, and topics. Provide it with a full, natural language question.

For each step, you must first think about your plan and then decide on an action. Follow this format exactly:

Thought: [Your reasoning and plan for the next step. If the previous step gave you the answer, explain how you will synthesize it. If the tool returned no results, state that the information is not available and prepare to give a final answer.]
Action: [Either a call to `knowledge_base_search(query='Your full, natural language question here')` OR, if you have sufficient information or have confirmed the information is not available, a call to `Final Answer(answer='Your final, synthesized answer, citing sources if possible.')`.]

RULES:
- Your final answer MUST be synthesized *exclusively* from the 'Observation' content provided by the tool. Do not use your own knowledge.
- If the 'Observation' is empty or does not contain the answer, you MUST state that the information could not be found in the documents. DO NOT provide generic explanations.
- The context from the "INTERNAL BRIEFING" above is for your understanding only. Do not mention it in your answer. Use it to better interpret the user's query and the search results.
- If the tool returns a message like "No documents were found", DO NOT try the same query again. Conclude that the information is not in the knowledge base and provide a final answer stating that.
"""
        # --- END OF MODIFIED PROMPT ---
        
#         base_prompt = """
# You are an autonomous AI business analyst for a real estate company. Your goal is to provide clear, accurate answers to the user's question by using the tool provided.
# You have access to ONE powerful tool: `knowledge_base_search(query: str)`. This tool is intelligent and can understand natural language queries with dates, names, and topics. Provide it with a full, natural language question.

# For each step, you must first think about your plan and then decide on an action. Follow this format exactly:

# Thought: [Your reasoning and plan for the next step. If the previous step gave you the answer, explain how you will synthesize it. If the tool returned no results, state that the information is not available and prepare to give a final answer.]
# Action: [Either a call to `knowledge_base_search(query='Your full, natural language question here')` OR, if you have sufficient information or have confirmed the information is not available, a call to `Final Answer(answer='Your final, synthesized answer, citing sources if possible.')`.]

# RULES:
# - ALWAYS use the `knowledge_base_search` tool at least once to answer the user's question. Do not try to answer from memory.
# - The context from the "INTERNAL BRIEFING" above is for your understanding only. Do not mention it in your answer. Use it to better interpret the user's query and the search results.
# -  If the tool returns a message like "No documents were found", DO NOT try the same query again. Conclude that the information is not in the knowledge base and provide a final answer stating that, or include the partially relevant info if something has been retrieved already. This is critical to avoid getting stuck.
# - Synthesize your final answer based *only* on the "Observation" provided by the tool.
# """
        self.system_prompt = f"INTERNAL BRIEFING:\n{briefing_context}\n\n---\n\n{base_prompt}"
        self.history = [("system", self.system_prompt)]
        
    def _load_briefing_context(self, metadata_store: list) -> str:
        """
        Finds and loads the content of the briefing document from the metadata store.
        """
        print(f"[*] Searching for briefing document: '{BRIEFING_DOC_NAME}'...")
        for record in metadata_store:
            # Use os.path.basename to match just the filename
            if os.path.basename(record.get('source', '')) == BRIEFING_DOC_NAME:
                print("[+] Briefing document found and loaded into system prompt.")
                return record.get('original_text', '')
        
        print(f"[!] Warning: Briefing document '{BRIEFING_DOC_NAME}' not found. Agent will proceed without it.")
        return "No briefing was provided."

    def run(self, user_query: str):
        self.history = [("system", self.system_prompt)] # Reset history with the full prompt
        self.history.append(("user", user_query))
        
        for i in range(5):
            print("\n" + "="*50 + f" STEP {i+1} " + "="*50)
            
            prompt_messages = [{"role": role, "content": content} for role, content in self.history]
            response = self.client.chat.completions.create(model="gpt-4o", messages=prompt_messages, temperature=0.0)
            action_text = response.choices[0].message.content
            
            print(action_text) 
            self.history.append(("assistant", action_text))

            if "Final Answer(" in action_text:
                try:
                    match = re.search(r"Final Answer\(answer=(['\"])(.*)\1\)", action_text, re.DOTALL)
                    if match:
                        print("\n" + "="*50 + " FINAL ANSWER " + "="*50)
                        print(match.group(2))
                    else:
                        print("\n[!] Could not parse the final answer.")
                except Exception as e:
                    print(f"\n[!] Error parsing final answer: {e}")
                return
            
            elif "knowledge_base_search(" in action_text:
                try:
                    match = re.search(r"knowledge_base_search\(query=(['\"])(.*)\1\)", action_text, re.DOTALL)
                    if match:
                        query = match.group(2)
                        observation = self.tool.search(query=query)
                        self.history.append(("user", f"Observation: {observation}"))
                    else:
                        self.history.append(("user", "Observation: Could not parse the tool query."))
                except Exception as e:
                    self.history.append(("user", f"Observation: Error executing the tool: {e}"))
            else:
                print("\n[!] Agent generated a response without a valid Action. Stopping.")
                return
        
        print("\n[!] Agent stopped after reaching the maximum number of steps.")

# --- MAIN SCRIPT EXECUTION (Modified) ---
if __name__ == "__main__":
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("[!] FATAL: OPENAI_API_KEY not found in .env file.")
    else:
        try:
            openai_client = OpenAI(api_key=openai_api_key)
            kb_tool = KnowledgeBaseTool(client=openai_client)
            
            # --- NEW: Pass the metadata_store to the agent during initialization ---
            agent = ReActAgent(
                client=openai_client, 
                tool=kb_tool, 
                metadata_store=kb_tool.metadata_store
            )
            
            print("\n--- AI Business Analyst is Ready ---")
            print("Ask a complex question (or type 'exit' to quit).")
            
            while True:
                user_input = input("> ")
                if user_input.lower() == 'exit': break
                agent.run(user_query=user_input)
        except Exception as e:
            print(f"\n[!] A critical error occurred during initialization: {e}")
            import traceback
            traceback.print_exc()