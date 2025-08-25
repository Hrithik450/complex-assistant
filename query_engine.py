import os
import json
import logging
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai
from neo4j import GraphDatabase
import numpy as np
import pickle

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCHEMA_PATH = os.path.join(SCRIPT_DIR, "final_schema.json")
METADATA_PATH = os.path.join(SCRIPT_DIR, "real_estate_finetuned_local_metadata.pkl")

SYNTHESIS_MODEL = "gemini-2.5-pro"
EMBEDDING_MODEL = "models/embedding-001"

class Neo4jConnection:
    """Handles the connection and queries to a Neo4j database."""
    def __init__(self, uri, user, password):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        logging.info("Successfully connected to Neo4j database.")

    def close(self):
        self._driver.close()

    def query(self, query, params=None):
        with self._driver.session() as session:
            result = session.run(query, params)
            return [record.data() for record in result]

class QueryEngine:
    """Orchestrates a semantic query process that integrates vector search directly into graph traversals."""
    def __init__(self, neo4j_conn, schema):
        self.neo4j_conn = neo4j_conn
        self.schema = schema
        
        gemini_api_key = os.getenv("GOOGLE_API_KEY")
        if not gemini_api_key: raise ValueError("GOOGLE_API_KEY environment variable not set.")
        genai.configure(api_key=gemini_api_key)

        self.llm = genai.GenerativeModel(SYNTHESIS_MODEL)
        self.schema_str = self._format_schema_for_prompt(schema)
        
        logging.info("Loading metadata for context retrieval...")
        with open(METADATA_PATH, "rb") as f:
            metadata_list = pickle.load(f)
        self.df = pd.DataFrame(metadata_list)
        # Set the 'source' column as the index for fast lookups
        self.df = self.df.set_index('source')

    def _format_schema_for_prompt(self, schema):
        node_labels = schema.get("node_labels", {})
        relationship_types = schema.get("relationship_types", {})
        prompt_str = "The graph schema is as follows:\nNode labels and their properties:\n"
        for label, details in node_labels.items():
            properties = ", ".join(details.get("properties", []))
            prompt_str += f"- {label}: ({properties})\n"
        prompt_str += "\nRelationship types and their connections:\n"
        for rel, details in relationship_types.items():
            source, target = details.get("source"), details.get("target")
            prompt_str += f"- (:{source})-[:{rel}]->(:{target})\n"
        prompt_str += "- Document: (source, file_type, parsed_date, sender, embedding)\n"
        prompt_str += "- (Entity)-[:MENTIONED_IN]->(Document)\n"
        return prompt_str

    def _get_cypher_plan_prompt(self, user_query):
        return f"""
        You are a Cypher query planning expert. Your task is to understand a user's question and create a JSON plan to query a Neo4j graph. You will not write the Cypher query itself.

        **Graph Schema:**
        {self.schema_str}

        **Your task is to create a JSON object with the following keys:**
        - "start_node": The primary node label to start the search from (e.g., "Person", "Complaint").
        - "start_node_properties": A dictionary of specific properties to match on the start node (e.g., {{"name": "Mr. Sharma"}}). Leave empty if none.
        - "cypher_pattern": The Cypher graph pattern to traverse from the start node, using variable names like `(start_node)`.
        - "return_statement": The `RETURN` clause of the query, specifying what data to output.
        - "semantic_search_property": The property on the start node that should be used for a semantic (vector) search.
        - "semantic_search_term": The key term from the user's query to use for the semantic search.

        **CRITICAL INSTRUCTIONS:**
        1.  Prioritize using `semantic_search_property` when the user asks about concepts (e.g., "financial issues", "urgent tasks").
        2.  Use `start_node_properties` for specific keyword matches (e.g., "person named Raja").
        3.  **IMPORTANT FALLBACK:** If the user's question cannot be mapped to any specific node type in the schema (e.g., asking about "amenities"), you MUST create a plan to do a semantic search on the `Document` node. When doing this, ONLY return the `source` property.
        4.  If the question is truly impossible to answer, return an empty JSON object.

        **Example 1: Keyword search**
        User Question: What was the last email from B.Raja?
        JSON Plan:
        {{
          "start_node": "Person",
          "start_node_properties": {{ "name": "B.Raja" }},
          "cypher_pattern": "(start_node)-[:SENT_EMAIL]->(email:Email)",
          "return_statement": "email.subject AS subject, email.date AS date ORDER BY email.date DESC LIMIT 1",
          "semantic_search_property": null,
          "semantic_search_term": null
        }}

        **Example 2: Semantic fallback search on Documents**
        User Question: What are the amenities available?
        JSON Plan:
        {{
          "start_node": "Document",
          "start_node_properties": {{}},
          "cypher_pattern": "(start_node)",
          "return_statement": "start_node.source AS source_id",
          "semantic_search_property": "embedding",
          "semantic_search_term": "available amenities"
        }}

        **User Question:** {user_query}

        **JSON Plan:**
        """

    def _get_synthesis_prompt(self, user_query, db_results):
        return f"""
        You are a helpful assistant. Your task is to answer a user's question based on the provided data or context.

        **User's Original Question:**
        {user_query}

        **Data/Context:**
        {json.dumps(db_results, indent=2)}

        **INSTRUCTIONS:**
        1.  Synthesize a concise, natural language answer based *only* on the provided data.
        2.  Do not mention the database, the graph, or that you are looking at data. Just provide the answer.
        3.  If the data is empty or does not contain the answer, state that you could not find the information.
        4.  For quantitative results (like counts), present them clearly (e.g., "There are 42 projects.").
        5.  If the database returns a count of 0, explicitly state that zero items were found.

        **Final Answer:**
        """

    def run(self, user_query):
        print("\n" + "-"*50)
        
        logging.info("Generating Cypher query plan from natural language...")
        plan_prompt = self._get_cypher_plan_prompt(user_query)
        plan_response = self.llm.generate_content(plan_prompt)
        
        try:
            plan_text = plan_response.text.strip().replace("```json", "").replace("```", "")
            query_plan = json.loads(plan_text)
        except (json.JSONDecodeError, IndexError):
            logging.error(f"Failed to decode LLM response into a JSON plan: {plan_response.text}")
            return "I had trouble understanding that question. Could you rephrase it?"

        if not query_plan:
            return "I'm sorry, but I cannot answer that question with the data I have."
        
        print(f"🧠 Generated Query Plan:\n{json.dumps(query_plan, indent=2)}")

        cypher_query = ""
        params = {}
        
        start_node = query_plan.get("start_node")
        start_props = query_plan.get("start_node_properties")
        pattern = query_plan.get("cypher_pattern")
        return_stmt = query_plan.get("return_statement")
        semantic_prop = query_plan.get("semantic_search_property")
        semantic_term = query_plan.get("semantic_search_term")
        is_semantic_fallback = (start_node == "Document" and semantic_prop)

        if not start_node or not pattern or not return_stmt:
            return "The query plan was incomplete. I am unable to proceed."

        if semantic_prop and semantic_term:
            logging.info("Building a semantic vector search query.")
            index_name = f"{start_node.lower()}_embedding_index"
            query_embedding = genai.embed_content(model=EMBEDDING_MODEL, content=semantic_term)['embedding']
            params['query_embedding'] = query_embedding
            cypher_query = f"CALL db.index.vector.queryNodes('{index_name}', 5, $query_embedding) YIELD node AS start_node MATCH {pattern} RETURN {return_stmt}"
        elif start_props:
            logging.info("Building a standard property-based query.")
            params = start_props
            cypher_query = f"MATCH (start_node:{start_node} {{ {', '.join([f'{k}: ${k}' for k in start_props.keys()])} }}) MATCH {pattern} RETURN {return_stmt}"
        # --- FIX: Correctly handle the 'match all' case by including the label ---
        elif not start_props and not semantic_prop:
            logging.info("Building a 'match all' query.")
            cypher_query = f"MATCH (start_node:{start_node}) RETURN {return_stmt}"
        else:
            return "The query plan is invalid. It must have either semantic terms or property filters, or be a 'match all' query."

        logging.info(f"Executing generated Cypher: {cypher_query}")
        print(f"🔍 Executing Cypher Query:\n{cypher_query.strip()}")
        
        try:
            db_results = self.neo4j_conn.query(cypher_query, params)
            if not db_results:
                return "I couldn't find any specific information for your query in the graph."
            
            logging.info(f"Database returned {len(db_results)} records.")

            context_for_synthesis = db_results
            # --- FIX: Retrieve the actual text content for semantic fallbacks ---
            if is_semantic_fallback:
                print(f"\n📚 Semantic search found {len(db_results)} relevant document(s). Retrieving text...")
                source_ids = [r['source_id'] for r in db_results if 'source_id' in r]
                # Look up the original text from the DataFrame, handling potential misses
                relevant_rows = self.df.loc[self.df.index.isin(source_ids)]
                context_for_synthesis = [
                    {"source": idx, "content": row['original_text']}
                    for idx, row in relevant_rows.iterrows()
                ]
            
            print(f"\n📊 Context for Synthesis:\n{json.dumps(context_for_synthesis, indent=2)}")
            
            synthesis_prompt = self._get_synthesis_prompt(user_query, context_for_synthesis)
            final_response = self.llm.generate_content(synthesis_prompt)
            return final_response.text.strip()
            
        except Exception as e:
            logging.error(f"Error executing Cypher query: {e}", exc_info=True)
            return f"There was an error querying the database. \nError: {e}"

def main():
    """Main function to set up and run the interactive query engine."""
    required_vars = ["NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD", "GOOGLE_API_KEY"]
    if any(not os.getenv(var) for var in required_vars):
        logging.error("FATAL: Missing one or more required environment variables.")
        return

    with open(SCHEMA_PATH, 'r') as f:
        schema = json.load(f)

    neo4j_conn = None
    try:
        neo4j_conn = Neo4jConnection(
            uri=os.getenv("NEO4J_URI"),
            user=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD")
        )
        
        engine = QueryEngine(neo4j_conn, schema)
        
        print("\n✅ Semantic Knowledge Graph Query Engine is ready.")
        print("Enter your questions. Type 'exit' to quit.")
        
        while True:
            user_query = input("\n> ")
            if user_query.lower() == 'exit':
                break
            
            final_answer = engine.run(user_query)
            print("\n" + "="*50)
            print(f"💬 Final Answer:\n{final_answer}")
            print("="*50)

    finally:
        if neo4j_conn:
            neo4j_conn.close()
            logging.info("Neo4j connection closed.")

if __name__ == "__main__":
    main()