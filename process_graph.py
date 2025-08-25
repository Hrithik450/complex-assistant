import os
import json
import logging
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import pickle
import asyncio
import google.generativeai as genai
from neo4j import GraphDatabase
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError

# --- CONFIGURATION ---
LOG_FILE = "graph_processing.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCHEMA_PATH = os.path.join(SCRIPT_DIR, "final_schema.json")
METADATA_PATH = os.path.join(SCRIPT_DIR, "real_estate_finetuned_local_metadata.pkl")

# --- PERFORMANCE TUNING ---
CONCURRENT_REQUESTS = 20
NEO4J_BATCH_SIZE = 100
EMBEDDING_MODEL = "models/embedding-001"
EXTRACTION_MODEL = "gemini-2.5-flash"
LLM_BATCH_SIZE = 20

class Neo4jConnection:
    """Handles the connection and queries to a Neo4j database."""
    def __init__(self, uri, user, password):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        logging.info("Successfully connected to Neo4j database.")

    def close(self):
        self._driver.close()

    def clear_database(self):
        logging.info("Clearing existing Neo4j database...")
        with self._driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        logging.info("Database cleared successfully.")

    def create_indexes(self, schema):
        """Creates uniqueness constraints and vector indexes."""
        logging.info("Creating uniqueness constraints and vector indexes in Neo4j...")
        with self._driver.session() as session:
            # Uniqueness Constraints
            for label, details in schema.get("node_labels", {}).items():
                if "properties" in details and details["properties"]:
                    primary_key = details["properties"][0]
                    query = f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.{primary_key} IS UNIQUE"
                    session.run(query)
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.source IS UNIQUE")

            # --- NEW: Create Vector Indexes ---
            # We create an index for each node type we want to perform semantic search on.
            vector_nodes = ["Person", "Complaint", "Project", "Task", "Document"]
            for label in vector_nodes:
                index_name = f"{label.lower()}_embedding_index"
                query = f"""
                CREATE VECTOR INDEX {index_name} IF NOT EXISTS
                FOR (n:{label}) ON (n.embedding)
                OPTIONS {{ indexConfig: {{
                    `vector.dimensions`: 768,
                    `vector.similarity_function`: 'cosine'
                }} }}
                """
                session.run(query)
                logging.info(f"Vector index '{index_name}' created or already exists.")
        logging.info("Constraints and indexes are set up successfully.")

class GraphProcessor:
    """Orchestrates the asynchronous extraction and batch loading of graph data."""
    def __init__(self, neo4j_conn, schema, dataframe):
        self.neo4j_conn = neo4j_conn
        self.schema = schema
        self.df = dataframe
        self.primary_keys = {
            label: details["properties"][0]
            for label, details in schema.get("node_labels", {}).items()
            if "properties" in details and details["properties"]
        }
        self.schema_labels = set(self.schema.get("node_labels", {}).keys())
        self.processed_chunks_count = 0
        self.failed_chunks_count = 0

        gemini_api_key = os.getenv("GOOGLE_API_KEY")
        if not gemini_api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set.")
        genai.configure(api_key=gemini_api_key)

        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        self.extraction_model = genai.GenerativeModel(EXTRACTION_MODEL, safety_settings=safety_settings)
        self.results_buffer = []

    # --- NEW: Helper to generate text for embedding ---
    def _get_text_for_embedding(self, node):
        """Creates a descriptive string for a node to be used for embedding."""
        props = node.get("properties", {})
        text_parts = []
        # Prioritize descriptive fields
        if "description" in props: text_parts.append(str(props["description"]))
        if "name" in props: text_parts.append(str(props["name"]))
        if "subject" in props: text_parts.append(str(props["subject"]))
        if "body" in props: text_parts.append(str(props["body"]))
        
        # Add other properties for context
        for key, value in props.items():
            if key not in ["description", "name", "subject", "body"]:
                text_parts.append(f"{key}: {value}")
        
        return ", ".join(text_parts)

    def _get_extraction_prompt(self, batch_chunks):
        return f"""
        You are a data extraction engine. Your task is to process a batch of text chunks. For each chunk, extract entities and relationships according to the schema.

        **Current Schema:**
        {json.dumps(self.schema, indent=2)}

        **Batch of Text Chunks (JSON Array):**
        {json.dumps(batch_chunks, indent=2)}

        **CRITICAL INSTRUCTIONS:**
        1. Your output MUST be a single, valid JSON object with one key: "results".
        2. The "results" value must be an array, with each element corresponding to an input chunk.
        3. Each element in the array MUST have an "id" key that matches the input "id".
        4. Each element should also contain "nodes" and "edges" keys for the entities extracted from its corresponding text. If nothing is extracted, return empty lists.
      
        **EXAMPLE JSON OUTPUT FORMAT:**
        ```json
        {{
          "results": [
            {{
              "id": 0,
              "nodes": [
                {{
                  "type": "Person",
                  "properties": {{"name": "Mr. Sharma"}}
                }}
              ],
              "edges": [
                {{
                  "source": "Mr. Sharma",
                  "target": "Some Complaint Description",
                  "label": "RAISED_COMPLAINT",
                  "source_type": "Person",
                  "target_type": "Complaint"
                }}
              ]
            }}
          ]
        }}
        ```

        **JSON Output:**
        """

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=60))
    async def _process_batch_with_retry(self, batch_of_rows):
        prompt_chunks = [{"id": i, "text": getattr(row, 'original_text', '')} for i, row in enumerate(batch_of_rows)]
        prompt = self._get_extraction_prompt(prompt_chunks) # Your batch prompt is still good
        
        generation_config = genai.types.GenerationConfig(temperature=0.0, response_mime_type="application/json")
        response = await self.extraction_model.generate_content_async(prompt, generation_config=generation_config)
        
        cleaned_text = response.text.strip().replace("```json", "").replace("```", "")
        batch_results = json.loads(cleaned_text)
        
        # --- NEW: Generate embeddings for the extracted nodes ---
        nodes_for_embedding = []
        for result in batch_results.get("results", []):
            for node in result.get("nodes", []):
                nodes_for_embedding.append(node)

        if nodes_for_embedding:
            # Create descriptive texts for each node
            embedding_texts = [self._get_text_for_embedding(node) for node in nodes_for_embedding]
            
            # Get embeddings in a single batch call
            embedding_response = genai.embed_content(model=EMBEDDING_MODEL, content=embedding_texts)
            embeddings = embedding_response['embedding']
            
            # Add the 'embedding' property back to each node dictionary
            for node, embedding in zip(nodes_for_embedding, embeddings):
                node["properties"]["embedding"] = embedding

        return batch_results.get("results", [])
    
    # The _process_batch_wrapper, _batch_load_to_neo4j, log_summary, and run methods
    # from the previous final version can remain. The `_batch_load_to_neo4j` function
    # will automatically handle the new `embedding` property as part of `SET n += properties`.

    async def _process_batch_wrapper(self, batch_of_rows, semaphore):
        async with semaphore:
            try:
                llm_results = await self._process_batch_with_retry(batch_of_rows)
                successful_items = []
                for result in llm_results:
                    try:
                        item_id = result["id"]
                        original_row = batch_of_rows[item_id]
                        graph_data = {"nodes": result.get("nodes", []), "edges": result.get("edges", [])}
                        successful_items.append((graph_data, original_row))
                    except (KeyError, IndexError, TypeError) as e:
                        logging.warning(f"Skipping malformed item in batch response: {e}. Item: {result}")
                        self.failed_chunks_count += 1
                        continue
                self.processed_chunks_count += len(successful_items)
                return successful_items
            except Exception as e:
                logging.error(f"Batch failed after all retries. Error: {e}")
                self.failed_chunks_count += len(batch_of_rows)
                return None

    def _batch_load_to_neo4j(self):
        if not self.results_buffer: return

        documents_to_create = {}
        nodes_to_create = {}
        edges_to_create = []
        mentions_to_create = {}
        label_map = {label.lower(): label for label in self.schema_labels}

        for graph_data, row in self.results_buffer:
            source_id = getattr(row, 'source', None)
            if not source_id: continue

            doc_properties = {"source": source_id, "file_type": getattr(row, 'file_type', None), "sender": getattr(row, 'sender', None)}
            parsed_date = getattr(row, 'parsed_date', None)
            if pd.notna(parsed_date): doc_properties["parsed_date"] = parsed_date.isoformat()
            documents_to_create[source_id] = doc_properties

            if not isinstance(graph_data, dict): continue
            
            for node in graph_data.get("nodes", []):
                raw_label = node.get("type", "").lower()
                if raw_label in label_map:
                    corrected_label = label_map[raw_label]
                    if "properties" in node and isinstance(node["properties"], dict):
                        primary_key = self.primary_keys.get(corrected_label)
                        pk_value = node["properties"].get(primary_key)
                        
                        if isinstance(pk_value, (list, dict)):
                            logging.warning(f"Skipping node due to unhashable primary key '{primary_key}': {pk_value}")
                            continue
                        
                        if pk_value is not None and pk_value != "":
                            if corrected_label not in nodes_to_create: nodes_to_create[corrected_label] = {}
                            nodes_to_create[corrected_label][pk_value] = node["properties"]
                            mention_key = (corrected_label, pk_value)
                            if mention_key not in mentions_to_create: mentions_to_create[mention_key] = set()
                            mentions_to_create[mention_key].add(source_id)
            edges_to_create.extend(graph_data.get("edges", []))

        logging.info(f"Preparing to load batch. Docs: {len(documents_to_create)}, Nodes: {{ {', '.join([f'{k}: {len(v)}' for k, v in nodes_to_create.items()])} }}, Edges: {len(edges_to_create)}")

        with self.neo4j_conn._driver.session() as session:
            tx = None
            try:
                tx = session.begin_transaction()
                if documents_to_create:
                    doc_query = "UNWIND $props as properties MERGE (d:Document {source: properties.source}) SET d += properties"
                    tx.run(doc_query, props=list(documents_to_create.values()))

                for label, props_dict in nodes_to_create.items():
                    primary_key = self.primary_keys.get(label)
                    node_query = f"UNWIND $props as properties MERGE (n:{label} {{ {primary_key}: properties.{primary_key} }}) SET n += properties"
                    tx.run(node_query, props=list(props_dict.values()))
                
                if mentions_to_create:
                    mention_list = [{"label": k[0], "pk_value": k[1], "source": s} for k, v in mentions_to_create.items() for s in v]
                    for label, pk_name in self.primary_keys.items():
                        relevant_mentions = [m for m in mention_list if m['label'] == label]
                        if relevant_mentions:
                            mention_query = f"""
                            UNWIND $mentions as mention
                            MATCH (n:{label} {{ {pk_name}: mention.pk_value }})
                            MATCH (d:Document {{ source: mention.source }})
                            MERGE (n)-[:MENTIONED_IN]->(d)
                            """
                            tx.run(mention_query, mentions=relevant_mentions)
                
                # --- FIX: New, more robust logic for creating relationships between entities ---
                if edges_to_create:
                    grouped_edges = {}
                    for edge in [e for e in edges_to_create if isinstance(e, dict)]:
                        source_type_raw = edge.get("source_type", "").lower()
                        target_type_raw = edge.get("target_type", "").lower()
                        source_id = edge.get("source")
                        target_id = edge.get("target")

                        # Handle the case where the LLM embeds the whole node
                        if not source_type_raw and isinstance(source_id, dict):
                            source_type_raw = source_id.get("type", "").lower()
                            source_label_temp = label_map.get(source_type_raw)
                            if source_label_temp:
                                source_pk = self.primary_keys.get(source_label_temp)
                                source_id = source_id.get("properties", {}).get(source_pk)

                        if not target_type_raw and isinstance(target_id, dict):
                            target_type_raw = target_id.get("type", "").lower()
                            target_label_temp = label_map.get(target_type_raw)
                            if target_label_temp:
                                target_pk = self.primary_keys.get(target_label_temp)
                                target_id = target_id.get("properties", {}).get(target_pk)

                        # Proceed if we have valid types and IDs
                        if source_type_raw in label_map and target_type_raw in label_map and source_id and target_id:
                            source_label = label_map[source_type_raw]
                            target_label = label_map[target_type_raw]
                            rel_label = (edge.get("label") or edge.get("type", "RELATED_TO")).replace(" ", "_").upper()
                            key = (source_label, target_label, rel_label)
                            if key not in grouped_edges: grouped_edges[key] = []
                            grouped_edges[key].append({"source_id": source_id, "target_id": target_id})
                        else:
                            logging.warning(f"Skipping edge due to malformed source/target: {edge}")
                    
                    for (source_label, target_label, rel_label), edges in grouped_edges.items():
                        source_pk = self.primary_keys.get(source_label)
                        target_pk = self.primary_keys.get(target_label)
                        if not source_pk or not target_pk: continue
                        edge_query = (
                            f"UNWIND $edges as edge "
                            f"MATCH (a:{source_label} {{ {source_pk}: edge.source_id }}) "
                            f"MATCH (b:{target_label} {{ {target_pk}: edge.target_id }}) "
                            f"MERGE (a)-[:{rel_label}]->(b)"
                        )
                        result = tx.run(edge_query, edges=edges)
                        summary = result.consume()
                        logging.info(f"Created {summary.counters.relationships_created} '[:{rel_label}]' relationships.")

                tx.commit()
            except Exception as e:
                logging.error(f"FATAL: Error during Neo4j batch loading: {e}", exc_info=True)
                if tx: tx.rollback()
            finally:
                self.results_buffer.clear()

    def log_summary(self):
        total_handled = self.processed_chunks_count + self.failed_chunks_count
        success_rate = (self.processed_chunks_count / total_handled * 100) if total_handled > 0 else 0
        summary = f"\n{'='*80}\n{'📊 PROCESSING SUMMARY 📊'.center(80)}\n{'='*80}\n"
        summary += f"  - Total Chunks in Dataset: {len(self.df)}\n"
        summary += f"  - Successfully Processed Chunks: {self.processed_chunks_count}\n"
        summary += f"  - Failed Chunks: {self.failed_chunks_count}\n"
        summary += f"  - Success Rate: {success_rate:.2f}%\n"
        summary += "="*80
        logging.info(summary)
        print(summary)

    async def run(self):
        semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
        tasks = []
        all_rows = list(self.df.itertuples(index=False))
        batches = [all_rows[i:i + LLM_BATCH_SIZE] for i in range(0, len(all_rows), LLM_BATCH_SIZE)]
        pbar = tqdm(total=len(self.df), desc="Processing Chunks")
        
        for batch in batches:
            tasks.append(self._process_batch_wrapper(batch, semaphore))

        for f in asyncio.as_completed(tasks):
            batch_result = await f
            update_count = LLM_BATCH_SIZE
            if batch_result is not None:
                self.results_buffer.extend(batch_result)
                if len(self.results_buffer) >= NEO4J_BATCH_SIZE:
                    self._batch_load_to_neo4j()
            pbar.update(update_count)
        pbar.close()

        if self.results_buffer: self._batch_load_to_neo4j()

async def main():
    load_dotenv()
    required_vars = ["NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD", "GOOGLE_API_KEY"]
    if any(not os.getenv(var) for var in required_vars):
        logging.error("FATAL: Missing one or more required environment variables.")
        return

    with open(SCHEMA_PATH, 'r') as f: schema = json.load(f)
    with open(METADATA_PATH, "rb") as f: metadata_list = pickle.load(f)
    df = pd.DataFrame(metadata_list)
    if 'parsed_date' in df.columns:
        df['parsed_date'] = pd.to_datetime(df['parsed_date'], errors='coerce')

    neo4j_conn = None
    try:
        neo4j_conn = Neo4jConnection(uri=os.getenv("NEO4J_URI"), user=os.getenv("NEO4J_USERNAME"), password=os.getenv("NEO4J_PASSWORD"))
        # --- MODIFIED CALL ---
        neo4j_conn.create_indexes(schema) 
        
        processor = GraphProcessor(neo4j_conn, schema, df)
        await processor.run()
        processor.log_summary()
    finally:
        if neo4j_conn:
            neo4j_conn.close()
            logging.info("Neo4j connection closed.")

if __name__ == "__main__":
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    loop.run_until_complete(main())