import os
import logging
from dotenv import load_dotenv
from neo4j import GraphDatabase
import pandas as pd

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# The number of example relationships to fetch and display.
SAMPLE_SIZE = 25

def format_node(node):
    """Helper function to format a Neo4j Node object for printing."""
    # A node's labels are returned as a frozenset, so we get the first one.
    label = list(node.labels)[0]
    # We'll display the first property as its identifier.
    props = dict(node)
    prop_key = list(props.keys())[0]
    prop_value = props[prop_key]
    return f"{label} {{ {prop_key}: '{prop_value}' }}"

def view_knowledge_graph():
    """
    Connects to the Neo4j database and prints a sample of its nodes and relationships.
    """
    load_dotenv()
    
    # 1. Check for credentials
    required_vars = ["NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD"]
    if any(not os.getenv(var) for var in required_vars):
        logging.error("FATAL: Missing Neo4j credentials in your .env file.")
        return

    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")

    driver = None
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        logging.info("Successfully connected to Neo4j database.")

        with driver.session() as session:
            # --- Main Report ---
            print("\n" + "="*80)
            print("                  KNOWLEDGE GRAPH INSPECTION REPORT")
            print("="*80)

            # 2. Get Overall Statistics
            print("\n## 1. Graph Statistics\n")
            node_count_result = session.run("MATCH (n) RETURN count(n) AS count").single()
            edge_count_result = session.run("MATCH ()-[r]->() RETURN count(r) AS count").single()
            print(f"  - Total Nodes in Graph      : {node_count_result['count']}")
            print(f"  - Total Relationships in Graph: {edge_count_result['count']}")

            # 3. Get a Sample of the Graph Structure
            print(f"\n\n## 2. Sample of Graph Structure (First {SAMPLE_SIZE} relationships found)\n")
            
            # This Cypher query finds any node (n) with any relationship (r) to any other node (m)
            query = f"MATCH (n)-[r]->(m) RETURN n, type(r) as relationship_type, m LIMIT {SAMPLE_SIZE}"
            results = session.run(query)

            # Format results for display in a pandas DataFrame
            graph_sample = []
            for record in results:
                node_n = record["n"]
                rel_type = record["relationship_type"]
                node_m = record["m"]
                
                graph_sample.append({
                    "Source Node": format_node(node_n),
                    "Relationship": f"-[:{rel_type}]->",
                    "Target Node": format_node(node_m)
                })

            if not graph_sample:
                print("  - No relationships found in the graph yet.")
            else:
                df = pd.DataFrame(graph_sample)
                # Use to_string() to ensure all columns and rows are fully displayed
                print(df.to_string())

            print("\n" + "="*80)

    except Exception as e:
        logging.error(f"\n[!] An error occurred while trying to read the graph: {e}")
    finally:
        if driver:
            driver.close()
            logging.info("Database connection closed.")

if __name__ == "__main__":
    view_knowledge_graph()