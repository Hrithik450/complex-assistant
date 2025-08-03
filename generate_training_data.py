import os
import pickle
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
import json
import random

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
METADATA_PATH = os.path.join(SCRIPT_DIR, "real_estate_finetuned_local_metadata.pkl")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "real_estate_training_dataset.jsonl")
NUM_SAMPLES_TO_GENERATE = 5000 # Increase this for better results if budget allows

# --- INITIALIZATION ---
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- MAIN SCRIPT ---
def generate_synthetic_dataset():
    """
    Loads clean text chunks, generates a question for each, and saves a
    validated (question, passage) dataset for fine-tuning.
    """
    print(f"[*] Loading metadata from '{METADATA_PATH}'...")
    try:
        with open(METADATA_PATH, "rb") as f:
            metadata_store = pickle.load(f)
    except FileNotFoundError:
        print(f"[!] FATAL: Metadata file not found. Please run 'process_data.py' first.")
        return

    print(f"[+] Found {len(metadata_store)} total chunks.")
    
    if len(metadata_store) > NUM_SAMPLES_TO_GENERATE:
        sampled_metadata = random.sample(metadata_store, NUM_SAMPLES_TO_GENERATE)
    else:
        sampled_metadata = metadata_store
        
    system_prompt = """
    You are an expert data analyst. Your task is to generate a single, relevant question that could be answered by the provided text passage.
    The question should be something a user in a real estate company would realistically ask.
    Focus on the specific details in the text. Your entire response must be ONLY the question text. Do not add any conversational text or formatting.
    """
    
    training_data = []
    pbar = tqdm(sampled_metadata, desc="Generating Questions", unit="chunk")
    for metadata in pbar:
        passage = metadata.get('original_text')
        
        # --- THE DEFINITIVE FIX: Validate and convert the passage to a string ---
        if not isinstance(passage, str):
            # If the passage is a list or dict, convert it to a simple string representation
            passage_str = " ".join(str(p) for p in passage) if isinstance(passage, list) else str(passage)
        else:
            passage_str = passage
            
        if not passage_str or len(passage_str.strip()) < 50:
            continue
        # --- END OF FIX ---

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": f"PASSAGE:\n---\n{passage_str}\n---\nQUESTION:"}],
                temperature=0.7
            )
            question = response.choices[0].message.content.strip()
            
            if question:
                training_data.append({"question": question, "passage": passage_str})
        except Exception as e:
            print(f"\n[!] Error generating question for a chunk: {e}")
            continue
            
    if not training_data:
        print("\n[!] No training data was generated.")
        return

    print(f"\n[*] Saving {len(training_data)} validated pairs to '{OUTPUT_FILE}'...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for item in training_data:
            f.write(json.dumps(item) + "\n")
            
    print(f"[+] Dataset generation complete! You can now upload '{os.path.basename(OUTPUT_FILE)}' to Colab.")

if __name__ == "__main__":
    generate_synthetic_dataset()