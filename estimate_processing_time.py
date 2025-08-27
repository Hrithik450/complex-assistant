import os
import json
import logging
import asyncio
import aiohttp
import time
from dotenv import load_dotenv

# --- Import the exact same summarization function from your main script ---
# This is crucial to ensure our timing measurements are accurate.
from process_emails import generate_summary

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')

# The number of API calls to make to get a reliable average time.
# 20 is a good balance between speed and accuracy.
SAMPLE_SIZE = 20

def count_total_operations():
    """
    Quickly scans the data directory to count the total number of API calls
    that the full processing script will need to make.
    """
    logging.info("Scanning data directory to calculate total workload...")
    total_jsonl_lines = 0
    total_pdf_files = 0

    for root, _, files in os.walk(DATA_DIR):
        for file in files:
            file_path = os.path.join(root, file)
            if file.lower().endswith('.jsonl'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    total_jsonl_lines += sum(1 for line in f)
            elif file.lower().endswith('.pdf'):
                total_pdf_files += 1
    
    logging.info(f"Found {total_jsonl_lines} emails in JSONL files.")
    logging.info(f"Found {total_pdf_files} PDF files to check.")
    
    # Each JSONL line is one summary call, and each PDF is one check call.
    return total_jsonl_lines + total_pdf_files

async def measure_average_api_time(api_key: str):
    """
    Makes a small number of concurrent API calls to measure the average
    response time, including network latency and retries.
    """
    logging.info(f"Performing {SAMPLE_SIZE} sample API calls to measure average speed...")
    
    # Find a large .jsonl file to use for sample data
    largest_jsonl = None
    max_size = 0
    for root, _, files in os.walk(DATA_DIR):
        for file in files:
            if file.lower().endswith('.jsonl'):
                file_path = os.path.join(root, file)
                size = os.path.getsize(file_path)
                if size > max_size:
                    max_size = size
                    largest_jsonl = file_path

    if not largest_jsonl:
        logging.error("No .jsonl files found to create a sample.")
        return None

    # Get sample email bodies from the largest file
    sample_bodies = []
    with open(largest_jsonl, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= SAMPLE_SIZE:
                break
            try:
                data = json.loads(line)
                body = data.get('body', {}).get('text', '') or data.get('snippet', '')
                if body:
                    sample_bodies.append(body)
            except json.JSONDecodeError:
                continue

    if not sample_bodies:
        logging.error("Could not extract any sample email bodies to test.")
        return None

    start_time = time.time()
    async with aiohttp.ClientSession() as session:
        tasks = [generate_summary(session, api_key, body) for body in sample_bodies]
        await asyncio.gather(*tasks)
    end_time = time.time()

    total_duration = end_time - start_time
    average_time_per_call = total_duration / len(sample_bodies)
    
    logging.info(f"Completed {len(sample_bodies)} sample calls in {total_duration:.2f} seconds.")
    return average_time_per_call

async def main():
    """Main function to orchestrate the estimation process."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logging.error("FATAL: OPENAI_API_KEY not found in .env file.")
        return

    total_ops = count_total_operations()
    if total_ops == 0:
        print("\nNo operations to perform. Your data directory might be empty of relevant files.")
        return

    avg_time = await measure_average_api_time(api_key)
    if avg_time is None:
        return

    total_seconds = total_ops * avg_time
    
    # Convert to human-readable format
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    print("\n" + "="*80)
    print("               PROCESS TIME ESTIMATION REPORT")
    print("="*80)
    print("\n[WORKLOAD ANALYSIS]")
    print(f"  - Total API Calls Required: {total_ops}")
    
    print("\n[PERFORMANCE BENCHMARK]")
    print(f"  - Average Time per API Call: {avg_time:.2f} seconds")
    
    print("\n[ESTIMATED TOTAL RUNTIME]")
    print(f"  - Approximately {int(hours)} hours, {int(minutes)} minutes, and {int(seconds)} seconds.")
    
    print("\n" + "-"*80)
    print("Disclaimer: This is an estimate. Actual time may vary based on network conditions")
    print("and OpenAI API load at the time of execution.")
    print("="*80)

if __name__ == "__main__":
    asyncio.run(main())