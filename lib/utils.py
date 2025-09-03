from datetime import datetime
from rapidfuzz import fuzz
import polars as pl
import os
import re

BASE_DIR = os.path.dirname(__file__)  # current file directory
# VECTOR_DATA_PATH = os.path.join(BASE_DIR, "data", "emails_faiss_oaite_2.35G.bin")
CHROMA_COLLECTION_NAME = "my_document_collection" 
EMAIL_JSON_PATH = os.path.join(BASE_DIR, "data", "full_mails.jsonl")
PICKLE_FILE_PATH = os.path.join(BASE_DIR, "data", "optimized_chunks.pkl")
EMBEDDING_MODEL_NAME = "text-embedding-3-large"
AGENT_MODEL = "gpt-4o" # Or another powerful model like "gpt-4-turbo"

# -------------------- SYSTEM PROMPT --------------------
SYSTEM_PROMPT = """
You are a helpful and friendly email assistant. 
Your goal is to assist the user professionally, making the experience pleasant and informative.

Guidelines:
- Always start your response with a polite and friendly tone.
- Use tools when absolutely necessary (e.g., when you cannot answer from the memory context).
- If the answer is already available from the user’s query or conversation context, DO NOT use any tools (Respond using your knowledge).
- When presenting an answer:
    - Be concise, clear, and professional.
    - If the requested information is found: summarize results in a natural tone.
- If the requested information is NOT found:
    - Clearly state that no data matched the search criteria.
    - Mention the search parameters (e.g., sender, date range).
    - Offer a helpful next step (e.g., “Would you like me to expand the date range or check for alternate senders?”).
    - Never leave the user without guidance.
- When the user refers to an email by its position (e.g., "the 1st one", "second email", etc..):
  - Do NOT use the index as the email id.
  - Instead, use the metadata already shown in the context (subject, sender, recipient, cc).
  - If subject and sender are available, pass them to conversation_retriever_tool.
  - Only use id if the actual unique email ID (16-character string) is explicitly available in context.
  - If metadata is missing, inform the user and suggest expanding search criteria.
- If the request involves summarizing, read the full thread and provide a clear, detailed, neutral summary in plain English, focusing on people, topic, and outcome, while ignoring technical details, metadata, and signatures.

Tone:
- Keep it conversational yet professional.
- Avoid sounding robotic; maintain a natural, helpful tone.
- Use phrases like:
    - "Sure, here’s what I found for you:"
    - "No results were found for this search, but we can try adjusting the filters if you’d like."

Examples of No-Data Responses:
"No emails were found from Deepa between Jan 1, 2025 and Sep 1, 2025. Would you like me to extend the date range or check for related senders?"
"I searched for emails about pending works in 2g Tula, but nothing came up in the system. We could try broadening the keywords or looking in a different folder."

Date & Time formatting:
- Always ensure the information is up-to-date.
- Convert the natural user query date expressions into a standard date format expression (like example:- "2024", "january 2024", "yesterday", "last 7 days", "last month", "today").   
Today’s date is {today_date} IST.
"""

# Helper functions
def format_date(d):
    if isinstance(d, datetime):
        return d.strftime('%Y-%m-%d %H:%M:%S')
    elif isinstance(d, str):
        return d
    return 'N/A'

def normalize_email_field(*values):
    """Normalize one or more email fields into clean lowercase emails."""
    normalized_emails = []
    
    for value in values:
        # Polars Series safe check
        if isinstance(value, pl.Series):
            if value.is_empty():
                continue
            value = value.to_list()

        if not value:
            continue

        if isinstance(value, list):
            for v in value:
                cleaned = re.sub(r'[\"\'<>]', '', v)
                normalized_emails.append(cleaned.strip().lower())
        else:
            cleaned = re.sub(r'[\"\'<>]', '', value)
            normalized_emails.append(cleaned.strip().lower())

    return normalized_emails

def match_value_in_columns(value, column_value):
    """
    Check if the global `value` matches any entry in `column_value (from, to, cc)`.

    Matching rules:
      1. If `column_value` is a list → check each item.
      2. If `column_value` is a string → check directly.
      3. A match is considered valid if:
            - `sender` is an exact substring, OR
            - fuzzy string similarity (partial_ratio) > 50.
      4. If no match found or input invalid → return False.
    """
    if not isinstance(value, str) or not value:
        return False

    # Case 1: column_value is a list
    if isinstance(column_value, list):
        for e in column_value:
            if value in e or fuzz.partial_ratio(value.lower(), e.lower()) > 65:
                return True
        return False

    # Case 2: column_value is a string
    if isinstance(column_value, str):
        return value in column_value or fuzz.partial_ratio(value.lower(), column_value.lower()) > 65

    return False

# Normalize the lists to string to apply filters
def normalize_list(lst) -> str:
    normalized = []

    if isinstance(lst, list):
        for i in lst:
            val = normalize_email_field(i)
            if isinstance(val, list):
                normalized.extend(map(str, val))  # flatten if list
            elif val is not None:
                normalized.append(str(val))

    elif lst is not None:
        val = normalize_email_field(lst)
        if isinstance(val, list):
            normalized.extend(map(str, val))
        elif val is not None:
            normalized.append(str(val))

    return ",".join(normalized)

# Helper to safely extract values
def safe_get(row, key, default=""):
    value = row.get(key, default) if isinstance(row, dict) else row[key]
    if value is None or str(value).lower() in {"nan", "none"}:
        return default
    return str(value)