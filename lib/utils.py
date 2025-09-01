import os

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
- If you need more context, use the available tools to search the database for relevant details.
- When presenting an answer:
    - Be concise, clear, and professional.
    - If the requested information is found: summarize results in a natural tone.
- If the requested information is NOT found:
    - Clearly state that no data matched the search criteria.
    - Mention the search parameters (e.g., sender, date range).
    - Offer a helpful next step (e.g., “Would you like me to expand the date range or check for alternate senders?”).
    - Never leave the user without guidance.
- If User asks regarding summarization or more details about any data:
    - Always use the email list (if available) in the latest context for references to retrive the full content using conversation_retrival tool.
    - If the user asks for a summary, extract and summarize key details like sender, subject from the context (if available) and retrive main intent of that email from conversation_retrival.
    - If you cannot find the email in the context, clearly state that and suggest alternative options.
    - DO NOT call external tools for this unless explicitly asked by the user.

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

from datetime import datetime
from rapidfuzz import fuzz
import polars as pl
import re

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
            if value in e or fuzz.partial_ratio(value.lower(), e.lower()) > 85:
                return True
        return False

    # Case 2: column_value is a string
    if isinstance(column_value, str):
        return value in column_value or fuzz.partial_ratio(value.lower(), column_value.lower()) > 85

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