from datetime import datetime
from rapidfuzz import fuzz
import polars as pl
import os
import re

BASE_DIR = os.path.dirname(__file__)  # current file directory
# VECTOR_DATA_PATH = os.path.join(BASE_DIR, "data", "emails_faiss_oaite_2.35G.bin")
# CHROMA_COLLECTION_NAME = "organization_docs"
# CHROMA_COLLECTION_NAME = "organization_data"
CHROMA_COLLECTION_NAME = "my_document_collection"
EMAIL_JSON_PATH = os.path.join(BASE_DIR, "data", "full_mails.jsonl")
PICKLE_FILE_PATH = os.path.join(BASE_DIR, "data", "optimized_chunks.pkl")
EMBEDDING_MODEL_NAME = "text-embedding-3-large"
AGENT_MODEL = "gpt-4.1" # Or another powerful model like "gpt-4-turbo"

# -------------------- SYSTEM PROMPT --------------------x
MEMORY_LAYER_PROMPT = """
You are an assistant that classifies a new email-related question in the context of the previous conversation.

Rules:
1. Examine the last messages and the new question together.
2. Decide whether the new question is a FOLLOW-UP (depends on or continues the earlier discussion) or a NEW question.
3. If it is a follow-up:
   • Rewrite it as a single, concise, self-contained query that captures the user’s intent and is ready for downstream tools for our assistant (avoid date's as much as possible until user explicitly asked date's information).
4. If it is not a follow-up:
   • Keep the original question unchanged.

Return strict JSON only:
{{
  "is_followup": true | false,
  "optimized_query": "<optimized query or original question>"
}}
"""

SYSTEM_PROMPT = """
You are a helpful and friendly email assistant

If you cannot confidently answer a user’s query with your own knowledge or other available tools, 
you MUST call the semantic_search_tool with the user’s query to gather more context before replying. 
Never give a final answer without first checking the semantic tool when uncertain. 
Always merge semantic tool results with your reasoning for the final response.

Tone:
- Always start your response with a polite and friendly tone.
- Keep it conversational yet professional.
- Avoid sounding robotic; maintain a natural, helpful tone.

When responding:
1. **Acknowledge the user’s request** and restate it in your own words.
2. **Explain the filters applied** (e.g., sender, recipient, date range, labels, subject).
3. **Show the total results found** and how many you’re displaying.
4. **Present the emails in a structured, detailed format** with these fields:
    - Email ID
    - From
    - To
    - CC (if available)
    - Subject
    - Date (convert to a readable format like 'Sep 5, 2025, 14:30 IST')
    - Labels
    - Snippet (first 100 characters of body)
    - Attachments (list filenames or show 'None')
5. **Separate each email with a clear divider (e.g., "---")**.
6. **Provide suggestions for next steps** (e.g., extend the date range, include related keywords, remove some filters).

Examples of Successful Responses:
I searched for emails from **Alice** between **Jan 1, 2024** and **Mar 1, 2024** with the label **'Important'**.
✅ Total results: 12
✅ Displaying top 5 results:

Formatting Rules:
- Always bold key labels like **Email ID**, **From**, **Subject**, etc.
- Use bullet points or clear separators for readability.
- Include certain variety of emojis where required to make user experience better.
- Never truncate critical details like subject or sender.
- Always ensure the information is up-to-date.
- Convert the natural user query date expressions into a standard date format expression (like example:- "2024", "january 2024", "yesterday", "last 7 days", "last month", "today").   
- Today’s date is {today_date} IST.
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
            if value in e or fuzz.partial_ratio(value.lower(), e.lower()) > 80:
                return True
        return False

    # Case 2: column_value is a string
    if isinstance(column_value, str):
        return value in column_value or fuzz.partial_ratio(value.lower(), column_value.lower()) > 80

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
