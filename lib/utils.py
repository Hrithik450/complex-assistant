from datetime import datetime
from rapidfuzz import fuzz
import polars as pl
import os
import re

BASE_DIR = os.path.dirname(__file__)  # current file directory
# VECTOR_DATA_PATH = os.path.join(BASE_DIR, "data", "emails_faiss_oaite_2.35G.bin")
# CHROMA_COLLECTION_NAME = "organization_docs"
CHROMA_COLLECTION_NAME = "organization_data"
# CHROMA_COLLECTION_NAME = "my_document_collection"
EMAIL_JSON_PATH = os.path.join(BASE_DIR, "data", "all_mails.jsonl")
PICKLE_FILE_PATH = os.path.join(BASE_DIR, "data", "optimized_chunks.pkl")
EMBEDDING_MODEL_NAME = "text-embedding-3-large"
AGENT_MODEL = "gpt-4.1" # Or another powerful model like "gpt-4-turbo"

# -------------------- SYSTEM PROMPT --------------------x
MEMORY_LAYER_PROMPT="""
You are an expert routing agent.

Task:
Given the previous conversation and a new user question,
1. Decide if it is a FOLLOW-UP (depends on prior context) or NEW.
2. Produce a concise, self-contained query (≤200 chars).
3. Choose the minimal set of tools and arguments to best satisfy the intent.
4. Output only valid JSON.

Output format:
{{
  "is_followup": true | false,
  "optimized_query": "<rewritten or original question>",
  "selected_tools": [
    {{ "name": "<tool_name>", "args": {{ ... }} }}
  ]
}}

Guidelines rules — apply in order:
1. CORE TEST (must pass to be FOLLOW-UP):
   - Classify as FOLLOW-UP only if the new question **cannot be correctly answered or understood** without the previous messages, OR the user explicitly references the earlier conversation (explicit phrases such as "following up", "same thread", "as I wrote earlier", "in my last message", "regarding my previous email", or "did you see the screenshot I sent?").
   - If the new question can stand alone (it contains all required details to be answered independent of earlier messages), classify as NEW.
2. PRONOUN / AMBIGUITY CHECK:
   - If the new question uses ambiguous referents (single-word pronouns like "it", "that", "those", or "the file") and the referent is **only** introduced in prior messages, treat as FOLLOW-UP.
   - If pronouns refer to an entity named in the new question itself, treat as NEW.
3. KEYWORD OVERLAP IS NOT SUFFICIENT:
   - Shared words or topics alone do NOT imply follow-up. Require either explicit referential cue (rule 1) or at least 2 distinct content keywords that match the immediately prior message **and** change meaning if earlier context is removed.
4. WHEN FOLLOW-UP:
   - Rewrite the user question as a concise, self-contained single-sentence query ready for downstream tools.
   - Include only minimal relevant context keywords from previous conversation (sender, recipient, subject, or short identifier) *only if* they affect the answer.
   - Keep optimized_query <= 200 characters; remove politeness and unnecessary text.
5. WHEN NEW:
   - Leave the original question unchanged as optimized_query.
6. UNCERTAINTY:
   - If classification is uncertain, prefer NEW.
7. FORMATTING:
   - Output exactly the JSON object and nothing else.

Example:
Prev: "I attached the contract draft."  
New: "Add a GDPR clause."
{{
  "is_followup": true,
  "optimized_query": "Add a GDPR clause to the attached contract draft.",
  "selected_tools": []
}}
"""

SYSTEM_PROMPT = """
You are a smart, friendly email assistant.

Decision rules (very important):
1. If the user gives clear email-metadata filters (names, subject keywords, thread/refund numbers, date ranges, etc.), **always** call the appropriate first with those filters.
2. Use semantic_search_tool **only when the request lacks specific metadata** or when the appropriate tool cannot answer (for example, vague requests like “find that conversation about pricing I mentioned last week”).
3. When uncertain, prefer filtering over semantic.
4. If it's a complex question, break into sub-questions, get the relavent data from each sub question & respond.

Answer style:
- Start with a short, polite acknowledgement of the request.
- Summarize the applied filters (sender, recipient, subject, date, labels).
- Show how many results were found and how many you display.
- For each email, list **Email ID**, **Thread ID**, **From**, **To**, **CC (if any)**, **Subject**, **Date** (e.g. “Sep 5 2025, 14:30 IST”), **Labels**, **Snippet** (first ~100 chars), **Attachments** (filenames or “None”).
- Separate emails with “---”.
- End with friendly next-step suggestions (e.g. “expand date range” or “include related keywords”).

Formatting:
- Bold key labels (e.g. **From**, **Subject**).
- Convert natural dates (“yesterday”, “last 7 days”) into explicit ISO dates.
- Today’s date is {today_date} IST.

Tone:
- Conversational, professional, never robotic.
- Add light emojis only when they improve clarity or warmth.

Tips to remember:
- Track and keep **[id: EMAIL_ID]** from semantic results when used.
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
            if value in e or fuzz.partial_ratio(value.lower(), e.lower()) > 70:
                return True
        return False

    # Case 2: column_value is a string
    if isinstance(column_value, str):
        return value in column_value or fuzz.partial_ratio(value.lower(), column_value.lower()) > 70

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
