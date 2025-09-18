from typing import Dict, Set, Optional, Tuple
from rapidfuzz import fuzz, process
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import polars as pl
import json
import math
import os
import re

BASE_DIR = os.path.dirname(__file__)  # current file directory
# VECTOR_DATA_PATH = os.path.join(BASE_DIR, "data", "emails_faiss_oaite_2.35G.bin")
# CHROMA_COLLECTION_NAME = "organization_docs"
CHROMA_COLLECTION_NAME = "organization_data"
# CHROMA_COLLECTION_NAME = "my_document_collection"
EMAIL_JSON_PATH = os.path.join(BASE_DIR, "data", "all_mails.jsonl")
TOKEN_MAP_PATH = os.path.join(BASE_DIR, "data", "token_map.jsonl")
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
0. Avoid date's as much as possible while optimizing the query unless user explicitly asks or provides.
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
0. Always prioritize the `optimized_query` and `selected_tools` exactly as provided.
   - Do not drop, add, or override arguments unless the user explicitly asks.
   - Never add default date filters unless explicitly provided.
1. If the user (or optimized_query) provides clear email-metadata filters 
   (threadId, messageId, subject, sender, recipient, labels, etc.), call the filtering tool with those exact filters.
2. - Semantic search results may include appended identifiers in the form `[id: EMAIL_ID]` (id is same as uid).  
   - Always keep track of these IDs.  .  
   - If additional details about that email are needed (e.g., CC list, full metadata), call the email fetching relavant tools with the `uid` to fetch them.
3. If it's a complex question, break into sub-questions, 
   get the relevant data from each, and respond.
4. If the user query is a follow-up or could be influenced by previous conversations, you must incorporate relevant prior messages in your response.
5. If an exact answer cannot be found, clearly state that fact.
   - Instead, present the closest available information, and explain how it might still help the user based on query.

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

def keyword_match(query: str, text: str, min_ratio: float = 0.75) -> bool:
    """
    Returns True if at least `min_ratio` (default 75%) of the keywords
    in `query` appear somewhere in `text`, case-insensitive.

    Example:
        keyword_match("Refund flat 805", "Re: Refund statement for flat 805")
        -> True
    """
    # Extract alphanumeric keywords from the query
    keywords = re.findall(r"[a-zA-Z0-9]+", query.lower())
    if not keywords:
        return False

    # Count how many of those keywords appear in the text
    hits = sum(1 for k in keywords if k in text.lower())

    # Calculate how many matches we need (ceil to avoid rounding down)
    required = math.ceil(len(keywords) * min_ratio)

    return hits >= required

def token_match(query: str, text: str, threshold: float = 0.75) -> bool:
    q_tokens = set(query.lower().split())
    t_tokens = set(text.lower().split())

    overlap = len(q_tokens & t_tokens) / len(q_tokens)
    return overlap >= threshold

def smart_subject_match(user_value: str, column_value: str) -> bool:
    """
    Strong subject matcher:
      1. ≥70% token overlap
      2. ≥75% keyword coverage
    """
    if not isinstance(column_value, str) or not column_value:
        return False

    uv = user_value.lower()
    cv = column_value.lower()

    return token_match(uv, cv, threshold=0.7) or keyword_match(uv, cv, min_ratio=0.75)

def build_name_dict(df: pl.DataFrame) -> pl.DataFrame:
    """
    Vectorized, memory-efficient building of:
        token -> {"full": full_name, "emails": [list of emails]}

    Uses DataFrame.unpivot (replacement for deprecated melt), explode, and Polars string ops.
    """
    cols = [c for c in ["from_normalized", "to_normalized", "cc_normalized"] if c in df.columns]
    if not cols:
        raise ValueError("No normalized columns found. Expect one of: from_normalized, to_normalized, cc_normalized")

    # 1) unpivot (stack the normalized columns into a single column "addr")
    stacked = df.unpivot(index=[], on=cols, variable_name="src", value_name="addr")

    stacked = stacked.filter(pl.col("addr") != "")

    stacked = stacked.with_columns(
        pl.col("addr").str.split(",").alias("addr_list")
    )

    stacked = stacked.explode("addr_list")

    stacked = stacked.with_columns(
        pl.col("addr_list").str.strip_chars().alias("addr")
    ).drop("addr_list")

    stacked = stacked.filter(
        pl.col("addr").is_first_distinct().alias("unique_addr")
    )
    
    return stacked

def normalize_text(s: str) -> str:
    """Normalize a name/email for robust matching."""
    if not s:
        return ""
    # replace separators with spaces, strip non-alphanumerics
    s = re.sub(r"[-_.]+", " ", s.lower())
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def get_best_match_from_token_map(
    sender: str,
    token_map: Dict[str, Set[str]],
    threshold: int = 75
) -> Optional[Tuple[str, float]]:
    """
    Return (best_full_name, score) if match >= threshold (0-100), else None.
    Uses rapidfuzz when present, otherwise difflib fallback.
    """
    if not sender or not token_map:
        return None
    sender_norm = normalize_text(sender)

    sender_lower = sender.lower().strip()
    if sender_lower in (k.lower() for k in token_map.keys()):
        # pick best full_name for that token by comparing normalized forms
        for k in token_map:
            if k.lower() == sender_lower:
                best_full = max(
                    token_map[k],
                    key=lambda f: fuzz.WRatio(sender_norm, normalize_text(f))
                )
                best_score = fuzz.WRatio(sender_norm, normalize_text(best_full))
                if best_score >= threshold:
                    return best_full, best_score
                return None
    
    candidates = []
    meta = {}
    for token_key, full_names in token_map.items():
        tnorm = normalize_text(token_key)
        if tnorm:
            candidates.append(tnorm)
            # token candidate references no specific full name (None)
            meta[tnorm] = (token_key, None)
        for full in full_names:
            fnorm = normalize_text(full)
            if fnorm:
                candidates.append(fnorm)
                meta[fnorm] = (token_key, full)

    # remove duplicates while preserving meta mapping (last wins but that's okay)
    unique_choices = list(dict.fromkeys(candidates))
    match = process.extractOne(
        sender_norm,
        unique_choices,
        scorer=fuzz.WRatio,
        score_cutoff=threshold
    )
    if not match:
        return None
    match_str, score, _ = match  # match_str is normalized candidate
    token_key, full_name = meta.get(match_str, (None, None))
    # If the match candidate was just a token key (full_name is None),
    # pick the best full_name under that token_key
    if full_name is None and token_key is not None:
        best_full = max(
            token_map[token_key],
            key=lambda f: fuzz.WRatio(sender_norm, normalize_text(f))
        )
        best_score = fuzz.WRatio(sender_norm, normalize_text(best_full))
        return (best_full, float(best_score)) if best_score >= threshold else None
    return (full_name, float(score))