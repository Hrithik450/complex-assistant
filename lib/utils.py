from typing import Dict, Set, Optional, Tuple
from datetime import datetime, timezone
from rapidfuzz import fuzz, process
from typing import Tuple
import polars as pl
import json
import os
import re

BASE_DIR = os.path.dirname(__file__)  # current file directory
# VECTOR_DATA_PATH = os.path.join(BASE_DIR, "data", "emails_faiss_oaite_2.35G.bin")
# CHROMA_COLLECTION_NAME = "organization_docs"
CHROMA_COLLECTION_NAME = "organization_data"
# CHROMA_COLLECTION_NAME = "my_document_collection"
EMAIL_JSON_PATH = os.path.join(BASE_DIR, "data", "clean_mails.jsonl")
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
4. Use semantic search only if the query is vague or no metadata (0 fields), avoid when metadata (even 1 field is present) explicitly provided.
5. Output only valid JSON.

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
   - Classify as FOLLOW-UP only if the new question **cannot be correctly answered or understood** without the previous messages, OR the user explicitly references the earlier conversation.
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
   - Rewrite the original question into a concise, self-contained query (≤200 chars) suitable for downstream tools.
   - Include all relevant context keywords from query (sender, recipient, subject, or short identifier) *only if* they affect the answer and can be retrived via any available tool.
6. FORMATTING:
   - Output exactly the JSON object and nothing else.
7. LIMIT HANDLING
   - Use limit=N only if the query explicitly requests a fixed number (e.g., “latest”, “last 5”), elif: And set default value to 5, else: for summaries or entire email chains do not use limit parameter.
"""

SYSTEM_PROMPT = """
You are a smart, friendly email assistant.

Decision rules (very important):
0. Always prioritize the `optimized_query` and `selected_tools` exactly as provided.
   - Do not drop, add, or override arguments unless the user explicitly asks.
   - Never add default date filters unless explicitly provided.
1. If the user (or optimized_query) provides clear email-metadata filters 
   (threadId, messageId, subject, sender, recipient, labels, etc.), call the filtering tool with those exact filters.
2. Use semantic_search_tool when relevant (e.g., vague queries or for context).
   - Track any email identifiers `[id: EMAIL_ID]` in the results.  
   - Fetch full email details with the appropriate tool using these IDs if needed.
3. If it's a complex question, break into sub-questions, 
   get the relevant data from each, and respond.
4. If the user query is a follow-up or could be influenced by previous conversations, you must incorporate relevant prior messages in your response.
5. If an exact answer cannot be found, clearly state that fact.
   - Instead, present the closest available information, and explain how it might still help the user based on query.
6. If the query provides metadata filters and the filtering tool returns no results, guide the user to cross-check the fields they provided, especially the subject (if present in query), to ensure they are correct.

Answer style guidelines:
1. Start every response with a short, polite acknowledgement of the request.
2. When handling emails (listing, filtering, or summarizing):
   - Always include both "id" and "threadId" fields explicitly in the output, if they are available.
   - These fields must never be omitted, skipped, or hidden.
3. For analytical, summary-based, or general questions, provide a broad and detailed summarized answer first, covering all relevant aspects.
4. Keep tracking of "id" and "threadId" for any follow-up questions.
5. Always end with a friendly next-step suggestion.

Formatting:
- Bold key labels (e.g. **id**, **ThreadId**, **From**, **Subject**).
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

def preprocess_subject(subject: str) -> str:
    if not isinstance(subject, str):
        return ""
    # Lowercase and replace symbols with space
    subject = re.sub(r'[:\-_,]', ' ', subject)
    subject = re.sub(r'\s+', ' ', subject)  # normalize spaces
    return subject.lower().strip()

def extract_numbers(text: str) -> set[str]:
    return set(re.findall(r'\b\d+\b', text))

def smart_subject_match(user_value: str, column_value: str) -> bool:
    if not column_value:
        return False
    
    user_clean = preprocess_subject(user_value)
    col_clean = preprocess_subject(column_value)

    user_nums = extract_numbers(user_clean)
    col_nums = extract_numbers(col_clean)

    # --- Number must match if present ---
    if user_nums and not (user_nums & col_nums):
        return False

    # --- Fuzzy match on remaining text ---
    fuzz_score = fuzz.token_set_ratio(user_clean, col_clean) / 100

    if user_nums:
        # numbers match → relax threshold
        return fuzz_score >= 0.65
    else:
        # no numbers → require stricter match
        return fuzz_score >= 0.85

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

def clean_date_in_jsonl():
    input_file = "lib/data/all_mails.jsonl"
    output_file = "lib/data/clean_mails.jsonl"

    def clean_date_str(date_str: str) -> str:
        """Clean ISO8601 timestamp string and convert to UTC."""
        if not date_str:
            return None

        date_str = date_str.strip()

        # Remove trailing Z if offset exists
        if ("+" in date_str or "-" in date_str) and date_str.endswith("Z"):
            date_str = date_str[:-1]

        # Remove double +00:00 after offset
        if date_str[-6:] == "+00:00" and ("+" in date_str[:-6] or "-" in date_str[:-6]):
            date_str = date_str[:-6]

        # Replace bare Z with +00:00
        if date_str.endswith("Z"):
            date_str = date_str[:-1] + "+00:00"

        # Parse datetime with offset
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S%z")
        except ValueError:
            return date_str  # keep original if parsing fails

        # Convert to UTC and format as ISO string
        dt_utc = dt.astimezone(timezone.utc)
        return dt_utc.strftime("%Y-%m-%dT%H:%M:%S%z")

    # Process JSONL
    cleaned_data = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            record["date"] = clean_date_str(record.get("date"))
            cleaned_data.append(record)

    # Save cleaned JSONL
    with open(output_file, "w", encoding="utf-8") as f:
        for record in cleaned_data:
            f.write(json.dumps(record) + "\n")

# Date formatting tools
from datetime import datetime, timezone, timedelta

def parse_datetime_utc_flexible(date_str: str) -> datetime:
    """Parse various date/time formats into a UTC-aware datetime."""
    try:
        dt = datetime.fromisoformat(date_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt
    except ValueError:
        pass
    raise ValueError(f"Cannot parse date: {date_str}")

def expand_start(dt: datetime, original_str: str) -> datetime:
    """Expand start bound depending on precision."""
    if len(original_str) == 10:       # YYYY-MM-DD
        return dt.replace(hour=0, minute=0, second=0)
    else:                             # exact second
        return dt

def expand_end(dt: datetime, original_str: str, start_date: str) -> datetime:
    """Expand end bound depending on precision."""
    if len(original_str) == 10:       # YYYY-MM-DD
        return dt.replace(hour=23, minute=59, second=59)
    elif len(original_str) == 16:
        return dt.replace(second=59)
    elif len(original_str) == 19:
        if original_str == start_date:
            if original_str.endswith("00:00"):
                return dt.replace(minute=59, second=59)
            else:
                return dt.replace(second=59)
        return dt  # YYYY-MM-DD HH:MM
    return dt

def build_date_range(start_date: str, end_date: str):
    """Return (range_start, range_end) that always forms a valid interval."""
    if not start_date and not end_date:
        return None, None
    
    range_start = parse_datetime_utc_flexible(start_date) if start_date else None
    range_end = parse_datetime_utc_flexible(end_date) if end_date else None

    if range_start:
        range_start = expand_start(range_start, start_date)
    if range_end:
        range_end = expand_end(range_end, end_date, start_date)

    print(range_start, range_end)

    return range_start, range_end