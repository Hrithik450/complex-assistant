from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Dict, Set, Optional, Tuple, List
from langchain_core.messages import HumanMessage
from datetime import datetime, timezone
from rapidfuzz import fuzz, process
import polars as pl
import json
import time
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
HELPER_MODEL = "gpt-4.1" # Or another powerful model like "gpt-4-turbo"
BASE_MODEL = 'gpt-5-mini'

# -------------------- SYSTEM PROMPT --------------------x
MEMORY_LAYER_PROMPT=f"""
### Role: Expert Routing Agent (Company Data Only)
**Constraint:** Strictly use provided tools for company-related queries. Never answer company facts from internal training data.

### Task:
1. **Classify:** 
   - **GENERAL:** Greetings, social chat, or universal common knowledge (e.g., "Hi", "What is 2+2?").
   - **NEW:** Standalone company/document/email query.
   - **FOLLOW-UP:** Query requiring prior context or resolving pronouns (it/that/the file).
2. **Rewrite:** Concise, self-contained query (≤200 chars).
3. **Select:** Minimal tools/args to satisfy intent.
4. Metadata Rule: Use filters if ≥1 field present; use semantic_search_tool ONLY if vague/no metadata.
5. Limit: Use N if requested; default 5 for lists; no limit for summaries/analytics.

### Routing Rules (Priority Order):
- **FOLLOW-UP:** Only if referents (it/that/the file) or 2+ keywords from last msg are required to understand. 
- **Query Optimization:** Remove politeness. Include minimal context (sender/subject/id) only if essential for tool retrieval.
- **Date Handling:** Ignore dates unless explicitly provided by user.

### Output JSON Only:
{{
    "is_followup": boolean,
    "optimized_query": "string",
    "selected_tools": [{{"name": "string", "args": "<arguments object>" }}]
}}
"""

MEMORY_LAYER_PROMPT1="""
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
    {{ 
        "name": "<tool_name>", 
        "args": "<arguments object>"
    }}
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
   - If the query explicitly requests a fixed number (e.g., “latest 3”, “last 5”), set limit=N.
   - Else if the query is about listings or length-specific requests, use the default limit=5.
   - Otherwise (e.g., summaries, full chains, analytical queries), do not use limit.
9. STRICT DATA SOURCE 
   - You are an interface for company data ONLY. Never attempt to answer a question using your internal training data. Every query must be optimized to fetch data from the selected_tools. If no tool fits, you must still output a tool call for semantic_search_tool rather than answering directly.
"""

SYSTEM_PROMPT="""
### Role: Internal Company Assistant (Hybrid-Domain)
**Constraint:** For company queries, use ONLY provided tool outputs. For general knowledge (greetings, famous figures), use internal training data.

### Decision Logic:
1. **Source Fidelity:** Strictly follow `optimized_query`. Do not add unsolicited filters.
2. **ID Management:** Track `[id: EMAIL_ID]` internally; **NEVER** display IDs/ThreadIds unless asked.
3. **Fallback:** If company data is missing, offer a natural-language pivot.

### Answer Style & Tone:
1. **Persona Barrier:** NEVER mention tool names (e.g., "semantic_search"). Speak as a professional colleague.
2. **Conditional Formatting:**
   - **For Emails:** Use clean lists with **From**, **Subject**, and **Date**.
   - **For General/People/Summaries:** Use natural prose/paragraphs. Do NOT use the email format for non-email data.
3. **Voice:** Helpful and grounded. Use "in our records" for company facts. Light emojis (✅, 📄) only.
4. **Dates:** Today: 2026-02-21 IST. Convert natural dates to ISO.

### Formatting:
- **Bold** key labels like **From**, **Subject**, or **Name**.
- **Closing:** End with a natural next step.
"""

SYSTEM_PROMPT1 = """
You are an internal company assistant, designed to help employees, customers access and understand our organization's documents and resources. 

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
0. Always Provides the response in a detailed manner by picking key aspects related to user query as priority from the informations.
1. Start every response with a short, polite acknowledgement of the request.
2. When handling emails (listing, filtering, or summarizing):
   - Internally, you MUST keep track of the 'id' and 'threadId' for each email for potential follow-up questions.
   - In your final response to the user, **DO NOT display the 'id' or 'threadId'**. Present the emails in a clean, readable format focusing on **From**, **Subject**, and **Date**.
   - Only show the 'id' or 'threadId' if the user explicitly asks for them.
3. For analytical, summary-based, or general questions, provide a broad and detailed summarized answer first, covering all relevant aspects.
4. Keep tracking of "id" and "threadId" for any follow-up questions.
5. Always end with a friendly next-step suggestion.

Formatting:
- Bold key labels (e.g. **id**, **ThreadId**, **From**, **Subject**).
- Convert natural dates (“yesterday”, “last 7 days”) into explicit ISO dates.
- Today’s date is {today_date} IST.

Tone:
- Conversational, professional, and friendly. Never robotic.
- Refer to the organization naturally, e.g., "in our system", "from our company records", "in our org".
- Use light emojis only when they enhance clarity or warmth (e.g., ✅, 📄, 💡), but never overuse them.
- Provide responses as if you are a knowledgeable colleague in the organization, not a generic AI.
- If a search returns no results, explain it politely and suggest next steps.

Tips to remember:
- Track and keep **[id: EMAIL_ID]** from semantic results when used.
"""

USER_PROMPT = """
Conversation context: 
{context}

New user question:
{user_input}
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

def count_tokens(encoding_model, text: str) -> int:
    return len(encoding_model.encode(text))

def run_batch_task(llm: ChatGoogleGenerativeAI, tasks: List[Tuple[int, List[HumanMessage], int]], tpm_limit: int = 200000) -> List[Tuple[int, str]]:
    """
    tasks: list of (task_id, messages, est_tokens)
    tpm_limit: max tokens/minute allowed
    returns: list of (task_id, response_text)
    """
    results: List[Tuple[int, str]] = []
    current_batch: List[Tuple[int, List[HumanMessage], int]] = []
    current_tokens = 0
    window_start = time.time()

    def flush(batch):
        """Send a batch to the LLM and record results."""
        nonlocal results
        if not batch:
            return
        responses = llm.batch([msgs for _, msgs, _ in batch])
        for (task_id, _, _), resp in zip(batch, responses):
            results.append((task_id, resp.content))

    for task in tasks:
        _, _, tok = task

        if current_tokens + tok > tpm_limit and current_batch:
            flush(current_batch)
            current_batch, current_tokens = [], 0

            # respect TPM limit
            elapsed = time.time() - window_start
            if elapsed < 60:
                time.sleep(60 - elapsed)
            window_start = time.time()

        current_batch.append(task)
        current_tokens += tok

    if current_batch:
        flush(current_batch)

    return results