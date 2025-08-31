from langchain.tools import tool
from lib.load_data import df
from rapidfuzz import fuzz
from datetime import datetime
import polars as pl
import re

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
            if value in e or fuzz.partial_ratio(value.lower(), e.lower()) > 60:
                return True
        return False

    # Case 2: column_value is a string
    if isinstance(column_value, str):
        return value in column_value or fuzz.partial_ratio(value.lower(), column_value.lower()) > 60

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

@tool("metadata_filtering_tool", parse_docstring=True)
def metadata_filtering_tool(
    sender: str = None,
    recipient: str = None,
    start_date: str = None,
    end_date: str = None,
    subject: str = None,
    threadId: str = None,
    sort_by: str = "date",
    sort_order: str = "desc",
    limit: int = 5
) -> str:
    """
    This tool filter emails based on metadata such as sender, recipient, date range, subject, or thread ID.
    
    Args:
        sender (str or list of str, optional): Filter emails by sender(s). Can be full email address, partial email, or sender names (case-insensitive).
        recipient (str or list of str, optional): Filter emails by recipient(s). Can be full email addresses, partial emails, or recipient names (case-insensitive).
        start_date (str, optional): Filter emails sent on or after this date. Format: 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'.
        end_date (str, optional): Filter emails sent on or before this date. Format: 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'.
        subject (str, optional): Filter emails containing this keyword in the subject (case-insensitive, partial match supported).
        threadId (str, optional): Filter emails belonging to a specific thread ID.
        sort_by (str, optional): Column to sort the results by. Default is 'date'.
        sort_order (str, optional): Sort order: 'asc' for ascending, 'desc' for descending. Default is 'desc'.
        limit (int, optional): Maximum number of results to return. Default is 10.
    """

    print(f"metadata_filtering_tool is being called {sender}, {recipient}, {start_date}, {end_date}, {subject}, {threadId}, {sort_by}, {sort_order}, {limit}")
    temp_df = df.clone()

    mask = pl.lit(True)
    
    # --- Sender filter (case-insensitive, matches name or email) ---
    if sender:
        sender = sender.lower()
        # Add a normalized column
        temp_df = temp_df.with_columns([
            pl.col("from").map_elements(normalize_email_field, return_dtype=list[str]).alias("from_normalized")
        ])
        # Filter rows where the normalized 'from' matches sender
        sender_mask = pl.col("from_normalized").map_elements(lambda x: match_value_in_columns(sender, x), return_dtype=bool)
        mask = mask & sender_mask

    # --- Recipient filter ---
    if recipient:
        recipient = recipient.lower()
        # Normalize 'to' and 'cc' columns which are lists
        temp_df = temp_df.with_columns([
            pl.col("to").map_elements(normalize_list, return_dtype=str).alias("to_normalized"),
            pl.col("cc").map_elements(normalize_list, return_dtype=str).alias("cc_normalized")
        ])
        # Filter rows where any normalized 'to' or 'cc' matches the recipient
        recipient_mask = (
            pl.col("to_normalized").map_elements(lambda x: match_value_in_columns(recipient, x), return_dtype=bool) |
            pl.col("cc_normalized").map_elements(lambda x: match_value_in_columns(recipient, x), return_dtype=bool)
        )
        mask = mask & recipient_mask

    # --- Date filtering (normalize to datetime) ---
    if start_date:
        try:
            # temp_df = temp_df.filter(pl.col("date") >= start_date)
            mask = mask & (pl.col("date") >= start_date)
        except Exception as e:
            return f"Error parsing start_date: {e}"

    if end_date:
        try:
            # temp_df = temp_df.filter(pl.col("date") <= end_date)
            mask = mask & (pl.col("date") <= end_date)
        except Exception as e:
            return f"Error parsing end_date: {e}"
        
    # Apply the mask only once
    temp_df = temp_df.filter(mask)

    # --- Handle empty result ---
    if temp_df.is_empty():
        return "No emails found matching the specified criteria."
    
    # --- Sorting ---
    ascending = (sort_order.lower() == "asc")
    if sort_by not in temp_df.columns:
        sort_by = "date"
    temp_df = temp_df.sort(sort_by, descending=not ascending)

    # --- Total count ---
    total_matches = temp_df.height

    # --- Preview results ---
    results_preview = temp_df.head(limit).select(['threadId', 'from', 'to', 'subject', 'date', 'cc']).to_dicts()
        
    formatted_results = "\n\n---\n\n".join([
        f"threadId: {res.get('threadId', 'N/A')}\n"
        f"From: {res.get('from', 'N/A')}\n"
        f"To: {res.get('to', 'N/A')}\n"
        f"CC: {res.get('cc', 'N/A')}\n"
        f"Subject: {res.get('subject', 'N/A')}\n"
        f"Date: {format_date(res.get('date'))}"
        for res in results_preview
    ])

    # print(formatted_results, "formatted_results from metadata_filtering_tool")
    return f"Found a total of {total_matches} emails matching the criteria. Here are the {min(limit, total_matches)} most relevant:\n\n{formatted_results}"