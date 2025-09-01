from lib.utils import normalize_list, match_value_in_columns, format_date
from langchain.tools import tool
from datetime import datetime
from lib.load_data import df
import polars as pl

@tool("email_filtering_tool", parse_docstring=True)
def email_filtering_tool(
    sender: str = None,
    recipient: str = None,
    start_date: str = None,
    end_date: str = None,
    threadId: str = None,
    sort_by: str = "date",
    sort_order: str = "desc",
    limit: int = 5
) -> str:
    """
    This tool filter emails based on metadata such as sender (human), recipient (human), date range, or thread ID.
    
    Args:
        sender (str or list of str, optional): Filter emails by sender(s). Can be full email address, partial email, or sender names (case-insensitive, only humans).
        recipient (str or list of str, optional): Filter emails by recipient(s). Can be full email addresses, partial emails, or recipient names, but strictly not numbers. (case-insensitive, only humans).
        start_date (str, optional): Filter emails sent on or after this date. Format: 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'.
        end_date (str, optional): Filter emails sent on or before this date. Format: 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'.
        threadId (str, optional): Filter emails belonging to a specific thread ID.
        sort_by (str, optional): Column to sort the results by. Default is 'date'.
        sort_order (str, optional): Sort order: 'asc' for ascending, 'desc' for descending. Default is 'desc'.
        limit (int, optional): Maximum number of results to return. Default is 10.
    """

    print(f"metadata_filtering_tool is being called {sender}, {recipient}, {start_date}, {end_date}, {threadId}, {sort_by}, {sort_order}, {limit}")
    temp_df = df.clone()
    mask = pl.lit(True)
    
    # --- Sender filter (case-insensitive, matches name or email) ---
    if sender:
        sender = sender.lower()
        # Add a normalized column
        temp_df = temp_df.with_columns([
            pl.col("from").map_elements(normalize_list, return_dtype=str).alias("from_normalized")
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
    if start_date or end_date:
        temp_df = temp_df.with_columns([
            pl.col("date").str.strptime(pl.Datetime, format="%Y-%m-%dT%H:%M:%SZ", strict=False).alias("date_dt")
        ])
        
    if start_date:
        start_date_dt = datetime.strptime(start_date, "%Y-%m-%d")
        try:
            mask = mask & (pl.col("date_dt") >= start_date_dt)
        except Exception as e:
            return f"Error parsing start_date: {e}"

    if end_date:
        end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")
        try:
            mask = mask & (pl.col("date_dt") <= end_date_dt)
        except Exception as e:
            return f"Error parsing end_date: {e}"

    # Apply the mask only once
    temp_df = temp_df.filter(mask)

    # --- Sorting ---
    temp_df = (
        temp_df
        .with_columns(
            pl.col("date").str.strptime(pl.Datetime, format="%Y-%m-%dT%H:%M:%SZ", strict=False)
        )
        .sort("date", descending=True)
    )

    # --- Handle empty result ---
    if temp_df.is_empty():
        return "No emails found matching the specified criteria."

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