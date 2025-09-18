from lib.utils import normalize_list, match_value_in_columns, smart_subject_match, get_best_match_from_token_map
from lib.load_data import df, token_map
from langchain.tools import tool
from datetime import datetime, timedelta
import polars as pl

@tool("email_filtering_tool", parse_docstring=True)
def email_filtering_tool(
    uid: str = None,
    threadId: str = None,
    sender: str = None,
    recipient: str = None,
    subject: str = None,
    cc: bool = False,
    labels: list[str] = None,
    start_date: str = None,
    end_date: str = None,
    body: bool = False,
    html: bool = False,
    sort_by: str = "date",
    sort_order: str = "desc",
    limit: int = 5
) -> str:
    """
    This tool filter emails based on metadata such as sender (human), recipient (human), date range, or thread ID.
    
    Args:
        uid (str, optional): Filter emails by their unique UID. Exact match required.
        threadId: Filter emails by their conversation (email chian) thread ID, Returns all messages belonging to that specific chain (thread).
        sender (str or list of str, optional): Filter emails by sender(s). Can be full email address, partial email, or sender names (case-insensitive, only humans).
        recipient (str or list of str, optional): Filter emails by recipient(s). Can be full email addresses, partial emails, or recipient names, but strictly not numbers. (case-insensitive, only humans).
        subject (str, optional): Filter email by subject text. Can be full or partial subject string (case-insensitive).
        cc (bool, optional): Filter cc recepients of the email only when explicitly requested. Default False.
        labels (list of str, optional): Filter emails by one or more labels. Matches any email that contains at least one of the provided labels (case-insensitive).
        start_date (str, optional): Filter emails sent on or after this date. Format: 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'.
        end_date (str, optional): Filter emails sent on or before this date. Format: 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'.
        body (bool, optional): Include the plain-text email body only when explicitly requested. Default False.
        html (bool, optional): Include the full HTML body only when explicitly requested. Default False.
        sort_by (str, optional): Column to sort the results by. Default is 'date'.
        sort_order (str, optional): Sort order: 'asc' for ascending, 'desc' for descending. Default is 'desc'.
        limit (int, optional): Maximum number of results to return. Default is 10.
    """

    print(f"email_filtering_tool is being called {uid}, {threadId}, {sender}, {recipient}, {subject}, {cc}, {labels}, {start_date}, {end_date}, {body}, {html}, {sort_by}, {sort_order}, {limit}")
    temp_df = df.clone()
    mask = pl.lit(True)

    if uid:
        mask = mask & (pl.col("id") == uid)

    if threadId:
        mask = mask & (pl.col("threadId") == threadId)
    
    # --- Sender filter (case-insensitive, matches name or email) ---
    if sender:
        sender = sender.lower()
        response = get_best_match_from_token_map(sender, token_map, threshold=75)
        print(response, "optimized sender")
        if response:
            best_full_name, _ = response
            sender = best_full_name
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
        response = get_best_match_from_token_map(recipient, token_map, threshold=75)
        print(response, "optimized recipient")
        if response:
            best_full_name, _ = response
            recipient = best_full_name
        # Normalize 'to' and 'cc' columns which are lists
        temp_df = temp_df.with_columns([
            pl.col("to").map_elements(normalize_list, return_dtype=str).alias("to_normalized")
        ])
        # Filter rows where any normalized 'to' or 'cc' matches the recipient
        recipient_mask = (
            pl.col("to_normalized").map_elements(lambda x: match_value_in_columns(recipient, x), return_dtype=bool)
        )
        if cc:
            # Normalize 'to' and 'cc' columns which are lists
            temp_df = temp_df.with_columns([
                pl.col("cc").map_elements(normalize_list, return_dtype=str).alias("cc_normalized")
            ])
            # Filter rows where any normalized 'to' or 'cc' matches the recipient
            cc_mask = (
                pl.col("cc_normalized").map_elements(lambda x: match_value_in_columns(recipient, x), return_dtype=bool)
            )
            recipient_mask = recipient_mask | cc_mask

        mask = mask & recipient_mask

    # --- Date filtering (normalize to datetime) ---
    if start_date or end_date:
        temp_df = temp_df.with_columns([
            pl.col("date").str.strptime(pl.Datetime, format="%Y-%m-%dT%H:%M:%SZ", strict=False).alias("date_dt")
        ])
        
    if start_date:
        try:
            start_date_dt = datetime.strptime(start_date, "%Y-%m-%d")
            mask = mask & (pl.col("date_dt") >= start_date_dt)
        except Exception as e:
            return f"Error parsing start_date: {e}"

    if end_date:
        try:
            end_date_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1) - timedelta(seconds=1)
            mask = mask & (pl.col("date_dt") <= end_date_dt)
        except Exception as e:
            return f"Error parsing end_date: {e}"
<<<<<<< Updated upstream
        
    if labels: 
        labels = [lbl.strip().lower() for lbl in labels]

        temp_df = temp_df.with_columns([
            pl.col("lables").map_elements(normalize_list, return_dtype=str).alias("labels_normalized")
        ])

        labels_mask = pl.col("labels_normalized").map_elements(
            lambda email_lables: any(lbl in email_lables for lbl in labels),
            return_dtype=bool
        )

        mask = mask & labels_mask

    if subject:    
        subject_mask = pl.col("subject").map_elements(lambda x: smart_subject_match(subject, x) or match_value_in_columns(subject, x), return_dtype=bool)
        mask = mask & subject_mask

    # Apply the mask only once
    temp_df = temp_df.filter(mask)

    # --- Sorting ---
    temp_df = temp_df.sort(
        by=sort_by,
        descending=(sort_order.lower() == "desc")
    )
=======

    # Apply the mask only once
    temp_df = temp_df.filter(mask)
>>>>>>> Stashed changes

    # --- Handle empty result ---
    if temp_df.is_empty():
        return "No emails found matching the specified criteria."
<<<<<<< Updated upstream

    # --- Preview results ---
    total_matches = temp_df.height
    preview_cols = ["id", "threadId", "from", "to", "subject", "date", "cc", "snippet", "labels", "attachments"]
    if body:
        preview_cols.append("body")
    if html:
        preview_cols.append("html")
=======

    # --- Sorting ---
    ascending = (sort_order.lower() == "asc")
    if sort_by not in temp_df.columns:
        sort_by = "date"
    temp_df = temp_df.sort(sort_by, descending=not ascending)

    # --- Total count ---
    total_matches = temp_df.height

    # --- Preview results ---
    results_preview = temp_df.head(limit).select(['threadId', 'from', 'to', 'subject', 'date', 'cc']).to_dicts()
    print()

    formatted_results = "\n\n---\n\n".join([
        f"threadId: {res.get('threadId', 'N/A')}\n"
        f"From: {res.get('from', 'N/A')}\n"
        f"To: {res.get('to', 'N/A')}\n"
        f"CC: {res.get('cc', 'N/A')}\n"
        f"Subject: {res.get('subject', 'N/A')}\n"
        f"Date: {format_date(res.get('date'))}"
        for res in results_preview
    ])
>>>>>>> Stashed changes

    results_preview = temp_df.head(limit).select(preview_cols).to_dicts()

    def fmt(res):
        parts = [
            f"id: {res.get('id','N/A')}",
            f"ThreadId: {res.get('threadId','N/A')}",
            f"From: {res.get('from','N/A')}",
            f"To: {res.get('to','N/A')}",
            f"CC: {res.get('cc','N/A')}",
            f"Subject: {res.get('subject','N/A')}",
            f"Date: {res.get('date','N/A')}",
            f"Snippet: {res.get('snippet','N/A')}",
            f"Labels: {res.get('labels','N/A')}",
            f"Attachments: {res.get('attachments','N/A')}",
        ]
        if body:
            parts.append(f"Body:\n{res.get('body','N/A')}")
        if html:
            parts.append(f"HTML:\n{res.get('html','N/A')}")
        return "\n".join(parts)

    formatted_results = "\n\n---\n\n".join(fmt(r) for r in results_preview)
    return f"Found {total_matches} emails matching the criteria. Showing {min(limit, total_matches)}:\n\n{formatted_results}"