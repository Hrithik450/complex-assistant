from langchain.tools import tool
from lib.dataframe import df

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
    limit: int = 10
) -> str:
    """
    This tool filter emails based on metadata such as sender, recipient, date range, subject, or thread ID.
    
    Args:
        sender (str, optional): Filter by sender email address (case-insensitive).
        recipient (str, optional): Filter by recipient email address (case-insensitive, supports list of recipients).
        start_date (str, optional): Start date (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS) for filtering emails.
        end_date (str, optional): End date (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS) for filtering emails.
        subject (str, optional): Filter by subject keyword (case-insensitive).
        threadId (str, optional): Filter by specific thread ID.
        sort_by (str, optional): Column to sort by (default is 'date').
        sort_order (str, optional): 'asc' for ascending, 'desc' for descending (default is 'desc').
        limit (int, optional): Number of recent results to return (default is 10).
    """

    print(f"metadata_filtering_tool is being called {sender}, {recipient}, {start_date}, {end_date}, {subject}, {threadId}, {sort_by}, {sort_order}, {limit}")
    temp_df = df.copy()
    
    # --- Sender filter (case-insensitive, matches name or email) ---
    if sender:
        temp_df = temp_df[temp_df['from'].astype(str).str.contains(sender, case=False, na=False)]

    # --- Recipient filter ---
    if recipient:
        temp_df = temp_df[temp_df['to'].apply(
            lambda x: any(recipient.lower() in str(t).lower() for t in (x if isinstance(x, list) else [x]))
        )]

    # --- Date filtering (normalize to datetime) ---
    if start_date:
        try:
            temp_df = temp_df[temp_df['date'] >= start_date]
        except Exception as e:
            return f"Error parsing start_date: {e}"

    if end_date:
        try:
            temp_df = temp_df[temp_df['date'] <= end_date]
        except Exception as e:
            return f"Error parsing end_date: {e}"

    # --- Subject filter ---
    if subject:
        temp_df = temp_df[temp_df['subject'].astype(str).str.contains(subject, case=False, na=False)]

    # --- ThreadId filter ---
    if threadId:
        temp_df = temp_df[temp_df['threadId'] == threadId]
  
    # --- Handle empty result ---
    if temp_df.empty:
        return "No emails found matching the specified criteria."
    
    # --- Sorting ---
    ascending = (sort_order.lower() == "asc")
    if sort_by not in temp_df.columns:
        sort_by = "date"
    temp_df = temp_df.sort_values(by=sort_by, ascending=ascending)

    # --- Total count ---
    total_matches = len(temp_df)

    # --- Preview results ---
    results_preview = temp_df.head(limit)[['threadId', 'from', 'to', 'subject', 'date']].to_dict('records')
        
    formatted_results = "\n\n---\n\n".join([
        f"threadId: {res.get('threadId', 'N/A')}\n"
        f"From: {res.get('from', 'N/A')}\n"
        f"To: {res.get('to', 'N/A')}\n"
        f"Subject: {res.get('subject', 'N/A')}\n"
        f"Date: {res.get('date', 'N/A').strftime('%Y-%m-%d %H:%M:%S') if res.get('date') else 'N/A'}"
        for res in results_preview
    ])

    return f"Found a total of {total_matches} emails matching the criteria. Here are the {min(limit, total_matches)} most relevant:\n\n{formatted_results}"