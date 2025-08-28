import pandas as pd
from lib.dataframe import df
from langchain.tools import tool

@tool("metadata_filtering_tool", parse_docstring=True)
def metadata_filtering_tool(sender: str = None, recipient: str = None, start_date: str = None, end_date: str = None, subject: str = None, threadId: str = None) -> str:
    """
    This tool filter emails based on metadata such as sender, recipient, date range, subject, or thread ID.
    
    Args:
        sender (str, optional): Filter by sender email address (case-insensitive).
        recipient (str, optional): Filter by recipient email address.
        start_date (str, optional): Start date (YYYY-MM-DD) for filtering emails.
        end_date (str, optional): End date (YYYY-MM-DD) for filtering emails.
        subject (str, optional): Filter by subject keyword (case-insensitive).
        threadId (str, optional): Filter by specific thread ID.

    Returns:
        str: Summary of total matching emails and up to 10 most recent results.
    """
    print('metadata_filtering_tool is being called')
    temp_df = df.copy()
    
    if sender:
        # Case-insensitive search for sender
        temp_df = temp_df[temp_df['from'].str.contains(sender, case=False, na=False)]
    if recipient:
        # Search if recipient is in the 'to' list (which can be a list of strings)
        temp_df = temp_df[temp_df['to'].apply(lambda x: any(recipient.lower() in t.lower() for t in x if isinstance(t, str)))]
    if start_date:
        temp_df = temp_df[temp_df['date'] >= pd.to_datetime(start_date)]
    if end_date:
        temp_df = temp_df[temp_df['date'] <= pd.to_datetime(end_date)]
    if subject:
        temp_df = temp_df[temp_df['subject'].str.contains(subject, case=False, na=False)]
    if threadId:
        temp_df = temp_df[temp_df['threadId'] == threadId]
    if temp_df.empty:
        return "No emails found matching the specified criteria."
        
    # --- FIX 1: Get the total count BEFORE slicing the dataframe ---
    total_matches = len(temp_df)
    
    # --- FIX 2: Sort the results by date in descending order ---
    temp_df = temp_df.sort_values(by='date', ascending=False)
    
    # Limit to the 10 most recent results for the preview
    results_preview = temp_df.head(10)[['threadId', 'from', 'to', 'subject', 'date']].to_dict('records')
    
    formatted_results = "\n\n---\n\n".join([
        f"threadId: {res.get('threadId', 'N/A')}\n"
        f"From: {res.get('from', 'N/A')}\n"
        f"To: {res.get('to', 'N/A')}\n"
        f"Subject: {res.get('subject', 'N/A')}\n"
        f"Date: {res.get('date', 'N/A').strftime('%Y-%m-%d') if res.get('date') else 'N/A'}"
        for res in results_preview
    ])

    # --- FIX 3: Return the total count along with the preview ---
    return f"Found a total of {total_matches} emails matching the criteria. Here are the 10 most recent:\n\n{formatted_results}"