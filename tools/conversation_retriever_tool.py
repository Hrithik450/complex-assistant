from lib.dataframe import df
from langchain.tools import tool

@tool("conversation_retriever_tool", parse_docstring=True)
def conversation_retriever_tool(threadId: str) -> str:
    """
    This tool retrieves the full history of an email conversation based on a specific thread ID.
    
    Args:
        threadId (str): The unique identifier of the email thread to retrieve.
    
    Returns:
        str: Full conversation history including metadata and email content.
    """
    print('conversation_retriever_tool is being called')
    thread_df = df[df['threadId'] == threadId].sort_values(by='date')

    if thread_df.empty:
        return f"No conversation found with threadId '{threadId}'."

    full_conversation = "\n\n---\n\n".join([
        f"threadId: {row['threadId']}\n"
        f"From: {row['from']}\n"
        f"To: {row['to']}\n"
        f"Subject: {row['subject']}\n"
        f"Date: {row['date'].strftime('%Y-%m-%d %H:%M')}\n\n"
        f"{row['original_text']}"
        for _, row in thread_df.iterrows()
    ])

    return full_conversation