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
    print(f"conversation_retriever_tool is being called {threadId}")
    thread_df = df[df['threadId'] == threadId]
    
    if thread_df.empty:
        return f"No conversation found with threadId '{threadId}'."

    full_conversation = "\n\n---\n\n".join([
        f"threadId: {row['threadId']}\n"
        f"From: {row['from']}\n"
        f"To: {row['to']}\n"
        f"CC: {row['cc']}\n"
        f"Subject: {row['subject']}\n"
        f"Date: {row['date']}\n\n"
        f"Snippet: {row['snippet']}\n\n"
        f"Body:\n{row['body']}\n\n"
        f"Labels: {row['labels']}\n"
        f"Attachments: {row['attachments']}\n"
        for _, row in thread_df.iterrows()
    ])

    return full_conversation