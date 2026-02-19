from lib.utils import normalize_list, match_value_in_columns, safe_get
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from lib.utils import AGENT_MODEL
from langchain.tools import tool
from lib.load_data import df
import polars as pl

# template = """
# You are an assistant that summarizes email conversations.
# Read the full thread and give a clear summary in plain English that anyone can understand.
# Rules:
# - Keep it long brief (detailed)
# - Focus on the main people, topic, and outcome.
# - Skip technical details, metadata, and signatures.
# - Combine the whole thread into one simple story.
# - Stay neutral and clear.
# """

# prompt_perspectives = ChatPromptTemplate.from_template(template)

# generate_summary = (
#     prompt_perspectives 
#     | ChatOpenAI(model=AGENT_MODEL, temperature=0) | StrOutputParser()
# )

#   id (str, optional): Unique mail id (fastest lookup if present).

@tool("conversation_retriever_tool", parse_docstring=True)
def conversation_retriever_tool(
    subject: str = None,
    sender: str = None,
    recipient: str = None,
    cc: str = None
) -> str:
    """
    Retrieves the full history of an email conversation based on given parameters.
    It will try to resolve the unique thread using id, or a combination of filters.

    Args:
        subject (str, optional): Email subject (supports partial match).
        sender (str, optional): Email address or name of sender (case-insensitive).
        recipient (str, optional): Email address or name of recipient (case-insensitive).
        cc (str, optional): Email address or name of a CC recipient (case-insensitive).

    Returns:
        str: Full conversation history including metadata and email content.
    """
    print(f"conversation_retriever_tool is being called {subject}, {sender}, {recipient}, {cc}")
    temp_df = df.clone()
    mask = pl.lit(True)

    # if id:
    #     mask = mask & (pl.col("id") == id)

    if sender:
        sender = sender.lower()
        # Add a normalized column
        temp_df = temp_df.with_columns([
            pl.col("from").map_elements(normalize_list, return_dtype=str).alias("from_normalized")
        ])
        # Filter rows where the normalized 'from' matches sender
        sender_mask = pl.col("from_normalized").map_elements(lambda x: match_value_in_columns(sender, x), return_dtype=bool)
        mask = mask & sender_mask

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

    # Step 3 → Apply filters only if still not narrowed
    if subject:
        subject = subject.lower()
        # Add a normalized column
        temp_df = temp_df.with_columns([
            pl.col("subject").map_elements(normalize_list, return_dtype=str).alias("subject_normalized")
        ])
        subject_mask = pl.col("subject_normalized").map_elements(lambda x: match_value_in_columns(subject, x), return_dtype=bool)
        mask = mask & subject_mask

    temp_df = temp_df.filter(mask)

    if temp_df.is_empty():
        return "No conversation found."

    resolved_id = temp_df[0, "id"]
    thread_df = temp_df.filter(pl.col("id") == resolved_id)

    # Build conversation text
    full_conversation = "\n\n--- EMAIL ---\n\n".join([
        f"ID: {safe_get(row, 'id')}\n"
        f"From: {safe_get(row, 'from')}\n"
        f"To: {safe_get(row, 'to')}\n"
        f"CC: {safe_get(row, 'cc')}\n"
        f"Subject: {safe_get(row, 'subject')}\n"
        f"Date: {safe_get(row, 'date')}\n\n"
        f"Snippet: {safe_get(row, 'snippet')}\n\n"
        f"Body:\n{safe_get(row, 'body')}\n\n"
        f"Labels: {safe_get(row, 'labels')}\n"
        f"Attachments: {safe_get(row, 'attachments')}\n"
        for row in thread_df.iter_rows(named=True)
    ])

    # Pass to summary generator
    # summary = generate_summary.invoke({
    #     "question": f"Summarize this email thread:\n{full_conversation}"
    # })

    return full_conversation