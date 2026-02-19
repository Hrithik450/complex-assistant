from langchain.tools import tool
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import polars as pl
from datetime import datetime, timedelta, timezone

# Import the full email DataFrame
from lib.load_data import df

# Import helper functions
from lib.utils import normalize_list, match_value_in_columns, smart_subject_match

# --- NEW: Import LLM for advanced analysis ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

# --- NEW: Initialize LLM and Prompt for rich sentiment analysis ---
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2, google_api_key=os.getenv("GOOGLE_API_KEY"))

sentiment_prompt_template = ChatPromptTemplate.from_template(
    """
    You are an expert in communication analysis. Based on the following email text(s), provide a detailed sentiment and emotional analysis.

    **Do not just say "positive" or "negative".** Instead, provide a structured breakdown covering the following points:

    1.  **Overall Atmosphere:** A one-sentence summary of the general feeling or mood of the conversation (e.g., "The conversation is tense and urgent," or "The atmosphere is collaborative and optimistic.").
    2.  **Key Emotions Detected:** List the primary emotions present (e.g., Frustration, Excitement, Confusion, Urgency, Appreciation).
    3.  **Positive Aspects:** Identify and quote 1-2 key phrases or sentences that are positive. Explain *why* they are positive (e.g., "Expresses gratitude," "Shows agreement").
    4.  **Negative Aspects:** Identify and quote 1-2 key phrases or sentences that are negative or indicate conflict/concern. Explain *why* they are negative (e.g., "Points out a problem," "Expresses disappointment").
    5.  **Final Classification:** Conclude with a final classification (Positive, Negative, Neutral, or Mixed).

    Here is the email content:
    ---
    {email_content}
    ---
    """
)

sentiment_analysis_chain = sentiment_prompt_template | llm | StrOutputParser()


# Initialize the VADER analyzer (can be kept for a quick numerical score if needed)
analyzer = SentimentIntensityAnalyzer()

def parse_datetime_utc(date_str: str) -> datetime:
    """
    Parse input date string and return a UTC-aware datetime object.
    Accepts 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'.
    """
    if len(date_str) == 10:  # date-only
        dt = datetime.strptime(date_str, "%Y-%m-%d")
    else:  # full datetime
        dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
    return dt.replace(tzinfo=timezone.utc)

def human_readable_date(timestamp) -> str:
    """
    Convert a timestamp to human-readable form.
    Accepts: str, datetime.datetime, or None
    """
    if timestamp is None:
        return "N/A"

    if not isinstance(timestamp, datetime):
        try:
            timestamp = datetime.fromisoformat(str(timestamp))
        except Exception:
            return "N/A"

    return timestamp.strftime("%a, %b %d, %Y %I:%M %p")

@tool("sentiment_analysis_tool", parse_docstring=True)
def sentiment_analysis_tool(
    threadId: str = None,
    sender: str = None,
    recipient: str = None,
    subject: str = None,
    start_date: str = None,
    end_date: str = None,
) -> str:
    """
    Analyzes the sentiment of emails using an advanced language model. It provides a detailed breakdown of the atmosphere, key emotions, and specific examples.
    It can analyze a complete conversation thread or filter emails by metadata.

    Args:
        threadId (str, optional): The unique ID of the email thread to analyze.
        sender (str or list of str, optional): Filter emails by sender(s).
        recipient (str or list of str, optional): Filter emails by recipient(s).
        subject (str, optional): Filter email by subject text.
        start_date (str, optional): The start date for the analysis (YYYY-MM-DD).
        end_date (str, optional): The end date for the analysis (YYYY-MM-DD).
    """
    print(f"sentiment_analysis_tool called with: threadId={threadId}, sender={sender}, recipient={recipient}, subject={subject}")

    if df.is_empty():
        return "Error: Email data is not loaded."

    # --- The existing filtering logic remains the same ---
    # (This section is omitted for brevity, but it's the same as your original file)
    temp_df = df.clone()
    mask = pl.lit(True)

    if threadId:
        mask = mask & (pl.col("threadId") == threadId)
    else:
        if sender:
            sender = sender.lower()
            temp_df = temp_df.with_columns(
                pl.col("from").map_elements(normalize_list, return_dtype=str).alias("from_normalized")
            )
            mask = mask & pl.col("from_normalized").map_elements(lambda x: match_value_in_columns(sender, x), return_dtype=bool)
        if recipient:
            recipient = recipient.lower()
            temp_df = temp_df.with_columns(
                pl.col("to").map_elements(normalize_list, return_dtype=str).alias("to_normalized"),
                pl.col("cc").map_elements(normalize_list, return_dtype=str).alias("cc_normalized")
            )
            mask = mask & (
                pl.col("to_normalized").map_elements(lambda x: match_value_in_columns(recipient, x), return_dtype=bool) |
                pl.col("cc_normalized").map_elements(lambda x: match_value_in_columns(recipient, x), return_dtype=bool)
            )
        if subject:
            mask = mask & pl.col("subject").map_elements(lambda x: smart_subject_match(subject, x), return_dtype=bool)

        temp_df = temp_df.with_columns(
            pl.col("date").str.to_datetime("%Y-%m-%dT%H:%M:%S%z", strict=False).dt.convert_time_zone("UTC").alias("date_dt")
        )
        if start_date:
            mask = mask & (pl.col("date_dt") >= parse_datetime_utc(start_date))
        if end_date:
            end_dt = parse_datetime_utc(end_date) + timedelta(days=1)
            mask = mask & (pl.col("date_dt") < end_dt)

    analysis_df = temp_df.filter(mask)

    if analysis_df.is_empty():
        return "No emails found for the specified criteria to analyze."

    # --- NEW: Prepare content for the LLM ---
    analysis_df = analysis_df.sort("date")

    # Extract text content from body, falling back to snippet
    def get_email_text(row):
        body_text = (row.get("body") or {}).get("text")
        return body_text if body_text else row.get("snippet", "")

    email_texts = analysis_df.apply(get_email_text, return_dtype=str).to_list()
    full_conversation_text = "\n\n--- Next Email ---\n\n".join(email_texts)

    # Limit context to avoid excessive token usage for very long threads
    # (This is a simple truncation; the next section discusses a better way)
    if len(full_conversation_text) > 20000:
        full_conversation_text = full_conversation_text[:20000] + "\n\n[Content truncated due to length]"

    # --- NEW: Call the LLM for rich analysis ---
    try:
        analysis_result = sentiment_analysis_chain.invoke({"email_content": full_conversation_text})
        return f"Found {analysis_df.height} emails. Here is the detailed sentiment analysis:\n\n{analysis_result}"
    except Exception as e:
        return f"An error occurred during sentiment analysis: {e}"