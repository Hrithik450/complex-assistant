from langchain.tools import tool
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import polars as pl
from datetime import datetime

# Import the full email DataFrame
from lib.load_data import df

# Import helper functions
from lib.utils import normalize_list, match_value_in_columns, smart_subject_match

# Initialize the VADER analyzer once
analyzer = SentimentIntensityAnalyzer()

def classify_sentiment(score):
    """Classifies a VADER compound score into a category."""
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

@tool("sentiment_analysis_tool", parse_docstring=True)
def sentiment_analysis_tool(
    threadId: str = None,
    sender: str = None,
    recipient: str = None,
    subject: str = None,
    start_date: str = None,
    end_date: str = None,
    timeline_granularity: str = None
) -> str:
    """
    Analyzes the sentiment of emails. It can analyze a complete conversation thread or filter emails by metadata.
    If sender, recipient, and subject are provided, it will find the entire conversation thread and analyze it.
    If only a threadId is provided, it will analyze that specific thread.
    Otherwise, it will analyze emails matching the given filters.

    Args:
        threadId (str, optional): The unique ID of the email thread to analyze.
        sender (str or list of str, optional): Filter emails by sender(s). Can be full email address, partial email, or sender names (case-insensitive, only humans).
        recipient (str or list of str, optional): Filter emails by recipient(s). Can be full email addresses, partial emails, or recipient names, but strictly not numbers. (case-insensitive, only humans).
        subject (str, optional): Filter email by subject text. Can be full or partial subject string (case-insensitive).
        start_date (str, optional): The start date for the analysis (YYYY-MM-DD).
        end_date (str, optional): The end date for the analysis (YYYY-MM-DD).
        timeline_granularity (str, optional): If provided, groups sentiment by 'month' or 'year'.
    """
    print(f"sentiment_analysis_tool called with: threadId={threadId}, sender={sender}, recipient={recipient}, subject={subject}, timeline={timeline_granularity}")
    
    if df.is_empty():
        return "Error: Email data is not loaded."

    temp_df = df.clone()
    
    if not threadId and sender and recipient and subject:
        print("Attempting to find conversation threadId...")
        sender_lower = sender.lower()
        recipient_lower = recipient.lower()
        
        temp_df = temp_df.with_columns(
            pl.col("from").map_elements(normalize_list, return_dtype=str).alias("from_normalized"),
            pl.col("to").map_elements(normalize_list, return_dtype=str).alias("to_normalized"),
            pl.col("cc").map_elements(normalize_list, return_dtype=str).alias("cc_normalized")
        )
        
        a_to_b = (
            pl.col("from_normalized").map_elements(lambda x: match_value_in_columns(sender_lower, x), return_dtype=bool) &
            (pl.col("to_normalized").map_elements(lambda x: match_value_in_columns(recipient_lower, x), return_dtype=bool) |
             pl.col("cc_normalized").map_elements(lambda x: match_value_in_columns(recipient_lower, x), return_dtype=bool))
        )
        b_to_a = (
            pl.col("from_normalized").map_elements(lambda x: match_value_in_columns(recipient_lower, x), return_dtype=bool) &
            (pl.col("to_normalized").map_elements(lambda x: match_value_in_columns(sender_lower, x), return_dtype=bool) |
             pl.col("cc_normalized").map_elements(lambda x: match_value_in_columns(sender_lower, x), return_dtype=bool))
        )
        
        subject_mask = pl.col("subject").map_elements(lambda x: smart_subject_match(subject, x), return_dtype=bool)
        conversation_mask = (a_to_b | b_to_a) & subject_mask
        matching_emails = temp_df.filter(conversation_mask)
        
        if matching_emails.is_empty():
            return "No conversation thread found matching the specified sender, recipient, and subject."
        
        resolved_threadId = matching_emails.row(0, named=True).get("threadId")
        if not resolved_threadId:
            return "Could not identify a unique threadId for the conversation."
            
        print(f"Found threadId: {resolved_threadId}. Proceeding with analysis...")
        threadId = resolved_threadId

    analysis_df = df.clone()
    if threadId:
        analysis_df = analysis_df.filter(pl.col("threadId") == threadId)
    else:
        mask = pl.lit(True)
        if sender:
            mask = mask & pl.col("from").map_elements(lambda x: match_value_in_columns(sender.lower(), normalize_list(x)), return_dtype=bool)
        if recipient:
            recipient_mask = (
                pl.col("to").map_elements(lambda x: match_value_in_columns(recipient.lower(), normalize_list(x)), return_dtype=bool) |
                pl.col("cc").map_elements(lambda x: match_value_in_columns(recipient.lower(), normalize_list(x)), return_dtype=bool)
            )
            mask = mask & recipient_mask
        if subject:
            mask = mask & pl.col("subject").map_elements(lambda x: smart_subject_match(subject, x), return_dtype=bool)
        analysis_df = analysis_df.filter(mask)

    if analysis_df.is_empty():
        return "No emails found for the specified criteria."

    # Sort emails by date to show the conversation in order
    analysis_df = analysis_df.sort("date")

    if timeline_granularity or start_date or end_date:
        analysis_df = analysis_df.with_columns(
            pl.col("date").str.to_datetime("%Y-%m-%dT%H:%M:%SZ", strict=False).alias("date_dt")
        )
        if start_date:
            analysis_df = analysis_df.filter(pl.col("date_dt") >= datetime.strptime(start_date, "%Y-%m-%d"))
        if end_date:
            analysis_df = analysis_df.filter(pl.col("date_dt") <= datetime.strptime(end_date, "%Y-%m-%d"))

    safe_text_extraction_expr = (
        pl.when(pl.col("body").is_not_null())
        .then(pl.col("body").struct.field("text"))
        .otherwise(pl.lit(""))
    )
    
    # Add sentiment scores directly to the analysis dataframe
    analysis_df = analysis_df.with_columns(
        safe_text_extraction_expr.map_elements(
            lambda text: analyzer.polarity_scores(str(text or ""))['compound'],
            return_dtype=pl.Float64
        ).alias("sentiment_score")
    )

    sentiments = analysis_df.select(["sentiment_score", "date_dt" if "date_dt" in analysis_df.columns else pl.lit(None, dtype=pl.Datetime).alias("date")])

    if sentiments.is_empty():
        return "Found emails, but could not extract text to analyze sentiment."

    # --- Synthesize and Return Results ---
    # High-level summary (always calculated)
    overall_summary = sentiments.select(
        pl.mean("sentiment_score").alias("average_sentiment"),
        pl.col("sentiment_score").map_elements(classify_sentiment, return_dtype=pl.String).value_counts().alias("sentiment_counts"),
        pl.len().alias("total_emails")
    ).to_dicts()[0]

    avg_score = overall_summary['average_sentiment']
    total_emails = overall_summary['total_emails']
    counts_data = overall_summary['sentiment_counts']
    if isinstance(counts_data, dict):
        counts_data = [counts_data]
    counts = {d['sentiment_score']: d['count'] for d in counts_data}
    
    summary_title = f"Overall Sentiment Analysis Summary for Thread ID: {threadId}" if threadId else "Overall Sentiment Analysis Summary"
    summary = (
        f"{summary_title}\n"
        f"- Total Emails Analyzed: {total_emails}\n"
        f"- Average Sentiment Score: {avg_score:.2f} ({classify_sentiment(avg_score)})\n"
        f"- Positive Emails: {counts.get('Positive', 0)}\n"
        f"- Negative Emails: {counts.get('Negative', 0)}\n"
        f"- Neutral Emails: {counts.get('Neutral', 0)}"
    )

    # --- NEW: Detailed Breakdown by Email ---
    breakdown_lines = ["\n---", "Sentiment Breakdown by Email (chronological):"]
    for email in analysis_df.iter_rows(named=True):
        score = email['sentiment_score']
        sentiment_class = classify_sentiment(score)
        # Format the 'from' field for better readability
        sender_name = str(email.get('from', 'N/A')).split('<')[0].strip().replace('"', '')
        
        line = (
            f"\n- From: {sender_name}\n"
            f"  Date: {email.get('date')}\n"
            f"  Snippet: {email.get('snippet', 'N/A')}\n"
            f"  Sentiment: {sentiment_class} (Score: {score:.2f})"
        )
        breakdown_lines.append(line)

    # Combine the summary and the breakdown
    return summary + "\n" + "\n".join(breakdown_lines)
